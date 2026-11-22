"""End-to-end audit of one recording: video (+ key log, + transcript or audio) -> findings + report.

    python -m a11y_replay.audit RECORDING_DIR [--out REPORT_DIR] [--model claude-sonnet-5-5] [--transcribe]

A recording directory contains:
    video.mp4|webm     required
    keys.jsonl         optional; enables clock sync and the focus check
    utterances.json    optional transcript of what the screen reader said ([{"start","end","text"}])
    audio.wav          optional; with --transcribe (or a cached transcript_asr.json) it is recognized

It deliberately uses ONLY these inputs. It does not read labels.json or any benchmark metadata, so an audit
of a benchmark sample is a fair test. It is a screening aid, not a compliance test.

Checks: focus visible (2.4.7), flashing (2.3.1), status messages seen vs heard (4.1.3). Pause/stop/hide
(2.2.2) and keyboard traps (2.1.2) are NOT implemented and are listed as such in every report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from a11y_replay import __version__
from a11y_replay.cascade import run_focus_cascade
from a11y_replay.config import DEFAULT, Config
from a11y_replay.detectors.announce import match_observations, observe_changes
from a11y_replay.detectors.flash import detect_flashes
from a11y_replay.events import Utterance
from a11y_replay.ingest.keys import load_keys
from a11y_replay.ingest.video import (
    blank_stretches,
    is_variable_frame_rate,
    normalize_to_cfr,
    probe,
    read_frame_at,
)
from a11y_replay.segment import detect_changes
from a11y_replay.sync import estimate_offset, estimate_offset_search
from a11y_replay.vlm.client import VLMClient
from a11y_replay.vlm.tasks import crop_region

NOT_IMPLEMENTED = [
    ("Pause, stop, hide moving content (WCAG 2.2.2)", "no detector yet"),
    ("No keyboard trap (WCAG 2.1.2)", "no detector yet"),
]


@dataclass
class ReportItem:
    id: str
    status: str                    # "finding" | "needs_review" | "pass" (only with include_passes)
    criterion: str
    title: str
    start: float                   # seconds on the key-log clock
    end: float
    method: str                    # "classical" | "vlm"
    reasoning: str
    confidence: float | None = None
    evidence: list = field(default_factory=list)   # [{"caption", "file"}]


@dataclass
class AuditReport:
    recording: str
    created: str
    version: str
    sync: dict
    checks: list
    items: list
    stats: dict
    out_dir: str = ""
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def findings(self):
        return [i for i in self.items if i.status == "finding"]

    @property
    def needs_review(self):
        return [i for i in self.items if i.status == "needs_review"]

    @property
    def passes(self):
        return [i for i in self.items if i.status == "pass"]


# ------------------------------------------------------------------ evidence images
def _fit_height(img: np.ndarray, h: int = 220) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def _strip(imgs: list) -> np.ndarray:
    imgs = [_fit_height(i) for i in imgs if i is not None]
    gap = np.full((imgs[0].shape[0], 8, 3), 255, np.uint8)
    out = imgs[0]
    for i in imgs[1:]:
        out = np.hstack([out, gap, i])
    return out


def _save(out_dir: Path, name: str, img: np.ndarray) -> str:
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
    rel = f"evidence/{name}.png"
    cv2.imwrite(str(out_dir / rel), img)
    return rel


def _frame_at(video, idx_or_t, fps: float, is_frame: bool):
    return read_frame_at(video, idx_or_t / fps if is_frame else idx_or_t)


# ------------------------------------------------------------------ the audit
def run_audit(recording_dir, out_dir=None, client: VLMClient | None = None, cfg: Config = DEFAULT,
              transcribe: bool = False, whisper_model: str = "base", transcript_clock: str = "log",
              include_passes: bool = False, progress=None) -> AuditReport:
    """transcript_clock: which clock the transcript's times are on. "log" = the key log's (benchmark
    transcripts); "video" = the video's own timeline (e.g. a screen reader recorded into the screen
    recording). A transcript recognized from the video's own audio track is always on the video clock.
    include_passes: also list what was judged fine (with evidence), so a reader can check the tool's passes as
    well as its findings. progress: optional callback progress(fraction 0-1, message) for a UI."""
    def _p(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    _p(0.02, "Opening the recording")
    rec = Path(recording_dir)
    videos = sorted(rec.glob("video.*"), key=lambda p: p.stat().st_mtime, reverse=True)   # newest first
    video = videos[0] if videos else None
    if video is None:
        raise FileNotFoundError(f"No video.* in {rec}")
    out = Path(out_dir) if out_dir else rec / "audit_report"
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "evidence", ignore_errors=True)       # never leave pictures from an earlier run beside a new report
    original_video, vfr_note = video, ""
    if is_variable_frame_rate(video):
        # Screen recorders write a frame only when the screen changes, so frame number / fps is NOT time. Analyze a
        # constant-rate copy instead (the original stays untouched and is still used for its audio track).
        fixed = out / "analysis_video.mp4"
        if normalize_to_cfr(video, fixed):
            video = fixed
            vfr_note = (" The recording has a variable frame rate (typical of screen recordings), so a constant 25 fps copy "
                        "was analyzed so that frame numbers map to real time.")
        else:
            vfr_note = (" The recording has a variable frame rate and could not be converted (is ffmpeg installed?), "
                        "so timing may be wrong.")
    info = probe(video)
    fps = info.fps or 25.0
    warnings: list = []
    if len(videos) > 1:
        warnings.append(f"Several video files were found ({', '.join(v.name for v in videos)}). The newest, {video.name}, was "
                        "analyzed. Remove the others so there is no doubt which recording a report is about.")
    black, _dur = blank_stretches(video, cfg.blank_mean_luma, cfg.blank_p995_luma, cfg.blank_min_stretch_s)
    black_s = sum(b - a for a, b in black)
    if info.duration and (black_s / info.duration > cfg.blank_warn_frac or any(b - a >= 3.0 for a, b in black)):
        longest = max(black, key=lambda ab: ab[1] - ab[0])
        warnings.append(
            f"About {black_s:.0f} s of the {info.duration:.0f} s recording ({black_s / info.duration:.0%}) is black; the longest "
            f"stretch is {longest[1] - longest[0]:.0f} s from {longest[0]:.0f} s. The browser window probably left the recorded "
            "area (another Space or full-screen app, a minimized window, a locked screen). Results from black stretches are "
            "unreliable, and checks that read 'nothing changed' as a failure are reported as unreadable instead.")
    items: list[ReportItem] = []
    checks: list[dict] = []
    stats = {"video_seconds": round(info.duration, 1), "model": getattr(client, "model", None),
             "video_file": original_video.name,
             "video_modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(original_video.stat().st_mtime))}

    _p(0.08, "Lining up the key log with the video")
    # ---- clock sync (from the recording itself) ----
    keys = load_keys(rec / "keys.jsonl") if (rec / "keys.jsonl").exists() else []
    if keys:
        all_changes = detect_changes(video, cfg)
        est = estimate_offset(keys, all_changes, cfg)
        searched = False
        if not est.ok:        # video and key log may have been started by hand, seconds apart
            wide = estimate_offset_search(keys, all_changes, cfg)
            if wide.ok:
                est, searched = wide, True
        offset = est.offset if est.n_pairs else 0.0
        calib = sum(1 for n, _t, _d, frac in est.detail if n == cfg.calibration_key and frac <= cfg.marker_max_frac)
        sync = {"offset_s": round(offset, 3), "trusted": est.ok, "pairs": est.n_pairs, "searched": searched,
                "calibration_pairs": calib, "spread_s": None if not est.n_pairs else round(est.spread, 3),
                "detail": [{"key": n, "key_time_s": t, "delta_s": d, "change_frac": m} for n, t, d, m in est.detail]}
        if est.ok and searched:
            sync["note"] = (f"The recording and the key log started about {abs(offset):.1f} s apart; the offset was "
                            "found by searching all key/change pairs, then refined.")
        elif est.ok:
            sync["note"] = "Video and key log aligned from key presses that cause a visible change."
        else:
            sync["note"] = ("Clock sync could NOT be established reliably (too few key presses with a visible, "
                            "consistent effect); timing-based checks (focus, announcements) may be wrong.")
    else:
        offset = 0.0
        sync = {"offset_s": 0.0, "trusted": False, "pairs": 0, "spread_s": None, "searched": False,
                "note": "No key log (keys.jsonl): clock sync, the focus check and key-based exclusions are unavailable."}

    # When keys exist but the clock alignment could not be trusted, results that depend on timing (focus, announcements)
    # may be comparing the wrong frames. They are reported for review, never as confident findings.
    timing_doubtful = bool(keys) and not sync["trusted"]
    DOUBT = ("The clock alignment between the key log and the video is not trusted, so the frames compared may not be "
             "the ones around this event. ")
    if keys and sync["trusted"] and sync.get("calibration_pairs", 0) == 0:
        sync["note"] += (" None of the calibration flashes was matched (was the recording started after the logger?), so the "
                         "alignment came only from key presses that visibly change the screen. For the focus check that is "
                         "partly circular: it cannot align a page whose Tab presses change nothing, and a change after every Tab "
                         "is then close to guaranteed.")
    sync["note"] += vfr_note
    _p(0.30, "Checking for flashing")
    # ---- flashing (2.3.1) ----
    flashes = detect_flashes(video, cfg, offset=offset)
    for n, f in enumerate(flashes, 1):
        frames = [_frame_at(video, i, fps, True) for i in f.evidence_frames]
        ev = [{"caption": "Start, middle and end of the flashing", "file": _save(out, f"flash{n}", _strip(frames))}]
        items.append(ReportItem(f"flash-{n}", "finding", f.criterion, "Flashing content", f.start, f.end,
                                f.method, f.reasoning, f.confidence, ev))
    checks.append({"name": "Three flashes or below threshold (WCAG 2.3.1)", "status": "ran",
                   "detail": f"{len(flashes)} flashing episode(s). General flash only; red flash not checked."})

    _p(0.50, "Checking keyboard focus")
    # ---- focus visible (2.4.7) ----
    focus_presses = [k for k in keys if k.key in cfg.focus_keys]
    if not keys:
        checks.append({"name": "Focus visible (WCAG 2.4.7)", "status": "skipped", "detail": "needs keys.jsonl"})
    elif not focus_presses:
        checks.append({"name": "Focus visible (WCAG 2.4.7)", "status": "skipped",
                       "detail": "no focus-key (Tab) presses in the key log"})
    else:
        casc = run_focus_cascade(video, keys, offset=offset, client=client, cfg=cfg)
        vcount = {"visible": 0, "not_visible": 0, "ambiguous": 0, "unreadable": 0}
        for n, r in enumerate(casc.results, 1):
            vcount[r.verdict] += 1
            if r.verdict == "visible" and not include_passes:
                continue
            c = r.classical
            before = _frame_at(video, c.before_frame, fps, True)
            after = _frame_at(video, c.after_frame, fps, True)
            crops = [crop_region(before, c.region, cfg), crop_region(after, c.region, cfg)] if before is not None and after is not None else []
            ev = [{"caption": "Before the key press, then after it", "file": _save(out, f"focus{n}", _strip(crops))}] if crops else []
            if r.verdict == "visible":
                items.append(ReportItem(f"focus-{n}", "pass", "2.4.7", "Focus indicator judged visible after Tab",
                                        r.key_t, round(r.key_t + cfg.focus_window_s, 3), r.source, r.reasoning,
                                        c.confidence if r.source == "classical" else None, ev))
            elif r.verdict == "not_visible" and timing_doubtful:
                items.append(ReportItem(f"focus-{n}", "needs_review", "2.4.7",
                                        "Focus not visible after Tab, but the clock alignment is not trusted",
                                        r.key_t, round(r.key_t + cfg.focus_window_s, 3), r.source, DOUBT + r.reasoning,
                                        c.confidence if r.source == "classical" else None, ev))
            elif r.verdict == "not_visible":
                items.append(ReportItem(f"focus-{n}", "finding", "2.4.7", "Keyboard focus not visible after Tab",
                                        r.key_t, round(r.key_t + cfg.focus_window_s, 3), r.source, r.reasoning,
                                        c.confidence if r.source == "classical" else None, ev))
            elif r.verdict == "unreadable":
                items.append(ReportItem(f"focus-{n}", "needs_review", "2.4.7", "Focus check not possible: the recorded screen was black",
                                        r.key_t, round(r.key_t + cfg.focus_window_s, 3), r.source, r.reasoning, None, ev))
            else:
                items.append(ReportItem(f"focus-{n}", "needs_review", "2.4.7", "Focus indicator unclear after Tab",
                                        r.key_t, round(r.key_t + cfg.focus_window_s, 3), r.source, r.reasoning,
                                        c.confidence if r.source == "classical" else None, ev))
        s = casc.stats
        stats.update({"vlm_calls": s.vlm_calls, "input_tokens": s.input_tokens, "output_tokens": s.output_tokens,
                      "est_cost_usd": s.est_cost_usd})
        used = f"; the model decided {s.escalated - s.unresolved} of {s.escalated} escalated presses" if client else \
               (f"; {vcount['ambiguous']} unclear presses left for a human or model (no model was used)" if vcount["ambiguous"] else "")
        dropped = len(focus_presses) - s.presses
        head = (f"{len(focus_presses)} Tab press(es) in the key log, but only {s.presses} could be analyzed ({dropped} fell "
                "outside the video or could not be read, which usually means the key log and video are misaligned)"
                if dropped > 0 else f"{s.presses} Tab press(es)")
        doubt = (" Findings were downgraded to 'needs review' because the clock alignment is not trusted."
                 if timing_doubtful and vcount["not_visible"] else "")
        checks.append({"name": "Focus visible (WCAG 2.4.7)", "status": "ran",
                       "detail": f"{head}: {vcount['visible']} visible, {vcount['not_visible']} not visible, "
                                 f"{vcount['ambiguous']} unresolved"
                                 + (f", {vcount['unreadable']} unreadable (the screen was black)" if vcount["unreadable"] else "")
                                 + f"{used}.{doubt}"})

    _p(0.75, "Comparing on-screen messages with the transcript")
    # ---- seen versus heard (4.1.3) ----
    utts, source, audio_offset = None, "", (offset if transcript_clock == "video" else 0.0)
    if (rec / "utterances.json").exists():
        utts = [Utterance(**u) for u in json.loads((rec / "utterances.json").read_text())]
        source = "utterances.json"
    elif (rec / "audio.wav").exists() and (transcribe or (rec / "transcript_asr.json").exists()):
        from a11y_replay.ingest.audio import load_or_transcribe
        utts = load_or_transcribe(rec, whisper_model)
        source = f"speech recognition of audio.wav ({whisper_model})"
    elif transcribe:
        from a11y_replay.ingest.audio import extract_audio
        from a11y_replay.ingest.audio import transcribe as asr
        wav = out / "audio_from_video.wav"
        if extract_audio(original_video, wav):
            utts = asr(wav, whisper_model)
            audio_offset = offset          # the video's own audio is on the video clock
            source = f"speech recognition of the video's audio track ({whisper_model})"
    if utts is None:
        why = ("audio.wav found but not recognized; pass --transcribe" if (rec / "audio.wav").exists()
               else "the video has no audio track to recognize" if transcribe
               else "needs a transcript (utterances.json), audio.wav, or --transcribe to use the video's audio track")
        checks.append({"name": "Status messages announced (WCAG 4.1.3)", "status": "skipped", "detail": why})
    else:
        obs = observe_changes(video, keys, offset=offset, cfg=cfg)
        results = match_observations(obs, utts, audio_offset=audio_offset, cfg=cfg)
        n_un = n_ok = 0
        for n, r in enumerate(results, 1):
            if r.verdict == "announced":
                n_ok += 1
                if not include_passes:
                    continue
            full = read_frame_at(video, r.t + offset + 0.3)
            ev = []
            if full is not None:
                x, y, w, h = r.bbox
                shot = full.copy()
                cv2.rectangle(shot, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 3)
                pad = 90
                ev = [{"caption": "The change on screen (red box)",
                       "file": _save(out, f"announce{n}", _fit_height(shot[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad], 200))}]
            if r.verdict == "announced":
                items.append(ReportItem(f"announce-{n}", "pass", "4.1.3", f'Status message judged announced: "{r.text}"',
                                        r.t, round(r.t + cfg.announce_window_s, 3), "classical", r.reasoning, None, ev))
            elif r.verdict == "unannounced" and timing_doubtful:
                items.append(ReportItem(f"announce-{n}", "needs_review", "4.1.3",
                                        f'Message possibly not announced, but the clock alignment is not trusted: "{r.text}"',
                                        r.t, round(r.t + cfg.announce_window_s, 3), "classical", DOUBT + r.reasoning, None, ev))
            elif r.verdict == "unannounced":
                n_un += 1
                items.append(ReportItem(f"announce-{n}", "finding", "4.1.3", f'Status message not announced: "{r.text}"',
                                        r.t, round(r.t + cfg.announce_window_s, 3), "classical", r.reasoning, None, ev))
            else:
                title = (f'Status message only partly announced: "{r.text}"' if r.text
                         else "Change on screen with no readable text")
                items.append(ReportItem(f"announce-{n}", "needs_review", "4.1.3", title,
                                        r.t, r.t if not r.text else round(r.t + cfg.announce_window_s, 3),
                                        "classical", r.reasoning, None, ev))
        checks.append({"name": "Status messages announced (WCAG 4.1.3)", "status": "ran",
                       "detail": f"{n_ok + n_un} message(s) read from the screen: {n_ok} announced, {n_un} not announced; "
                                 f"transcript from {source}."})

    for name, why in NOT_IMPLEMENTED:
        checks.append({"name": name, "status": "not_implemented", "detail": why})

    # a flashing episode is also "a change with no text"; do not list it a second time for review
    flash_spans = [(i.start - 0.5, i.end + 0.5) for i in items if i.criterion == "2.3.1" and i.status == "finding"]
    items = [i for i in items if not (i.criterion == "4.1.3" and i.status == "needs_review"
                                      and any(a <= i.start <= b for a, b in flash_spans))]
    items.sort(key=lambda i: i.start)
    report = AuditReport(recording=rec.name, created=time.strftime("%Y-%m-%d %H:%M"), version=__version__,
                         sync=sync, checks=checks, items=items, stats=stats, out_dir=str(out), warnings=warnings)
    _p(0.95, "Writing the report")
    from a11y_replay.report import write_report
    write_report(report, out)
    _p(1.0, "Done")
    return report


def audit_uploads(video_path, keys_path=None, transcript_path=None, model: str | None = None,
                  work_dir=None) -> Path:
    """For the demo UI: gather uploaded files into a recording folder, audit it, return the report folder."""
    work = Path(work_dir or tempfile.mkdtemp(prefix="a11y_audit_"))
    rec = work / "recording"
    rec.mkdir(parents=True, exist_ok=True)
    shutil.copy(video_path, rec / f"video{Path(video_path).suffix or '.mp4'}")
    if keys_path:
        shutil.copy(keys_path, rec / "keys.jsonl")
    if transcript_path:
        shutil.copy(transcript_path, rec / "utterances.json")
    client = None
    if model and model != "none":
        from a11y_replay.vlm.client import AnthropicClient, CachedClient
        client = CachedClient(AnthropicClient(model=model), work / "vlm_cache")
    return Path(run_audit(rec, work / "report", client=client).out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--out", default=None, help="report folder (default: <recording>/audit_report)")
    ap.add_argument("--model", default=None, help='VLM for unclear focus cases, e.g. claude-sonnet-5-5 (default: none)')
    ap.add_argument("--transcribe", action="store_true", help="recognize audio.wav with faster-whisper")
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--show-passes", action="store_true",
                    help="also list what was judged fine, with evidence, so the passes can be checked too")
    ap.add_argument("--transcript-clock", choices=("log", "video"), default="log",
                    help="clock of utterances.json times: the key log's (default) or the video's own timeline")
    a = ap.parse_args(argv)
    client = None
    if a.model:
        from a11y_replay.vlm.client import AnthropicClient, CachedClient
        client = CachedClient(AnthropicClient(model=a.model), Path(a.out or a.recording) / "vlm_cache")
    r = run_audit(a.recording, a.out, client=client, transcribe=a.transcribe, whisper_model=a.whisper_model,
                  transcript_clock=a.transcript_clock, include_passes=a.show_passes)
    ran = sum(c["status"] == "ran" for c in r.checks)
    skipped = [c for c in r.checks if c["status"] == "skipped"]
    print(f"\nAudit of {r.recording}: {len(r.findings)} finding(s), {len(r.needs_review)} item(s) needing review")
    print(f"  Video: {r.stats['video_file']}, {r.stats['video_seconds']} s, last modified {r.stats['video_modified']}")
    for w in r.warnings:
        print(f"  WARNING: {w}")
    s = r.sync
    spread = "n/a" if s.get("spread_s") is None else f"{s['spread_s']:.2f} s"
    rule = f"needs at least {DEFAULT.sync_min_pairs} pairs and spread under {DEFAULT.sync_max_spread_s:.2f} s"
    print(f"  Clock sync: offset {s['offset_s']:+.2f} s from {s['pairs']} key/change pair(s), "
          f"{s.get('calibration_pairs', 0)} of them calibration flashes, spread {spread}; "
          f"{'trusted' if s['trusted'] else 'NOT TRUSTED (' + rule + ')'}. {s['note']}")
    if s.get("detail") and not s["trusted"]:
        print("    pairs (key @ log time, how far its matched screen change was, how much of the frame that change covered): "
              + ", ".join(f"{d['key']}@{d['key_time_s']:.1f}s {d['delta_s']:+.2f}s ({d['change_frac'] * 100:.1f}%)"
                          for d in s["detail"]))
        if any(d["key"] == DEFAULT.calibration_key and d["change_frac"] > DEFAULT.marker_max_frac for d in s["detail"]):
            print(f"    NOTE: a flash-time key matched a change covering more than {DEFAULT.marker_max_frac:.0%} of the frame. "
                  "A calibration flash is a small square, so that match was NOT counted as a flash.")
    if r.passes:
        print(f"  ({len(r.passes)} item(s) judged fine are listed in the report because --show-passes was used)")
    if skipped:
        print(f"  NOTE: only {ran} of {ran + len(skipped)} available checks ran, so zero findings does NOT mean the recording is clean.")
    for c in r.checks:
        print(f"  [{c['status']:15}] {c['name']}: {c['detail']}")
    for i in r.findings:
        print(f"  FINDING  {i.start:7.2f}s  WCAG {i.criterion}  {i.title}")
    print(f"\nReport: {Path(r.out_dir) / 'report.html'}")


if __name__ == "__main__":
    main()