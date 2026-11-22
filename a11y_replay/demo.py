"""Everything the demo UI does, as plain functions with no Gradio in them, so it can be tested without a display.

app.py only wires these to widgets. If the UI misbehaves on your machine, `python -m a11y_replay.audit` does the
same work from the command line, and `python app.py --selftest` runs this module without opening a browser.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from a11y_replay.audit import AuditReport, run_audit

DISCLAIMER = ("A screening aid, not a compliance test. It checks keyboard-focus visibility (2.4.7), flashing content (2.3.1) "
              "and unannounced status messages (4.1.3) from a screen recording. A report with no findings does not mean the "
              "page is accessible, and every finding should be confirmed by a person.")
HEADERS = ["Time", "WCAG", "Status", "Decided by", "What", "Why"]


@dataclass
class DemoResult:
    ok: bool
    summary: str                                   # markdown for the top of the results panel
    rows: list = field(default_factory=list)       # one list per item, matching HEADERS
    evidence: list = field(default_factory=list)   # (image path, caption)
    report_html: str | None = None
    report_dir: str | None = None
    evidence_bytes: list = field(default_factory=list)   # (png bytes, caption), filled when keep_files=False
    report_bytes: bytes | None = None                    # the report.html, filled when keep_files=False


def find_examples(roots=("recordings", "benchmark/data")) -> dict:
    """Recording folders the demo can run without an upload: they contain a video. Label -> path."""
    found = {}
    for root in roots:
        for d in sorted(Path(root).glob("*")):
            if d.is_dir() and any(d.glob("video.*")):
                label = f"{root}/{d.name}"
                found[label] = str(d)
    return found


def _fail(message: str) -> DemoResult:
    return DemoResult(ok=False, summary=f"**Could not run the audit.** {message}")


def _summary(r: AuditReport) -> str:
    ran = [c for c in r.checks if c["status"] == "ran"]
    skipped = [c for c in r.checks if c["status"] == "skipped"]
    lines = [f"### {len(r.findings)} finding(s), {len(r.needs_review)} needing review",
             f"**{len(ran)} of {len(ran) + len(skipped)} available checks ran.**"
             + (" A result of zero findings therefore does not mean the recording is clean." if skipped else ""),
             "", f"> {DISCLAIMER}", ""]
    for w in r.warnings:
        lines.append(f"> **Warning.** {w}")
        lines.append("")
    s = r.sync
    lines.append(f"**Clock sync:** offset {s['offset_s']:+.2f} s from {s['pairs']} key press(es); "
                 f"{'trusted' if s['trusted'] else '**NOT trusted**'}. {s['note']}")
    lines.append("")
    lines.append("**Checks**")
    words = {"ran": "Ran", "skipped": "Skipped", "not_implemented": "Not implemented"}
    for c in r.checks:
        lines.append(f"- {words.get(c['status'], c['status'])}: {c['name']}. {c['detail']}")
    if r.stats.get("model"):
        lines.append(f"\n**Model:** {r.stats['model']}, {r.stats.get('vlm_calls', 0)} call(s).")
    return "\n".join(lines)


def _rows(r: AuditReport) -> list:
    label = {"finding": "Finding", "needs_review": "Needs review", "pass": "Judged fine"}
    return [[f"{i.start:.1f} s", f"WCAG {i.criterion}", label.get(i.status, i.status),
             "language model" if i.method == "vlm" else "classical detector", i.title, i.reasoning] for i in r.items]


def check_upload(path, max_mb: float = 60.0, max_seconds: float = 180.0):
    """None if the file is acceptable, otherwise a message saying what to do. Keeps a public demo from being stalled by
    one huge or unreadable file."""
    size = Path(path).stat().st_size / 1e6
    if size > max_mb:
        return (f"This file is {size:.0f} MB; the demo accepts up to {max_mb:.0f} MB. Trim it, or shrink it with "
                "`ffmpeg -i in.mov -vf scale=1280:-2 -c:v libx264 -pix_fmt yuv420p out.mp4`.")
    from a11y_replay.ingest.video import container_info, probe
    try:
        info = container_info(path)
        seconds = info["duration"] if info and info["duration"] else probe(path).duration
    except Exception:
        return "This video could not be read. Try converting it to .mp4 with ffmpeg."
    if seconds > max_seconds:
        return f"This recording is {seconds:.0f} s long; the demo accepts up to {max_seconds:.0f} s. Please upload a shorter clip."
    return None


def _run_demo(example, video, keys, transcript, model, transcript_clock, show_passes, progress, examples, work) -> DemoResult:
    try:
        if example and example in examples:
            rec = Path(examples[example])
        elif video:
            rec = work / "recording"
            rec.mkdir()
            shutil.copy(video, rec / f"video{Path(video).suffix or '.mp4'}")
            if keys:
                shutil.copy(keys, rec / "keys.jsonl")
            if transcript:
                shutil.copy(transcript, rec / "utterances.json")
        else:
            return _fail("Choose an example recording, or upload a screen recording.")

        client = None
        if model and model != "none":
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            if not os.environ.get("ANTHROPIC_API_KEY"):
                return _fail(f"The model `{model}` needs ANTHROPIC_API_KEY (put it in a .env file) and it is not set. "
                             "Choose 'none' to run without a language model.")
            from a11y_replay.vlm.client import AnthropicClient, CachedClient
            client = CachedClient(AnthropicClient(model=model), work / "vlm_cache")

        report = run_audit(rec, work / "report", client=client, transcript_clock=transcript_clock,
                           include_passes=show_passes, progress=progress)
        out = Path(report.out_dir)
        evidence = [(str(out / e["file"]), f"{i.title} ({i.start:.1f} s)") for i in report.items for e in i.evidence
                    if (out / e["file"]).exists()]
        return DemoResult(ok=True, summary=_summary(report), rows=_rows(report), evidence=evidence,
                          report_html=str(out / "report.html"), report_dir=str(out))
    except FileNotFoundError as e:
        return _fail(f"{e}. If the video will not open, convert it: `ffmpeg -i video.mov -vf scale=1280:-2 "
                     "-c:v libx264 -pix_fmt yuv420p video.mp4`.")
    except RuntimeError as e:
        text = str(e)
        if "esseract" in text:
            return _fail("The status-message check needs Tesseract. Install it with `brew install tesseract` and "
                         "`pip install pytesseract`, or run without a transcript.")
        return _fail(text)
    except Exception as e:  # last resort: a message, not a stack trace, in front of an audience
        return _fail(f"Unexpected problem: {type(e).__name__}: {e}. The command-line version "
                     "(`python -m a11y_replay.audit <folder>`) does the same work and prints a full error.")


def run_demo(example: str | None = None, video: str | None = None, keys: str | None = None,
             transcript: str | None = None, model: str = "none", transcript_clock: str = "log",
             show_passes: bool = False, progress=None, examples: dict | None = None,
             keep_files: bool = True) -> DemoResult:
    """Audit an example folder or an uploaded video (+ optional key log and transcript). Never raises: problems
    come back as a DemoResult with ok=False and a message a person can act on.

    keep_files=True (local use): the report and evidence stay on disk and the result holds their paths.
    keep_files=False (a hosted demo): everything is read into memory and ALL temporary files, including the uploaded
    recording, are deleted before returning. A failed run always cleans up after itself."""
    examples = find_examples() if examples is None else examples
    work = Path(tempfile.mkdtemp(prefix="a11y_demo_"))
    result = None
    try:
        result = _run_demo(example, video, keys, transcript, model, transcript_clock, show_passes, progress, examples, work)
        if result.ok and not keep_files:
            result.evidence_bytes = [(Path(p).read_bytes(), cap) for p, cap in result.evidence]
            result.report_bytes = Path(result.report_html).read_bytes()
            result.evidence, result.report_html, result.report_dir = [], None, None
        return result
    finally:
        if not keep_files or result is None or not result.ok:
            shutil.rmtree(work, ignore_errors=True)