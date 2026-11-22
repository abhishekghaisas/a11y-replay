"""Playwright generator: record video + keys.jsonl + labels.json per sample.

Each app in benchmark/testapps/ has a toggleable bug (?bug=1). Because the bug is injected,
its type and timestamp are known exactly. Clean versions are the negative controls.

Usage:
    python -m benchmark.generate --apps focus_outline flashing_banner --n 3 --seed 0
    python -m benchmark.generate --list
    python -m benchmark.generate --no-validate     # skip the clock-sync quality gate

Output per sample (under benchmark/data/<sample_id>/):
    video.mp4 (or video.webm if ffmpeg is missing), keys.jsonl, labels.json

Quality gate:
    After recording, each sample is checked with the same clock-sync estimator the pipeline
    uses. If the key log and video cannot be aligned (video timing was distorted, e.g. by CPU
    load during recording), the sample is re-recorded, up to --max-attempts times, and
    otherwise discarded. The outcome is stored under "qa" in labels.json (offset, pairs,
    spread, attempts) so the discard/retry rate can be reported. Without this gate, a
    recording whose video drifts against the log would carry wrong ground-truth timing.

Timing model:
    t0 is captured right after the page is created. Key events and labels are measured
    against t0 with time.monotonic(). Playwright's video starts slightly before/after t0,
    so a small natural video-to-log offset remains. That is intentional: it is what
    a11y_replay/sync.py has to estimate. The measured offset is NOT written to the labels.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from a11y_replay.config import DEFAULT
from a11y_replay.taxonomy import WCAG_CRITERION, IssueType

TESTAPPS = Path(__file__).parent / "testapps"
DEFAULT_OUT = Path(__file__).parent / "data"
VIEWPORT = {"width": 1280, "height": 720}


# --------------------------------------------------------------------------- helpers
class Recorder:
    """Wraps a Playwright page; logs key presses against t0."""

    def __init__(self, page, t0: float, clock=time.monotonic):
        self.page = page
        self.t0 = t0
        self.clock = clock                 # injectable so tests can make waits advance time
        self.keys: list[dict] = []
        self.extra: dict = {}  # extra metadata a scenario wants written into labels.json

    def now(self) -> float:
        return self.clock() - self.t0

    def press(self, key: str) -> float:
        t = self.now()
        self.page.keyboard.press(key)
        self.keys.append({"t": round(t, 3), "key": key, "modifiers": []})
        return t

    def wait(self, seconds: float) -> None:
        self.page.wait_for_timeout(int(seconds * 1000))


def make_label(issue: IssueType, t: float, t_end: float | None = None, **extra) -> dict:
    label = {"issue_type": issue.value, "criterion": WCAG_CRITERION[issue], "t": round(t, 3)}
    if t_end is not None:
        label["t_end"] = round(t_end, 3)
    label.update(extra)
    return label


# --------------------------------------------------------------------------- scenarios
def calibrate(rec: Recorder, rng: random.Random, n: int = 4) -> None:
    """Press the calibration key n times at the start. Each press flashes a marker in the page,
    giving sync.py key/effect pairs even when the bug under test shows nothing visible.
    n is larger than Config.sync_min_pairs so one missed pair does not make the estimate
    untrusted. The lead-in wait keeps the first press clear of the page's initial render.
    Detectors must ignore these key presses (they are not part of the workflow being audited)."""
    rec.wait(rng.uniform(1.5, 2.0))  # let the page render first
    for _ in range(n):
        rec.press(DEFAULT.calibration_key)
        rec.wait(rng.uniform(0.7, 1.0))


def scenario_focus_outline(rec: Recorder, rng: random.Random, bug: bool) -> list[dict]:
    """Tab through the form. In the bug version no focus indicator is ever visible,
    so every Tab press is a WCAG 2.4.7 failure instance."""
    n_tabs = rng.randint(4, 6)  # page has 7 tabbable elements; stay inside the page
    calibrate(rec, rng)
    rec.wait(rng.uniform(1.0, 1.8))  # idle lead-in
    labels = []
    for i in range(n_tabs):
        t = rec.press("Tab")
        if bug:
            labels.append(make_label(IssueType.FOCUS_NOT_VISIBLE, t, tab_index=i + 1))
        rec.wait(rng.uniform(0.8, 1.6))
    rec.wait(0.8)
    return labels


def scenario_flashing_banner(rec: Recorder, rng: random.Random, bug: bool) -> list[dict]:
    """Trigger the banner animation at a known time. Bug: >3 flashes/s (WCAG 2.3.1).
    Clean: a slow 1 Hz pulse, a hard negative that moves but stays under the threshold."""
    calibrate(rec, rng)
    rec.wait(rng.uniform(1.0, 2.0))
    t_start = rec.now()
    rec.page.evaluate("window.startFlashing()")
    rec.wait(2.0)
    t_end = rec.now()
    rec.wait(1.0)
    if bug:
        return [make_label(IssueType.FLASHING, t_start, t_end)]
    return []


# Expected focus-visibility verdict per style. "ambiguous" means reasonable reviewers could
# disagree; those presses are not counted as failures. Rationale: docs/labeling_guide.md
STYLE_EXPECTED = {
    "strong_ring": "visible",
    "box_shadow_ring": "visible",
    "thin_dotted": "visible",
    "underline": "visible",
    "bg_change": "visible",
    "faint_ring": "ambiguous",
    "selection_only": "ambiguous",
    "subtle_color": "not_visible",
    "none": "not_visible",
}


def scenario_focus_styles(rec: Recorder, rng: random.Random, bug: bool) -> list[dict]:
    """Tab through nine controls, each with a different focus style, in a shuffled order.
    Failures (expected not_visible) are labeled as issues; every press is recorded in
    `focus_checks` with its expected verdict so the detector can be scored on all three."""
    order = list(STYLE_EXPECTED)
    rng.shuffle(order)
    rec.page.evaluate("order => window.setOrder(order)", order)
    calibrate(rec, rng)
    rec.wait(rng.uniform(0.8, 1.4))
    labels, checks = [], []
    for i, style in enumerate(order):
        t = rec.press("Tab")
        expected = STYLE_EXPECTED[style]
        checks.append({"t": round(t, 3), "tab_index": i + 1, "style": style, "expected": expected})
        if expected == "not_visible":
            labels.append(make_label(IssueType.FOCUS_NOT_VISIBLE, t, tab_index=i + 1, style=style))
        rec.wait(rng.uniform(0.8, 1.4))
    rec.wait(0.6)
    rec.extra["focus_checks"] = checks
    return labels


STATUS_TEXT = {
    "toast": "Changes saved",
    "banner": "Upload failed: file too large",
    "inline": "3 results found",
}
SPOKEN_PREFIX = {"banner": "Alert, "}  # screen readers often prepend the role; exercises fuzzy matching


# Wrong speech that can appear near a SILENT message, by what kind of mistake it would be. The page never
# sent any of it, so the message is still a failure. Categories let a report say which kinds fool the matcher.
DISTRACTORS = {
    "toast": [("near_miss", "Changes discarded"), ("confusable", "Changes saving")],
    "banner": [("near_miss", "Upload complete"), ("truncated", "Upload failed")],
    "inline": [("wrong_number", "13 results found"), ("negation", "No results found")],
}
UNRELATED = ["Navigation landmark, main", "Link, Terms", "Heading level 1, Project dashboard",
             "Edit text, Search files", "Button, Save"]


def _speech_len(text: str) -> float:
    return 0.2 + 0.35 * len(text.split())


def _status_messages(rec: Recorder, rng: random.Random, distractors: bool) -> list[dict]:
    """Three status messages (toast, error banner, inline text), each randomly announced (inserted into a live
    region, with a simulated screen-reader line) or silent (plain element, nothing announced = the failure),
    triggered by a timer or by Enter on a button. Silent messages are failures whatever triggered them.
    With distractors=True each silent message also gets wrong speech inside the announcement window: a near-miss
    of one kind (see DISTRACTORS) shortly after it appears, then an unrelated line near the end of the window.
    Transcript lines are simulated and error-free. Utterances are spaced so they never overlap and never sit
    closer than 0.9 s, which keeps the audio version of the sample cleanly separable."""
    calibrate(rec, rng)
    rec.wait(rng.uniform(0.8, 1.4))
    kinds = list(STATUS_TEXT)
    rng.shuffle(kinds)
    if distractors:
        # one message of a random kind is ALWAYS silent, so every kind (and so every kind of wrong speech)
        # gets covered across samples instead of depending on a coin flip; the others are random, with at
        # least one announced so both classes appear
        forced = rng.choice(kinds)
        announced = [k != forced and rng.random() < 0.5 for k in kinds]
        if not any(announced):
            announced[[i for i, k in enumerate(kinds) if k != forced][-1]] = True
    else:
        announced = [rng.random() < 0.5 for _ in kinds]
        if len(set(announced)) == 1:        # always have both classes in every sample
            announced[-1] = not announced[-1]
    labels, checks, utterances = [], [], []
    for kind, ann in zip(kinds, announced):
        text, trigger = STATUS_TEXT[kind], rng.choice(["timer", "key"])
        cfg = {"kind": kind, "text": text, "announced": ann}
        if trigger == "key":
            rec.page.evaluate("c => window.armButton(c)", cfg)
            rec.page.focus("#save")      # programmatic focus: no key event, no visible change
            rec.wait(0.5)
            t = rec.press("Enter")
        else:
            rec.wait(rng.uniform(0.3, 0.8))
            t = rec.now()
            rec.page.evaluate("c => window.showStatus(c)", cfg)
        extra = []
        if ann:
            start = t + rng.uniform(0.3, 1.2)     # screen-reader latency
            spoken = SPOKEN_PREFIX.get(kind, "") + text
            utterances.append({"start": round(start, 3), "end": round(start + _speech_len(spoken), 3), "text": spoken})
        else:
            labels.append(make_label(IssueType.UNANNOUNCED_CHANGE, t, kind=kind, text=text, trigger=trigger))
            if distractors:
                cat, wrong = rng.choice(DISTRACTORS[kind])
                s1 = t + rng.uniform(0.3, 0.5)
                s2 = t + rng.uniform(2.7, 2.95)
                other = rng.choice(UNRELATED)
                utterances.append({"start": round(s1, 3), "end": round(s1 + _speech_len(wrong), 3), "text": wrong})
                utterances.append({"start": round(s2, 3), "end": round(s2 + _speech_len(other), 3), "text": other})
                extra = [{"category": cat, "text": wrong}, {"category": "unrelated", "text": other}]
        checks.append({"t": round(t, 3), "kind": kind, "text": text, "announced": ann, "trigger": trigger,
                       "distractors": extra})
        rec.wait((4.8 if distractors else 3.8) + rng.uniform(0.6, 1.2))     # let it auto-dismiss, keep speech apart
    rec.extra["status_checks"] = checks
    rec.extra["utterances"] = sorted(utterances, key=lambda u: u["start"])
    rec.extra["transcript"] = {"file": "utterances.json", "kind": "simulated, error-free"}
    return labels


def scenario_status_messages(rec: Recorder, rng: random.Random, bug: bool) -> list[dict]:
    return _status_messages(rec, rng, distractors=False)


def scenario_status_messages_noisy(rec: Recorder, rng: random.Random, bug: bool) -> list[dict]:
    return _status_messages(rec, rng, distractors=True)


SCENARIOS = {
    "focus_outline": scenario_focus_outline,
    "flashing_banner": scenario_flashing_banner,
    "focus_styles": scenario_focus_styles,
    "status_messages": scenario_status_messages,
    "status_messages_noisy": scenario_status_messages_noisy,
}
APP_PAGE = {"status_messages_noisy": "status_messages"}   # same test page, different scenario

# Variants recorded per app. Apps with a single mixed page have one variant.
VARIANTS = {"focus_styles": ("mixed",), "status_messages": ("mixed",), "status_messages_noisy": ("mixed",)}
DEFAULT_VARIANTS = ("bug", "clean")


# --------------------------------------------------------------------------- recording
def finalize_video(src_webm: Path, dest_dir: Path) -> str:
    """Move the recording into the sample dir; convert to mp4 if ffmpeg exists."""
    if shutil.which("ffmpeg"):
        dest = dest_dir / "video.mp4"
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_webm),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)],
            capture_output=True,
        )
        if result.returncode == 0:
            return dest.name
    dest = dest_dir / "video.webm"
    shutil.move(str(src_webm), dest)
    return dest.name


MAX_ABS_OFFSET_S = 0.25  # a plausible video-to-log offset; larger means something is wrong


def sync_qa(sample_dir: Path) -> dict:
    """Run the pipeline's own clock-sync estimator on a recorded sample."""
    from a11y_replay.ingest.keys import load_keys  # lazy: needs OpenCV
    from a11y_replay.segment import detect_changes
    from a11y_replay.sync import estimate_offset

    video = next(sample_dir.glob("video.*"))
    est = estimate_offset(load_keys(sample_dir / "keys.jsonl"), detect_changes(video))
    return {
        "ok": bool(est.ok and abs(est.offset) <= MAX_ABS_OFFSET_S),
        "offset_s": round(est.offset, 3),
        "n_pairs": est.n_pairs,
        "spread_s": None if est.n_pairs == 0 else round(est.spread, 3),
    }


def record_validated(record, qa, max_attempts: int = 3):
    """Call record() until qa(path)["ok"] or attempts run out.

    Returns (path, qa_result, attempts); path is None if every attempt failed the check.
    """
    result = None
    for attempt in range(1, max_attempts + 1):
        path = record()
        result = qa(path)
        if result["ok"]:
            return path, result, attempt
    return None, result, max_attempts


def record_sample(browser, app: str, variant: str, sample_id: str, seed: int, out_root: Path) -> Path:
    bug = variant == "bug"
    rng = random.Random(seed)
    sample_dir = out_root / sample_id
    if sample_dir.exists():
        shutil.rmtree(sample_dir)  # a retry must not inherit files from a failed attempt
    sample_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = browser.new_context(
            viewport=VIEWPORT, record_video_dir=tmp, record_video_size=VIEWPORT
        )
        page = ctx.new_page()
        t0 = time.monotonic()
        url = (TESTAPPS / APP_PAGE.get(app, app) / "index.html").resolve().as_uri() + ("?bug=1" if bug else "")
        page.goto(url)
        rec = Recorder(page, t0)

        labels = SCENARIOS[app](rec, rng, bug)
        duration = rec.now()

        page.close()
        ctx.close()  # finalizes the video file
        video_file = finalize_video(Path(page.video.path()), sample_dir)

    utterances = rec.extra.pop("utterances", None)
    if utterances is not None:
        (sample_dir / "utterances.json").write_text(json.dumps(utterances, indent=2), encoding="utf-8")

    with open(sample_dir / "keys.jsonl", "w", encoding="utf-8") as f:
        for k in rec.keys:
            f.write(json.dumps(k) + "\n")

    meta = {
        "sample_id": sample_id,
        "app": app,
        "variant": variant,
        "seed": seed,
        "video": video_file,
        "viewport": VIEWPORT,
        "duration_s": round(duration, 3),
        "keys_source": "logger",
        "calibration_key": DEFAULT.calibration_key,
        "audio": None,  # filled in later by benchmark/tts.py (simulated)
        "labels": labels,
        **rec.extra,
    }
    with open(sample_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return sample_dir


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument("--n", type=int, default=3, help="samples per app per variant (bug/clean)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-validate", action="store_true", help="skip the clock-sync quality gate")
    ap.add_argument("--max-attempts", type=int, default=3, help="re-record attempts per sample")
    ap.add_argument("--list", action="store_true", help="list available apps and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("\n".join(SCENARIOS))
        return

    from playwright.sync_api import sync_playwright  # imported lazily: heavy dependency

    args.out.mkdir(parents=True, exist_ok=True)
    written = discarded = retries = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for app in args.apps:
            for variant in VARIANTS.get(app, DEFAULT_VARIANTS):
                for i in range(args.n):
                    seed = args.seed * 10_000 + i
                    sid = f"{app}_{variant}_{i:03d}"

                    def record(app=app, variant=variant, sid=sid, seed=seed):
                        return record_sample(browser, app, variant, sid, seed, args.out)

                    if args.no_validate:
                        print(f"wrote {record()}")
                        written += 1
                        continue

                    path, qa, attempts = record_validated(record, sync_qa, args.max_attempts)
                    retries += attempts - 1
                    if path is None:
                        shutil.rmtree(args.out / sid, ignore_errors=True)
                        discarded += 1
                        print(f"DISCARDED {sid}: clock sync failed {attempts}x "
                              f"(pairs={qa['n_pairs']}, offset={qa['offset_s']:+.3f}s)")
                        continue
                    meta_path = path / "labels.json"
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["qa"] = {**qa, "attempts": attempts}
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    written += 1
                    print(f"wrote {path}  (sync offset {qa['offset_s']:+.3f}s, attempts {attempts})")
        browser.close()
    print(f"\ndone: {written} written, {discarded} discarded, {retries} re-recordings")


if __name__ == "__main__":
    main()