"""Estimate the video-to-key-log offset for one or more samples.

Usage (from the repo root):
    python check_sync.py benchmark/data/focus_outline_*
    python check_sync.py benchmark/data/focus_outline_clean_000 --shift 0.25

--verbose lists, for each sample, every key press and visual change, marking which key presses
found a partner. Use it to see WHY a sample has fewer pairs than expected.

--shift S adds S seconds to every key time, simulating a clock-sync error. If the estimator
works, the reported offset should move by about -S compared with the unshifted run.
"""
import sys
from dataclasses import replace
from pathlib import Path

from a11y_replay.config import DEFAULT
from a11y_replay.ingest.keys import load_keys
from a11y_replay.segment import detect_changes
from a11y_replay.sync import estimate_offset

argv = sys.argv[1:]
verbose = "--verbose" in argv
argv = [a for a in argv if a != "--verbose"]
shift = 0.0
if "--shift" in argv:
    i = argv.index("--shift")
    shift = float(argv[i + 1])
    del argv[i:i + 2]
samples = [Path(a) for a in argv] or [Path("benchmark/data/focus_outline_clean_000")]

print(f"{'sample':28} {'pairs':>5} {'offset(s)':>10} {'spread(s)':>10}  trusted")
for s in samples:
    video = next(s.glob("video.*"))
    keys = [replace(k, t=k.t + shift) for k in load_keys(s / "keys.jsonl")]
    changes = detect_changes(video)
    est = estimate_offset(keys, changes)
    spread = "-" if est.n_pairs == 0 else f"{est.spread:.3f}"
    print(f"{s.name:28} {est.n_pairs:5d} {est.offset:10.3f} {spread:>10}  {'yes' if est.ok else 'no'}")
    if verbose:
        for k in keys:
            near = [c for c in changes if k.t - DEFAULT.sync_pre_s <= c.t <= k.t + DEFAULT.sync_post_s]
            tag = f"change at {near[0].t:.2f}s ({near[0].t - k.t:+.3f})" if near else "NO change in window"
            print(f"    key {k.key:>4} @ {k.t:6.2f}s  -> {tag}")
        print("    video change onsets:", ", ".join(f"{c.t:.2f}" for c in changes))