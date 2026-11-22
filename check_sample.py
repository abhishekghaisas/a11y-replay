"""Sanity check: do segmentation events line up with the key log for one sample?

Usage (from the repo root):
    python check_sample.py benchmark/data/focus_outline_clean_000
    python check_sample.py benchmark/data/focus_outline_bug_001 --save

--save writes an annotated frame (changed region boxed in red) for every key press that
had a nearby change, into check_out/. Open them to see WHAT changed, not just that
something did.
"""
import json
import sys
from pathlib import Path

import cv2

from a11y_replay.config import DEFAULT
from a11y_replay.ingest.video import read_frame_at
from a11y_replay.segment import detect_changes

args = [a for a in sys.argv[1:] if not a.startswith("--")]
save = "--save" in sys.argv
sample = Path(args[0] if args else "benchmark/data/focus_outline_clean_000")
video = next(sample.glob("video.*"))
events = detect_changes(video)
keys = [k for k in (json.loads(line) for line in open(sample / "keys.jsonl"))
        if k["key"] != DEFAULT.calibration_key]  # calibration presses are not the workflow

if save:
    Path("check_out").mkdir(exist_ok=True)

print(f"{sample.name}: {len(events)} change events")
hits = 0
for i, k in enumerate(keys, 1):
    near = [e for e in events if k["t"] - 0.3 <= e.t <= k["t"] + 0.5]
    if not near:
        print(f"  {k['key']}@{k['t']:.2f}s -> no change")
        continue
    hits += 1
    e = near[0]
    print(f"  {k['key']}@{k['t']:.2f}s -> change at {e.t:.2f}s, bbox(x,y,w,h)={e.bbox}, "
          f"{e.magnitude * 100:.2f}% of frame")
    if save:
        frame = read_frame_at(video, e.t_end + 0.1)
        if frame is not None:
            x, y, w, h = e.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite(f"check_out/{sample.name}_key{i}.png", frame)
print(f"Key presses with a nearby change: {hits}/{len(keys)}")