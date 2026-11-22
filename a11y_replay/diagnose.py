"""Look at what is actually in a recording over a stretch of time, to find out why clock alignment failed.

    python -m a11y_replay.diagnose recordings/focus_bug --from 0 --to 12

Prints every screen change the audit would see in that range (time, size, where) next to the key log's events in
the same range, and writes a contact sheet of frames, `diagnose_<from>-<to>s.png`, into the report folder, so you
can see whether the calibration flashes are in the video at all. Upload the PNG and paste the printed lists.
The contact sheet shows your screen recording, so look before sharing.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from a11y_replay.ingest.video import is_variable_frame_rate, normalize_to_cfr, probe, read_frame_at
from a11y_replay.segment import detect_changes


def analysis_video(rec: Path, work: Path) -> Path:
    """The same video the audit analyzes: its constant-rate copy if one exists or is needed."""
    existing = rec / "audit_report" / "analysis_video.mp4"
    videos = sorted(rec.glob("video.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    video = videos[0] if videos else None
    if video is None:
        raise FileNotFoundError(f"No video.* in {rec}")
    if existing.exists() and existing.stat().st_mtime >= video.stat().st_mtime:   # a copy made BEFORE the video was replaced is stale
        return existing
    if is_variable_frame_rate(video) and normalize_to_cfr(video, work / "analysis_video.mp4"):
        return work / "analysis_video.mp4"
    return video


def contact_sheet(video: Path, t0: float, t1: float, step: float = 0.25, cols: int = 4, width: int = 320,
                  events: list | None = None) -> np.ndarray:
    """Frames from t0 to t1 every `step` seconds, each labeled with its time; tiles where a change begins are outlined."""
    events = events or []
    tiles = []
    t = t0
    while t <= t1 + 1e-9:
        f = read_frame_at(video, t)
        if f is None:
            f = np.full((int(width * 0.6), width, 3), 60, np.uint8)
        else:
            f = cv2.resize(f, (width, max(1, int(f.shape[0] * width / f.shape[1]))), interpolation=cv2.INTER_AREA)
        hit = any(t - step / 2 <= e.t < t + step / 2 for e in events)
        if hit:
            cv2.rectangle(f, (0, 0), (f.shape[1] - 1, f.shape[0] - 1), (0, 0, 255), 4)
        cv2.rectangle(f, (0, 0), (92, 22), (0, 0, 0), -1)
        cv2.putText(f, f"{t:6.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(f)
        t = round(t + step, 6)
    h = max(x.shape[0] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for x in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 30))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--from", dest="t0", type=float, default=0.0, help="start, seconds on the VIDEO clock")
    ap.add_argument("--to", dest="t1", type=float, default=12.0)
    ap.add_argument("--step", type=float, default=0.25)
    a = ap.parse_args(argv)
    rec = Path(a.recording)
    with tempfile.TemporaryDirectory() as tmp:
        video = analysis_video(rec, Path(tmp))
        info = probe(video)
        events = [e for e in detect_changes(video) if a.t0 - 0.5 <= e.t <= a.t1 + 0.5]
        print(f"{rec.name}: analyzed video {video.name}, {info.duration:.1f} s at {info.fps:.0f} fps, frame {info.width}x{info.height}")
        print(f"\nScreen changes the audit sees between {a.t0:.1f} s and {a.t1:.1f} s of the VIDEO:")
        for e in events:
            x, y, w, h = e.bbox
            print(f"  {e.t:7.2f}-{e.t_end:7.2f}s  {w}x{h} px at ({x},{y})  {e.magnitude * 100:5.2f}% of the frame")
        if not events:
            print("  (none)")
        keys_file = rec / "keys.jsonl"
        if keys_file.exists():
            rows = [json.loads(x) for x in keys_file.read_text().splitlines() if x.strip()]
            print("\nKey log events in the same numeric range (KEY-LOG clock; the two clocks may differ):")
            for r in rows:
                if a.t0 - 0.5 <= r["t"] <= a.t1 + 0.5:
                    print(f"  {r['t']:7.2f}s  {r['key']}{'  (calibration flash)' if r.get('source') == 'marker' else ''}")
        out = rec / "audit_report"
        out.mkdir(exist_ok=True)
        path = out / f"diagnose_{a.t0:g}-{a.t1:g}s.png"
        cv2.imwrite(str(path), contact_sheet(video, a.t0, a.t1, a.step, events=events))
        print(f"\nContact sheet written to {path}  (tiles outlined in red are where a change begins)")


if __name__ == "__main__":
    main()