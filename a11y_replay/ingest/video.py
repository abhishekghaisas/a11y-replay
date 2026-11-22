"""Video loading and frame extraction (OpenCV).

All times are seconds on the video clock: t = frame_index / fps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        return VideoInfo(
            path=path,
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(
    path: str | Path, gray: bool = False, width: int | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, time_s, frame). Optionally convert to grayscale and/or downscale to `width`."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if width and frame.shape[1] > width:
                scale = width / frame.shape[1]
                frame = cv2.resize(
                    frame, (width, max(1, int(round(frame.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            yield i, i / fps, frame
            i += 1
    finally:
        cap.release()


def read_frame_at(path: str | Path, t: float) -> np.ndarray | None:
    """Return the (BGR) frame nearest time t, or None if out of range."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def crop(frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> np.ndarray:
    """Crop (x, y, w, h) from a frame with optional padding, clamped to the frame."""
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return frame[y0:y1, x0:x1]


# ---------------------------------------------------------------------------------------------------------------
# Variable frame rate. macOS screen recordings only emit a frame when the screen changes, so a still page produces
# very few frames. Every module here turns a frame NUMBER into seconds as number / fps, which is only right for a
# constant frame rate. Such a video is converted to a constant-rate copy before it is analyzed.
import json as _json
import shutil as _shutil
import subprocess as _subprocess


def _fraction(text: str) -> float:
    try:
        a, _, b = (text or "0").partition("/")
        return float(a) / float(b or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def container_info(path: str | Path):
    """{'r_fps', 'avg_fps', 'duration'} from ffprobe, or None if ffprobe is unavailable or fails."""
    if not _shutil.which("ffprobe"):
        return None
    r = _subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=r_frame_rate,avg_frame_rate,duration:format=duration", "-of", "json", str(path)],
                        capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        d = _json.loads(r.stdout)
        s = (d.get("streams") or [{}])[0]
        dur = s.get("duration") or d.get("format", {}).get("duration")
        return {"r_fps": _fraction(s.get("r_frame_rate")), "avg_fps": _fraction(s.get("avg_frame_rate")),
                "duration": float(dur) if dur not in (None, "N/A") else None}
    except (ValueError, KeyError, IndexError):
        return None


def frame_times(path: str | Path) -> list:
    """The real timestamp of every frame, in seconds, as the decoder reports them."""
    cap = cv2.VideoCapture(str(path))
    ts = []
    try:
        while cap.grab():
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()
    return ts


def is_variable_frame_rate(path: str | Path) -> bool:
    """True if frame numbers cannot be converted to time with one fps value.

    Uses ffprobe (the stream's nominal rate versus its average rate) when available, otherwise scans the frame
    timestamps: a gap of more than 3 times the median, and over 0.15 s, means frames were skipped."""
    info = container_info(path)
    if info and info["r_fps"] > 0 and info["avg_fps"] > 0:
        return abs(info["r_fps"] - info["avg_fps"]) / info["r_fps"] > 0.05
    ts = frame_times(path)
    if len(ts) < 3:
        return False
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    med = sorted(gaps)[len(gaps) // 2]
    return max(gaps) > max(3 * med, 0.15)


def normalize_to_cfr(src: str | Path, dst: str | Path, fps: int = 25, max_width: int = 1280) -> bool:
    """Write a constant-rate (and, if larger, downscaled) copy of src to dst with ffmpeg. Frames are repeated through
    static stretches, and the last frame is held so the copy is as long as the original container. Returns success."""
    if not _shutil.which("ffmpeg"):
        return False
    # The container's own duration field is NOT reliable for variable-rate files (it can be frame count / average rate,
    # far shorter than the real recording), so the length comes from the decoder's real frame timestamps.
    info = container_info(src)
    ts = frame_times(src)
    dur = max([ts[-1] + 1.0 / fps if ts else 0.0, (info or {}).get("duration") or 0.0]) or None
    vf = f"fps={fps},scale='min({max_width},iw)':-2"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an"]
    if dur:
        vf += f",tpad=stop_mode=clone:stop_duration={dur:.3f}"
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dst)]
    r = _subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and Path(dst).exists() and Path(dst).stat().st_size > 1000


# ---------------------------------------------------------------------------------------------------------------
# Black frames. When the recorded area shows nothing (the window moved to another Space, was minimized, the screen locked),
# "nothing changed" does not mean "no focus indicator". Every check must know the screen was blank.
def is_blank_frame(frame: np.ndarray, mean_max: float = 8.0, p995_max: float = 40.0) -> bool:
    """True if the frame is (almost) entirely black. A lone mouse cursor does not count as content."""
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(g.mean()) < mean_max and float(np.percentile(g, 99.5)) < p995_max


def blank_stretches(path: str | Path, mean_max: float = 8.0, p995_max: float = 40.0, min_len: float = 1.0):
    """(stretches, duration): runs of black frames at least `min_len` seconds long, as (start, end) in video seconds."""
    runs, start, last_t, duration = [], None, 0.0, 0.0
    for _, t, frame in iter_frames(path, gray=True, width=64):
        duration = t
        if is_blank_frame(frame, mean_max, p995_max):
            if start is None:
                start = t
            last_t = t
        elif start is not None:
            if t - last_t > 0.3:                      # a brief non-black blip (the cursor, a flash) does not end a stretch
                runs.append((start, last_t)); start = None
    if start is not None:
        runs.append((start, last_t))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= 0.3:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(round(a, 2), round(b, 2)) for a, b in merged if b - a >= min_len], duration