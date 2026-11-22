"""UI-state segmentation: find moments the screen meaningfully changes.

Method: downscale -> grayscale -> blur -> absolute difference against the previous frame ->
threshold. A frame counts as "changed" if enough pixels differ. Consecutive changed frames
(allowing small gaps) are merged into one ChangeEvent episode.

This is intentionally simple and deterministic. SSIM or optical flow can replace the diff
later if the ablations show it matters. All thresholds live in config.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from a11y_replay.config import DEFAULT, Config
from a11y_replay.events import ChangeEvent
from a11y_replay.ingest.video import iter_frames, probe


def detect_changes(video_path: str | Path, cfg: Config = DEFAULT) -> list[ChangeEvent]:
    info = probe(video_path)
    k = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1

    prev = None
    # per changed frame: (time, fraction changed, (x0, y0, x1, y1) at analysis resolution)
    changed: list[tuple[float, float, tuple[int, int, int, int]]] = []
    scale_back = 1.0
    for _, t, frame in iter_frames(video_path, gray=True, width=cfg.analysis_width):
        scale_back = info.width / frame.shape[1]
        blurred = cv2.GaussianBlur(frame, (k, k), 0)
        if prev is not None:
            mask = cv2.absdiff(blurred, prev) > cfg.pixel_diff_threshold
            frac = float(mask.mean())
            if frac >= cfg.min_change_area_frac:
                x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                changed.append((t, frac, (x, y, x + w, y + h)))
        prev = blurred

    return _merge(changed, info.fps, scale_back, cfg.merge_gap_s)


def _merge(changed, fps: float, scale_back: float, gap_s: float) -> list[ChangeEvent]:
    if not changed:
        return []
    events: list[ChangeEvent] = []
    group = [changed[0]]
    for item in changed[1:]:
        if item[0] - group[-1][0] <= gap_s + 1e-9:
            group.append(item)
        else:
            events.append(_make_event(group, scale_back))
            group = [item]
    events.append(_make_event(group, scale_back))
    return events


def _make_event(group, scale_back: float) -> ChangeEvent:
    x0 = min(g[2][0] for g in group)
    y0 = min(g[2][1] for g in group)
    x1 = max(g[2][2] for g in group)
    y1 = max(g[2][3] for g in group)
    bbox = (
        int(round(x0 * scale_back)),
        int(round(y0 * scale_back)),
        int(round((x1 - x0) * scale_back)),
        int(round((y1 - y0) * scale_back)),
    )
    return ChangeEvent(
        t=round(group[0][0], 3),
        t_end=round(group[-1][0], 3),
        bbox=bbox,
        magnitude=round(max(g[1] for g in group), 5),
        n_frames=len(group),
    )