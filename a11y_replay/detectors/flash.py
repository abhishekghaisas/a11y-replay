"""WCAG 2.3.1 Three Flashes or Below Threshold: general-flash detector (classical, no VLM).

What it implements (an APPROXIMATION of the general flash definition):
  * relative luminance per pixel (WCAG formula, sRGB linearized) on a downscaled frame
  * a TRANSITION is a swing between a valley and a peak of at least `flash_luminance_delta`
    (10% of max relative luminance) where the darker image is below `flash_dark_max` (0.80)
  * a FLASH is two opposing transitions, so a pixel flashes `transitions // 2` times
  * a pixel is "flashing" in a 1-second window if it flashes more than `flash_max_per_second`
  * a pixel only counts if the average brightness of its REGION (a cell of a `flash_grid` x `flash_grid`
    division) flashes by the same rule, so scrolling text, whose pixels alternate while the region average
    stays put, is not reported (first real recording: 14 false findings from scrolling and window switching)
  * a finding needs the flashing pixels to cover at least `flash_min_area_frac` of the frame

What it does NOT do (say so in any write-up):
  * red flashes (saturated red) are not detected
  * no check of the "below 50 Hz" bound, and a 25 fps recording cannot show flashes faster
    than 12.5 Hz at all (aliasing), so very fast flashing can be missed
  * area is a fraction of the whole frame; the criterion's area is "contiguous" within a
    10-degree field, which we do not model
  * fine patterns that invert (a checkerboard) have a constant regional average and are NOT detected
  * frames are averaged in sRGB space when downscaling, not in linear light
  * video compression and the recording frame rate affect luminance swings
It is a screening aid, not a compliance test. Use a dedicated tool for conformance claims.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from a11y_replay.config import DEFAULT, Config
from a11y_replay.events import Finding
from a11y_replay.ingest.video import iter_frames, probe
from a11y_replay.taxonomy import WCAG_CRITERION, IssueType


def relative_luminance(frame_bgr: np.ndarray) -> np.ndarray:
    """WCAG relative luminance (0 = black, 1 = white) of a BGR uint8 frame."""
    rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def find_transitions(lum: np.ndarray, delta: float, dark_max: float) -> np.ndarray:
    """Per-pixel transitions. lum is (T, H, W); returns a bool array (T, H, W).

    A transition is recorded at the frame where a swing of at least `delta` between the last
    valley and the running peak (or the reverse) is confirmed, provided the darker side of the
    swing is below `dark_max`.
    """
    T, H, W = lum.shape
    trans = np.zeros((T, H, W), dtype=bool)
    lo = lum[0].copy()
    hi = lum[0].copy()
    direction = np.zeros((H, W), dtype=np.int8)  # 0 undecided, +1 last swing up, -1 last swing down
    for t in range(1, T):
        L = lum[t]
        und, up, dn = direction == 0, direction == 1, direction == -1
        hi = np.where(und | up, np.maximum(hi, L), hi)
        lo = np.where(und | dn, np.minimum(lo, L), lo)
        go_up = (und | dn) & ((L - lo) >= delta)
        go_dn = (und | up) & ((hi - L) >= delta)
        trans[t] = (go_up & (lo < dark_max)) | (go_dn & (L < dark_max))
        direction = np.where(go_up, 1, np.where(go_dn, -1, direction)).astype(np.int8)
        hi = np.where(go_up, L, hi)
        lo = np.where(go_dn, L, lo)
    return trans


def detect_flashes(
    video_path: str | Path, cfg: Config = DEFAULT, offset: float = 0.0
) -> list[Finding]:
    """Return one Finding per flashing episode. `offset` is video_time - log_time (see sync.py);
    returned times are on the key-log clock (video time minus offset)."""
    fps = probe(video_path).fps or 25.0
    lum = np.stack([
        relative_luminance(f) for _, _, f in iter_frames(video_path, width=cfg.flash_analysis_width)
    ])
    g = cfg.flash_grid
    blk = np.stack([cv2.resize(l, (g, g), interpolation=cv2.INTER_AREA) for l in lum])   # regional averages
    return _detect(lum, fps, cfg, offset, blk)


def _windowed_flashes(trans: np.ndarray, win: int) -> np.ndarray:
    """Flashes (pairs of transitions) in every 1 s window, per cell. trans is (T, ...)."""
    T = trans.shape[0]
    zeros = np.zeros((1,) + trans.shape[1:], np.int16)
    cs = np.concatenate([zeros, np.cumsum(trans, axis=0, dtype=np.int16)])
    counts = cs[win:] - cs[:-win] if T > win else (cs[-1:] - cs[:1])
    return counts // 2


def _detect(lum: np.ndarray, fps: float, cfg: Config, offset: float, blk: np.ndarray | None = None) -> list[Finding]:
    T, H, W = lum.shape
    if T < 2:
        return []
    trans = find_transitions(lum, cfg.flash_luminance_delta, cfg.flash_dark_max)

    win = min(T, max(1, int(round(fps))))  # frames in one second
    flashes = _windowed_flashes(trans, win)
    flagged = flashes > cfg.flash_max_per_second            # (n_windows, H, W), per pixel
    if blk is not None:                                      # ...and the pixel's region must flash as a whole
        bt = find_transitions(blk, cfg.flash_luminance_delta, cfg.flash_dark_max)
        region = _windowed_flashes(bt, win) > cfg.flash_max_per_second      # (n_windows, g, g)
        rows = (np.arange(H) * blk.shape[1]) // H
        cols = (np.arange(W) * blk.shape[2]) // W
        flagged = flagged & region[:, rows][:, :, cols]
    frac = flagged.mean(axis=(1, 2))
    violating = frac >= cfg.flash_min_area_frac
    if not violating.any():
        return []

    # group violating windows into episodes (allow small gaps)
    idx = np.flatnonzero(violating)
    gap = max(1, win // 2)
    episodes, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            episodes.append((start, prev))
            start = i
        prev = i
    episodes.append((start, prev))

    findings = []
    for w0, w1 in episodes:
        onset, end, peak, area = None, None, 0, 0.0
        for f in range(w0, w1 + 1):
            if not violating[f]:
                continue
            tr = (trans[f:f + win] & flagged[f]).any(axis=(1, 2))
            hits = np.flatnonzero(tr)
            if hits.size:
                onset = f + hits[0] if onset is None else min(onset, f + hits[0])
                end = f + hits[-1] if end is None else max(end, f + hits[-1])
            peak = max(peak, int(flashes[f][flagged[f]].max()))
            area = max(area, float(frac[f]))
        if onset is None:
            continue
        findings.append(Finding(
            issue_type=IssueType.FLASHING,
            criterion=WCAG_CRITERION[IssueType.FLASHING],
            start=round(onset / fps - offset, 3),
            end=round(end / fps - offset, 3),
            confidence=1.0,
            method="classical",
            reasoning=(
                f"Up to {peak} flashes within one second over {area * 100:.1f}% of the frame "
                f"(rule: more than {cfg.flash_max_per_second} flashes over at least "
                f"{cfg.flash_min_area_frac * 100:.1f}% of the frame; luminance swing >= "
                f"{cfg.flash_luminance_delta:.0%}, darker side < {cfg.flash_dark_max}). "
                f"General flash only; approximation, not a compliance test."
            ),
            evidence_frames=(int(onset), int((onset + end) // 2), int(end)),
        ))
    return findings