"""Estimate and correct the video-to-key-log clock offset.

Convention:  offset = video_time - log_time.
    offset < 0  -> the video is AHEAD of the log (an effect shows up at an earlier video time)
    log_time = video_time - offset

Method: pair each key press with the nearest visual change whose onset falls in
[key.t - sync_pre_s, key.t + sync_post_s], take the median of (change.t - key.t). The median is
robust to a few mismatched pairs; the median absolute deviation (spread) says how consistent
the pairs are.

Limits (by design, and worth reporting):
  * It needs key presses that CAUSE a visible change. On a recording where the bug is
    "nothing visibly happens" (e.g. no focus indicator), there may be no pairs at all. The
    estimate is then marked not ok and the offset falls back to 0.0.
  * Unrelated changes close to a key press can pair with it and bias the estimate.
  * Only a constant offset is modeled. Clock drift over a long recording is not.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from statistics import median
from typing import Iterable, Sequence

from a11y_replay.config import DEFAULT, Config
from a11y_replay.events import ChangeEvent, KeyEvent


@dataclass(frozen=True)
class OffsetEstimate:
    offset: float                 # video_time - log_time, seconds
    n_pairs: int
    spread: float                 # median absolute deviation of the pair deltas, seconds
    ok: bool                      # enough consistent pairs to trust the estimate
    deltas: tuple[float, ...] = ()
    keys_used: tuple = ()         # names of the keys that were paired with a change, e.g. ("F9", "F9", "Tab")
    detail: tuple = ()            # one (key, key_time, delta, change_frac) per pair: delta = change time - key time (s),
                                  # change_frac = how much of the frame that change covered


def estimate_offset(
    keys: Sequence[KeyEvent], changes: Sequence[ChangeEvent], cfg: Config = DEFAULT
) -> OffsetEstimate:
    deltas: list[float] = []
    names: list[str] = []
    detail: list = []
    used: set[int] = set()
    for k in sorted(keys, key=lambda e: e.t):
        candidates = [
            (i, c) for i, c in enumerate(changes)
            if i not in used and k.t - cfg.sync_pre_s <= c.t <= k.t + cfg.sync_post_s
        ]
        if not candidates:
            continue
        i, c = min(candidates, key=lambda ic: abs(ic[1].t - k.t))
        used.add(i)
        deltas.append(c.t - k.t)
        names.append(k.key)
        detail.append((k.key, round(k.t, 3), round(c.t - k.t, 3), round(c.magnitude, 4)))

    if not deltas:
        return OffsetEstimate(offset=0.0, n_pairs=0, spread=float("inf"), ok=False)

    off = median(deltas)
    spread = median(abs(d - off) for d in deltas)
    ok = len(deltas) >= cfg.sync_min_pairs and spread <= cfg.sync_max_spread_s
    return OffsetEstimate(offset=off, n_pairs=len(deltas), spread=spread, ok=ok,
                          deltas=tuple(deltas), keys_used=tuple(names), detail=tuple(detail))


def estimate_offset_search(
    keys: Sequence[KeyEvent], changes: Sequence[ChangeEvent], cfg: Config = DEFAULT,
    max_abs_s: float = 120.0, bin_s: float = 0.1,
) -> OffsetEstimate:
    """Like estimate_offset, but for recordings whose video and key log started at DIFFERENT times
    (seconds apart, as happens when a person starts the screen recorder and the key logger by hand).

    Step 1: histogram of (change time - key time) over every pair within +-max_abs_s; the peak is the
    coarse offset, because the key presses that really caused changes all agree on it.
    Step 2: shift the key log by that and run the local estimator for the fine value and a trust flag.

    The key presses must be IRREGULARLY spaced for the peak to be unique: evenly spaced presses make the
    histogram repeat at multiples of the spacing. The recorder's calibration flashes are randomized for this.
    """
    deltas = [c.t - k.t for k in keys for c in changes if abs(c.t - k.t) <= max_abs_s]
    if not deltas:
        return OffsetEstimate(offset=0.0, n_pairs=0, spread=float("inf"), ok=False)
    counts = Counter(int(round(d / bin_s)) for d in deltas)
    score = {b: counts[b - 1] + counts[b] + counts[b + 1] for b in counts}
    best = max(score, key=lambda b: (score[b], -abs(b)))
    coarse = best * bin_s
    shifted = [replace(k, t=k.t + coarse) for k in keys]
    est = estimate_offset(shifted, changes, cfg)
    detail = tuple((n, round(t - coarse, 3), round(d + coarse, 3), m) for n, t, d, m in est.detail)   # back to the real clocks
    return replace(est, offset=est.offset + coarse, detail=detail)


def to_log_clock(changes: Iterable[ChangeEvent], offset: float) -> list[ChangeEvent]:
    """Map video-clock change events onto the key-log clock: log_time = video_time - offset."""
    return [replace(c, t=round(c.t - offset, 4), t_end=round(c.t_end - offset, 4))
            for c in changes]