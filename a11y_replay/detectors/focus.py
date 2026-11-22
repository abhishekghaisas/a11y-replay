"""WCAG 2.4.7 Focus Visible, driven by key events (classical layer).

For each focus key press (Tab) at log time t, compare the frame just before with the frame just
after and decide:

  not_visible : essentially nothing changed on screen            -> a finding (confident)
  visible     : the change contains a ring-like outline with enough contrast
  ambiguous   : something changed but it is not clearly a focus indicator (a text-selection
                highlight, an underline, a faint ring, a fill). These are the cases a VLM
                should judge; the cascade will escalate results below the confidence gate.

"Ring-like" = a closed outline of changed pixels enclosing ONE large unchanged interior. This catches
the common focus ring but not every legitimate indicator (e.g. an underline or a color change of
a whole control are reported as ambiguous, not as failures).

Direction matters: when focus moves, the previous element's indicator DISAPPEARS, which also changes
the screen. Each changed region is therefore tested for whether it appeared or disappeared, and only
appearing regions can count as an indicator on the newly focused element.

Known limits: two changed regions that touch (an old ring vanishing right next to a new one) can
merge into one region whose direction is unclear, which is reported as ambiguous (escalated, never
a confident wrong answer); indicators fainter than `pixel_diff_threshold` are treated as not visible;
thresholds are untuned.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from a11y_replay.config import DEFAULT, Config
from a11y_replay.detectors.flash import relative_luminance
from a11y_replay.events import Finding, KeyEvent
from a11y_replay.ingest.video import is_blank_frame, probe, read_frame_at
from a11y_replay.taxonomy import WCAG_CRITERION, IssueType


@dataclass(frozen=True)
class Classification:
    verdict: str                       # "visible" | "not_visible" | "ambiguous"
    confidence: float
    reasoning: str
    bbox: tuple | None = None          # (x, y, w, h) of the best ring-like region
    contrast: float | None = None      # median per-pixel luminance contrast, before vs after, in that region
    region: tuple | None = None        # (x, y, w, h) covering ALL changed regions, for the VLM crop


@dataclass(frozen=True)
class FocusResult:
    key_t: float                       # key press time on the log clock
    key: str
    verdict: str
    confidence: float
    reasoning: str
    bbox: tuple | None
    contrast: float | None
    before_frame: int                  # video frame indices used as evidence
    after_frame: int
    region: tuple | None = None        # union of all changed regions (x, y, w, h)


def _stands_out(lum: np.ndarray, region_px: np.ndarray, surround: float) -> float:
    """How strongly a region differs from its surroundings, judged by its most extreme pixels
    (10th and 90th percentile) and not the median. Thin or sparse structures such as a 1 px dotted
    outline are mostly light halo after blurring, so the median looks like the background."""
    v = lum[region_px]
    return max(abs(float(np.percentile(v, 10)) - surround), abs(float(np.percentile(v, 90)) - surround))


def _direction(lum0, lum1, region_px, filled, mask, cfg) -> str:
    """Did this changed region APPEAR (it stands out from its surroundings only in `after`) or
    DISAPPEAR (only in `before`)? Compares the region's luminance with a ring of surrounding pixels
    in each frame. Returns "appeared", "disappeared" or "unclear"."""
    near = cv2.dilate(filled, np.ones((5, 5), np.uint8)) > 0
    far = cv2.dilate(filled, np.ones((21, 21), np.uint8)) > 0
    annulus = far & ~near & ~(mask > 0)
    if annulus.sum() < 20 or region_px.sum() == 0:
        return "unclear"
    d_before = _stands_out(lum0, region_px, float(np.median(lum0[annulus])))
    d_after = _stands_out(lum1, region_px, float(np.median(lum1[annulus])))
    if d_after - d_before >= cfg.focus_direction_delta:
        return "appeared"
    if d_before - d_after >= cfg.focus_direction_delta:
        return "disappeared"
    return "unclear"


def _union(boxes):
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def classify_focus_change(before: np.ndarray, after: np.ndarray, cfg: Config = DEFAULT) -> Classification:
    """Classify the visual change between two BGR frames of the same size.

    Every changed region is described by its shape (ring-like outline with an unchanged interior,
    or solid) and its direction (appeared / disappeared). Only regions that APPEARED can be a focus
    indicator on the newly focused element: the previous element losing its indicator also changes
    the screen, but proves nothing about the new one.
    """
    g0 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    g1 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    mask = ((cv2.absdiff(g0, g1) > cfg.pixel_diff_threshold) * 255).astype(np.uint8)

    changed_frac = float((mask > 0).mean())
    if changed_frac < cfg.focus_min_change_area_frac:
        return Classification(
            "not_visible", 0.9,
            f"Almost nothing changed on screen after the key press ({changed_frac * 100:.3f}% of "
            f"pixels; minimum {cfg.focus_min_change_area_frac * 100:.3f}%), so no focus indicator appeared.",
        )

    lum0, lum1 = relative_luminance(before), relative_luminance(after)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    comps = []  # dicts: kind, direction, bbox, contrast, hole_frac
    if hierarchy is not None:
        h = hierarchy[0]
        for i, c in enumerate(contours):
            if h[i][3] != -1:  # outer contours only
                continue
            holes = [contours[j] for j in range(len(contours)) if h[j][3] == i]
            filled = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(filled, [c], -1, 255, -1)
            body = filled.copy()
            for k in holes:
                cv2.drawContours(body, [k], -1, 0, -1)
            px = (body > 0) & (mask > 0)
            if int(px.sum()) < cfg.focus_min_component_px:
                continue
            x, y, w, hh = cv2.boundingRect(c)
            outer_area = cv2.contourArea(c)
            # LARGEST single hole, not the total: a ring encloses one big interior, while a text
            # highlight over dark text is full of small holes (the glyph strokes) that can add up
            # to a third of the region
            hole_frac = (max((cv2.contourArea(k) for k in holes), default=0.0) / outer_area) if outer_area > 0 else 0.0
            ring = w >= 8 and hh >= 8 and hole_frac >= cfg.focus_ring_hole_frac
            l0, l1 = lum0[px], lum1[px]
            # PER-PIXEL contrast (averaging luminance first would cancel out wherever one thing gets
            # lighter and another darker in the same region), summarized by the 90th percentile:
            # thin outlines are mostly light halo after blurring, so the median would understate them
            contrast = float(np.percentile((np.maximum(l0, l1) + 0.05) / (np.minimum(l0, l1) + 0.05), 90))
            comps.append({
                "kind": "ring" if ring else "solid",
                "direction": _direction(lum0, lum1, px, filled, mask, cfg),
                "bbox": (x, y, w, hh), "contrast": contrast, "hole_frac": hole_frac,
            })

    region = _union([c["bbox"] for c in comps])
    appeared = [c for c in comps if c["direction"] == "appeared"]
    unclear = [c for c in comps if c["direction"] == "unclear"]
    rings = [c for c in appeared if c["kind"] == "ring"]

    if rings:
        best = max(rings, key=lambda c: c["contrast"])
        if best["contrast"] >= cfg.focus_min_contrast:
            return Classification(
                "visible", 0.9,
                f"A ring-like outline appeared (contrast {best['contrast']:.1f}:1, interior "
                f"{best['hole_frac'] * 100:.0f}% of outline).", best["bbox"], best["contrast"], region,
            )
        return Classification(
            "ambiguous", 0.5,
            f"A ring-like outline appeared but with low contrast ({best['contrast']:.2f}:1 < "
            f"{cfg.focus_min_contrast}:1). A semantic check is needed.", best["bbox"], best["contrast"], region,
        )
    if appeared or unclear:
        what = appeared[0] if appeared else unclear[0]
        return Classification(
            "ambiguous", 0.4,
            f"Something changed ({changed_frac * 100:.2f}% of pixels) but not as a clear ring: "
            f"{'a solid region appeared (text highlight, underline or fill?)' if appeared else 'direction unclear'}. "
            f"A semantic check is needed.", what["bbox"], what["contrast"], region,
        )
    return Classification(
        "not_visible", 0.8,
        f"The screen changed ({changed_frac * 100:.2f}% of pixels) but only by things disappearing "
        f"(e.g. the previous element's indicator going away); nothing new appeared on the newly focused element.",
        region=region,
    )


def detect_focus(
    video_path: str | Path, keys: Iterable[KeyEvent], offset: float = 0.0, cfg: Config = DEFAULT
) -> list[FocusResult]:
    """Classify every focus-key press. `offset` is video_time - log_time (see sync.py)."""
    fps = probe(video_path).fps or 25.0
    results = []
    for k in keys:
        if k.key not in cfg.focus_keys:
            continue  # includes the calibration key
        t_before = max(0.0, k.t + offset - cfg.focus_before_s)
        t_after = k.t + offset + cfg.focus_window_s
        before = read_frame_at(video_path, t_before)
        after = read_frame_at(video_path, t_after)
        if before is None or after is None or before.shape != after.shape:
            continue  # key press falls outside the recording
        if is_blank_frame(before, cfg.blank_mean_luma, cfg.blank_p995_luma) or \
                is_blank_frame(after, cfg.blank_mean_luma, cfg.blank_p995_luma):
            results.append(FocusResult(
                key_t=k.t, key=k.key, verdict="unreadable", confidence=0.0,
                reasoning="The recorded area is black at this moment (the window may have left the recorded area, changed Space, "
                          "or the screen locked), so 'nothing changed' says nothing about whether a focus indicator appeared.",
                bbox=None, contrast=None, before_frame=int(round(t_before * fps)), after_frame=int(round(t_after * fps))))
            continue
        c = classify_focus_change(before, after, cfg)
        results.append(FocusResult(
            key_t=k.t, key=k.key, verdict=c.verdict, confidence=c.confidence, reasoning=c.reasoning,
            bbox=c.bbox, contrast=c.contrast,
            before_frame=int(round(t_before * fps)), after_frame=int(round(t_after * fps)),
            region=c.region,
        ))
    return results


def to_findings(results: Iterable[FocusResult], cfg: Config = DEFAULT) -> list[Finding]:
    """Confident failures only. Ambiguous results are left for the cascade (VLM) to decide."""
    return [
        Finding(
            issue_type=IssueType.FOCUS_NOT_VISIBLE,
            criterion=WCAG_CRITERION[IssueType.FOCUS_NOT_VISIBLE],
            start=r.key_t, end=round(r.key_t + cfg.focus_window_s, 3),
            confidence=r.confidence, method="classical", reasoning=r.reasoning,
            evidence_frames=(r.before_frame, r.after_frame),
        )
        for r in results if r.verdict == "not_visible"
    ]