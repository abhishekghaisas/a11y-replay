"""The VLM's jobs. Only one is implemented so far:

    judge_focus(before, after, region)  -> did the newly focused element gain a visible indicator?

Still to come: classify_change (WCAG 4.1.3) and find_pause_control (WCAG 2.2.2).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import cv2
import numpy as np

from a11y_replay.config import DEFAULT, Config
from a11y_replay.vlm import prompts
from a11y_replay.vlm.client import VLMClient

ANSWERS = ("yes", "no", "unsure")


@dataclass(frozen=True)
class JudgeResult:
    answer: str                # "yes" | "no" | "unsure"
    indicator_type: str
    evidence: str
    parse_ok: bool             # False if the model never produced valid JSON (answer forced to unsure)
    raw_text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    attempts: int
    prompt_version: str = prompts.PROMPT_VERSION


def parse_judge_json(text: str):
    """Extract the answer object from a model reply, tolerating code fences and stray text."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    candidates = [t]
    if "{" in t and "}" in t:
        candidates.append(t[t.index("{"): t.rindex("}") + 1])
    for c in candidates:
        try:
            d = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(d, dict) and str(d.get("indicator_visible", "")).strip().lower() in ANSWERS:
            return {
                "answer": str(d["indicator_visible"]).strip().lower(),
                "indicator_type": str(d.get("indicator_type", "") or ""),
                "evidence": str(d.get("evidence", "") or ""),
            }
    return None


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()


def crop_region(frame: np.ndarray, region, cfg: Config = DEFAULT) -> np.ndarray:
    """Crop `region` (x, y, w, h) with padding, clamp to the frame, and downscale big crops."""
    H, W = frame.shape[:2]
    if region is None:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x, y, w, h = region
        p = cfg.vlm_crop_pad_px
        x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    out = frame[y0:y1, x0:x1]
    longest = max(out.shape[:2])
    if longest > cfg.vlm_max_side_px:
        s = cfg.vlm_max_side_px / longest
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return out


def build_focus_content(before_png: bytes, after_png: bytes, key: str = "Tab") -> list:
    return [
        {"type": "text", "text": prompts.FOCUS_JUDGE_INTRO.format(key=key)},
        {"type": "text", "text": "BEFORE the key press:"},
        {"type": "image", "png": before_png},
        {"type": "text", "text": "AFTER the key press:"},
        {"type": "image", "png": after_png},
        {"type": "text", "text": prompts.FOCUS_JUDGE_QUESTION},
    ]


def judge_focus(client: VLMClient, before: np.ndarray, after: np.ndarray, region=None,
                key: str = "Tab", cfg: Config = DEFAULT) -> JudgeResult:
    """Ask the model the closed focus question about a before/after frame pair. One retry on bad JSON."""
    content = build_focus_content(_png(crop_region(before, region, cfg)),
                                  _png(crop_region(after, region, cfg)), key)
    tin = tout = attempts = 0
    lat = 0.0
    parsed, text = None, ""
    for attempt in range(2):
        asked = content if attempt == 0 else content + [{"type": "text", "text": prompts.RETRY_REMINDER}]
        r = client.complete(asked)
        attempts += 1
        tin, tout, lat, text = tin + r.input_tokens, tout + r.output_tokens, lat + r.latency_s, r.text
        parsed = parse_judge_json(r.text)
        if parsed:
            break
    return JudgeResult(
        answer=parsed["answer"] if parsed else "unsure",
        indicator_type=parsed["indicator_type"] if parsed else "",
        evidence=parsed["evidence"] if parsed else "",
        parse_ok=bool(parsed), raw_text=text, model=client.model,
        input_tokens=tin, output_tokens=tout, latency_s=lat, attempts=attempts,
    )