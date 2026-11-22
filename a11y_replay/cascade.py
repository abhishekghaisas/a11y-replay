"""Confidence gate: classical detectors first, escalate to the VLM only for what they cannot decide.

For focus visibility: a classical verdict of visible / not_visible with confidence at or above
`escalation_confidence` is final. An `ambiguous` result (or any below the gate) goes to the VLM with a
cropped before/after pair, and the model's answer maps to: yes -> visible, no -> not_visible,
unsure -> stays ambiguous (unresolved; never turned into a finding).

Every VLM call is counted so the report can state calls per minute of video and estimated cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from a11y_replay.config import DEFAULT, Config
from a11y_replay.detectors.focus import FocusResult, detect_focus
from a11y_replay.events import Finding, KeyEvent
from a11y_replay.ingest.video import probe, read_frame_at
from a11y_replay.taxonomy import WCAG_CRITERION, IssueType
from a11y_replay.vlm.client import VLMClient
from a11y_replay.vlm.pricing import estimate_cost
from a11y_replay.vlm.tasks import JudgeResult, judge_focus

ANSWER_TO_VERDICT = {"yes": "visible", "no": "not_visible", "unsure": "ambiguous"}


@dataclass(frozen=True)
class FinalFocusResult:
    key_t: float
    key: str
    verdict: str                       # visible | not_visible | ambiguous (unresolved)
    source: str                        # "classical" or "vlm"
    classical: FocusResult
    vlm: JudgeResult | None = None

    @property
    def reasoning(self) -> str:
        if self.vlm is not None:
            return f"Model judgment ({self.vlm.model}): {self.vlm.answer}. {self.vlm.evidence}".strip()
        return self.classical.reasoning


@dataclass
class CascadeStats:
    presses: int = 0
    escalated: int = 0
    vlm_calls: int = 0
    unresolved: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    video_minutes: float = 0.0
    est_cost_usd: float | None = None

    @property
    def calls_per_minute(self) -> float:
        return self.vlm_calls / self.video_minutes if self.video_minutes else 0.0


@dataclass
class CascadeOutput:
    results: list = field(default_factory=list)
    stats: CascadeStats = field(default_factory=CascadeStats)


def judge_press(client: VLMClient, video_path, r: FocusResult, fps: float, cfg: Config = DEFAULT) -> JudgeResult | None:
    """Read this press's before/after frames from the video and ask the model about them."""
    before = read_frame_at(video_path, r.before_frame / fps)
    after = read_frame_at(video_path, r.after_frame / fps)
    if before is None or after is None:
        return None
    return judge_focus(client, before, after, r.region, r.key, cfg)


def run_focus_cascade(video_path, keys: Iterable[KeyEvent], offset: float = 0.0,
                      client: VLMClient | None = None, cfg: Config = DEFAULT) -> CascadeOutput:
    info = probe(video_path)
    fps = info.fps or 25.0
    out = CascadeOutput()
    out.stats.video_minutes = info.duration / 60.0
    calls_before = client.meter.calls if client else 0
    tin_before = client.meter.input_tokens if client else 0
    tout_before = client.meter.output_tokens if client else 0

    for r in detect_focus(video_path, keys, offset=offset, cfg=cfg):
        out.stats.presses += 1
        needs_vlm = r.verdict != "unreadable" and (r.verdict == "ambiguous" or r.confidence < cfg.escalation_confidence)
        if not needs_vlm or client is None:
            out.results.append(FinalFocusResult(r.key_t, r.key, r.verdict, "classical", r))
            continue
        out.stats.escalated += 1
        j = judge_press(client, video_path, r, fps, cfg)
        if j is None:
            out.results.append(FinalFocusResult(r.key_t, r.key, r.verdict, "classical", r))
            continue
        verdict = ANSWER_TO_VERDICT[j.answer]
        out.results.append(FinalFocusResult(r.key_t, r.key, verdict, "vlm", r, j))
        if verdict == "ambiguous":
            out.stats.unresolved += 1

    if client:
        s = out.stats
        s.vlm_calls = client.meter.calls - calls_before
        s.input_tokens = client.meter.input_tokens - tin_before
        s.output_tokens = client.meter.output_tokens - tout_before
        s.est_cost_usd = estimate_cost(client.model, s.input_tokens, s.output_tokens)
    return out


def to_findings(results: Iterable[FinalFocusResult], cfg: Config = DEFAULT) -> list[Finding]:
    """Findings for confident failures, whether decided by the classical layer or by the model."""
    return [
        Finding(
            issue_type=IssueType.FOCUS_NOT_VISIBLE, criterion=WCAG_CRITERION[IssueType.FOCUS_NOT_VISIBLE],
            start=r.key_t, end=round(r.key_t + cfg.focus_window_s, 3),
            confidence=r.classical.confidence if r.source == "classical" else None,
            method=r.source, reasoning=r.reasoning,
            evidence_frames=(r.classical.before_frame, r.classical.after_frame),
        )
        for r in results if r.verdict == "not_visible"
    ]