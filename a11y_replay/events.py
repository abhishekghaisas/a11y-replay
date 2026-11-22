"""Lightweight, dependency-free event types used inside the pipeline.

Pydantic models in schemas.py are for I/O boundaries (reports, findings). Internal events stay
as plain frozen dataclasses so they are cheap and easy to test.

Clocks: key events are on the LOG clock (the master clock). Change events come off the video
as VIDEO-clock times; sync.py estimates the offset and maps them onto the log clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from a11y_replay.taxonomy import IssueType


@dataclass(frozen=True)
class KeyEvent:
    """One line of keys.jsonl. The key log is trusted ground truth (master clock)."""
    t: float
    key: str
    modifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Utterance:
    """A transcribed screen-reader utterance."""
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ChangeEvent:
    """A contiguous episode of visual change found by UI-state segmentation.

    t / t_end are seconds on the VIDEO clock (frame_index / fps) until mapped by sync.py.
    """
    t: float                          # onset: first frame that differs from its predecessor
    t_end: float                      # last frame of the episode
    bbox: tuple[int, int, int, int]   # x, y, w, h of the union of changed pixels (full-res)
    magnitude: float                  # peak fraction of the frame that changed (0-1)
    n_frames: int                     # changed frames in the episode

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Finding:
    """One detected accessibility issue. Times are seconds on the key-log clock."""
    issue_type: IssueType
    criterion: str
    start: float
    end: float
    confidence: float | None          # classical: the detector's own score; model verdicts: None (no calibrated confidence)
    method: str = "classical"         # "classical" or "vlm"
    reasoning: str = ""
    evidence_frames: tuple = ()       # video frame indices worth showing as evidence

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issue_type"] = self.issue_type.value
        return d