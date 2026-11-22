"""Pydantic models shared across the pipeline."""
from pathlib import Path

from pydantic import BaseModel

from a11y_replay.events import ChangeEvent, Finding, KeyEvent, Utterance  # noqa: F401  (re-exported)
from a11y_replay.taxonomy import WCAG_CRITERION, IssueType  # noqa: F401  (re-exported)


class Sample(BaseModel):
    """A recording bundle. keys_source matters for the real-world set."""
    root: Path
    has_audio: bool = False
    has_keys: bool = False
    keys_source: str = "logger"  # "logger" or "overlay"