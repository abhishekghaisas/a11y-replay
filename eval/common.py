"""Helpers shared by the evaluation scripts."""
from __future__ import annotations

from pathlib import Path

from a11y_replay.ingest.keys import load_keys
from a11y_replay.segment import detect_changes
from a11y_replay.sync import estimate_offset


def sample_offset(sample: Path, meta: dict, video: Path):
    """Return (offset, trusted). Uses the generator's stored qa result if present, else estimates."""
    qa = meta.get("qa")
    if qa and qa.get("ok"):
        return qa["offset_s"], True
    est = estimate_offset(load_keys(sample / "keys.jsonl"), detect_changes(video))
    return est.offset, est.ok


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by reference length. Uses the same normalization as the
    matcher (lowercase, no punctuation, numbers spelled out), so "3" and "three" are not an error."""
    from a11y_replay.detectors.announce import normalize
    r, h = normalize(reference).split(), normalize(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)