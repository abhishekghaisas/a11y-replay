"""Parse keys.jsonl into KeyEvent objects (trusted ground truth)."""
from __future__ import annotations

import json
from pathlib import Path

from a11y_replay.events import KeyEvent


def load_keys(path: str | Path) -> list[KeyEvent]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                events.append(KeyEvent(t=float(d["t"]), key=d["key"],
                                       modifiers=tuple(d.get("modifiers", ()))))
    return sorted(events, key=lambda e: e.t)