"""One ordered timeline of key presses, visual changes, and speech, on the key-log clock.

The key log is the master clock. Visual changes are mapped onto it using the offset from
sync.py; speech is shifted by an audio offset (default 0, to be estimated separately).
Detectors read from the timeline and never touch raw inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from a11y_replay.config import DEFAULT, Config
from a11y_replay.events import ChangeEvent, KeyEvent, Utterance
from a11y_replay.sync import to_log_clock

KINDS = ("key", "change", "speech")


@dataclass(frozen=True)
class Item:
    kind: str       # "key" | "change" | "speech"
    t: float        # onset on the log clock
    t_end: float    # end on the log clock (== t for keys)
    data: Any       # the underlying KeyEvent / ChangeEvent / Utterance


class Timeline:
    def __init__(self, items: Iterable[Item]):
        order = {k: i for i, k in enumerate(KINDS)}
        self._items = sorted(items, key=lambda it: (it.t, order[it.kind]))

    @classmethod
    def build(
        cls,
        keys: Iterable[KeyEvent],
        changes: Iterable[ChangeEvent],
        utterances: Iterable[Utterance] = (),
        video_offset: float = 0.0,
        audio_offset: float = 0.0,
    ) -> "Timeline":
        """video_offset / audio_offset follow sync.py: offset = stream_time - log_time."""
        items = [Item("key", k.t, k.t, k) for k in keys]
        items += [Item("change", c.t, c.t_end, c) for c in to_log_clock(changes, video_offset)]
        items += [Item("speech", u.start - audio_offset, u.end - audio_offset, u) for u in utterances]
        return cls(items)

    # ---- access ----
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items)

    def items(self, kind: str | None = None) -> list[Item]:
        return [it for it in self._items if kind is None or it.kind == kind]

    # ---- queries ----
    def window(self, t0: float, t1: float, kind: str | None = None) -> list[Item]:
        """Items whose ONSET lies in [t0, t1]."""
        return [it for it in self._items if t0 <= it.t <= t1 and (kind is None or it.kind == kind)]

    def after(self, item: Item, within: float, kind: str | None = None,
              before: float = 0.0) -> list[Item]:
        """Items of `kind` with onset in [item.t - before, item.t + within], excluding item."""
        return [it for it in self.window(item.t - before, item.t + within, kind) if it is not item]

    def effects_of(self, key_item: Item, cfg: Config = DEFAULT) -> list[Item]:
        """Visual changes plausibly caused by a key press."""
        return self.after(key_item, cfg.effect_after_s, kind="change", before=cfg.effect_before_s)