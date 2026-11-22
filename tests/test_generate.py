"""Unit tests for the benchmark generator that need no browser."""
import random

from a11y_replay.taxonomy import IssueType
from benchmark.generate import (
    STYLE_EXPECTED,
    Recorder,
    make_label,
    record_validated,
    scenario_focus_outline,
    scenario_focus_styles,
    scenario_status_messages_noisy,
)


class FakeKeyboard:
    def __init__(self):
        self.pressed = []

    def press(self, key):
        self.pressed.append(key)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakePage:
    """Stands in for a Playwright page. Its waits advance a fake clock so scenario timings are real."""

    def __init__(self, clock=None):
        self.keyboard = FakeKeyboard()
        self.clock = clock

    def wait_for_timeout(self, ms):
        if self.clock:
            self.clock.t += ms / 1000.0

    def evaluate(self, *args, **kwargs):
        pass

    def focus(self, selector):
        pass


def test_make_label_maps_criterion():
    label = make_label(IssueType.FLASHING, 2.4081, 4.4231)
    assert label["criterion"] == "2.3.1"
    assert label["t"] == 2.408 and label["t_end"] == 4.423


def test_focus_scenario_bug_labels_every_tab():
    rec = Recorder(FakePage(), t0=0.0)
    labels = scenario_focus_outline(rec, random.Random(0), bug=True)
    tabs = [k for k in rec.keys if k["key"] == "Tab"]
    assert len(labels) == len(tabs) >= 4  # one label per Tab press, none for calibration keys
    assert all(l["criterion"] == "2.4.7" for l in labels)


def test_focus_scenario_clean_has_no_labels():
    rec = Recorder(FakePage(), t0=0.0)
    assert scenario_focus_outline(rec, random.Random(0), bug=False) == []
    assert len(rec.keys) >= 4


def test_scenarios_are_seed_deterministic():
    a, b = Recorder(FakePage(), 0.0), Recorder(FakePage(), 0.0)
    scenario_focus_outline(a, random.Random(7), bug=True)
    scenario_focus_outline(b, random.Random(7), bug=True)
    assert [k["key"] for k in a.keys] == [k["key"] for k in b.keys]


def test_calibration_keys_come_first_and_are_never_labeled():
    from a11y_replay.config import DEFAULT
    rec = Recorder(FakePage(), t0=0.0)
    labels = scenario_focus_outline(rec, random.Random(1), bug=True)
    keys = [k["key"] for k in rec.keys]
    assert keys[:4] == [DEFAULT.calibration_key] * 4
    assert all(l.get("tab_index") for l in labels)  # labels exist only for Tab presses


def test_record_validated_retries_until_ok():
    results = iter([{"ok": False}, {"ok": False}, {"ok": True}])
    calls = []
    path, qa, attempts = record_validated(lambda: calls.append(1) or "p", lambda _: next(results))
    assert (path, attempts, len(calls)) == ("p", 3, 3) and qa["ok"]


def test_record_validated_gives_up_after_max_attempts():
    path, qa, attempts = record_validated(lambda: "p", lambda _: {"ok": False}, max_attempts=2)
    assert path is None and attempts == 2 and not qa["ok"]


def test_record_validated_does_not_retry_on_first_success():
    calls = []
    _, _, attempts = record_validated(lambda: calls.append(1) or "p", lambda _: {"ok": True})
    assert attempts == 1 and len(calls) == 1


def test_focus_styles_covers_every_style_once_and_labels_only_failures():
    rec = Recorder(FakePage(), t0=0.0)
    labels = scenario_focus_styles(rec, random.Random(3), bug=False)
    checks = rec.extra["focus_checks"]
    assert sorted(c["style"] for c in checks) == sorted(STYLE_EXPECTED)
    assert {c["style"] for c in checks if c["expected"] == "not_visible"} == {l["style"] for l in labels}
    assert all(l["issue_type"] == "focus_not_visible" for l in labels)
    assert [k["key"] for k in rec.keys].count("Tab") == len(STYLE_EXPECTED)


def test_focus_styles_order_is_shuffled_per_seed():
    orders = set()
    for seed in range(5):
        rec = Recorder(FakePage(), t0=0.0)
        scenario_focus_styles(rec, random.Random(seed), bug=False)
        orders.add(tuple(c["style"] for c in rec.extra["focus_checks"]))
    assert len(orders) > 1


def _noisy(seed):
    clock = FakeClock()
    rec = Recorder(FakePage(clock), t0=0.0, clock=clock)
    labels = scenario_status_messages_noisy(rec, random.Random(seed), bug=False)
    return rec, labels


def test_noisy_scenario_puts_wrong_speech_only_inside_the_window_of_silent_messages():
    for seed in range(8):
        rec, labels = _noisy(seed)
        utts = rec.extra["utterances"]
        for c in rec.extra["status_checks"]:
            near = [u for u in utts if c["t"] - 0.2 <= u["start"] <= c["t"] + 3.0]
            if c["announced"]:
                assert c["distractors"] == [] and len(near) == 1        # exactly its own announcement
            else:
                assert len(c["distractors"]) == 2 and len(near) == 2
                assert {d["category"] for d in c["distractors"]} - {"unrelated"} <= {
                    "near_miss", "confusable", "truncated", "wrong_number", "negation"}


def test_noisy_scenario_keeps_utterances_apart_so_the_audio_version_stays_separable():
    for seed in range(8):
        utts = _noisy(seed)[0].extra["utterances"]
        for a, b in zip(utts, utts[1:]):
            assert b["start"] - a["end"] >= 0.8, (seed, a, b)


def test_noisy_silent_messages_are_still_labeled_failures_whatever_was_said_nearby():
    rec, labels = _noisy(3)
    silent = [c for c in rec.extra["status_checks"] if not c["announced"]]
    assert len(labels) == len(silent) >= 1 and all(l["criterion"] == "4.1.3" for l in labels)


def test_noisy_scenario_always_has_a_silent_and_an_announced_message_and_covers_every_kind_over_seeds():
    silent_kinds = set()
    for seed in range(30):
        rec, labels = _noisy(seed)
        checks = rec.extra["status_checks"]
        assert any(c["announced"] for c in checks) and any(not c["announced"] for c in checks)
        silent_kinds |= {c["kind"] for c in checks if not c["announced"]}
    assert silent_kinds == {"toast", "banner", "inline"}