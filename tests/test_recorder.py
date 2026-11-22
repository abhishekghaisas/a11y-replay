"""Key logger core tests with a fake clock and fake key objects. pynput and tkinter are never touched."""
import json
import random
import types

from a11y_replay.ingest.keys import load_keys
from a11y_replay.recorder import (
    KeyLogger,
    QuietTerminal,
    build_parser,
    check_permissions,
    describe,
    marker_schedule,
)

Key = lambda name: types.SimpleNamespace(name=name)     # like pynput.keyboard.Key.tab
Char = lambda ch: types.SimpleNamespace(char=ch)        # like pynput.keyboard.KeyCode(char="a")


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_describe_names_match_what_the_detectors_expect():
    assert describe(Key("tab")) == ("Tab", "navigation")
    assert describe(Key("enter")) == ("Enter", "navigation")
    assert describe(Key("space")) == ("Space", "navigation")
    assert describe(Key("esc")) == ("Escape", "navigation")
    assert describe(Key("down")) == ("ArrowDown", "navigation")
    assert describe(Key("f9")) == ("F9", "navigation")
    assert describe(Key("shift_r")) == ("Shift", "modifier") and describe(Key("cmd")) == ("Meta", "modifier")
    assert describe(Char("a")) == ("a", "text")
    assert describe(Key("caps_lock"))[1] == "other"


def test_navigation_keys_are_logged_with_times_relative_to_start(tmp_path):
    clock = Clock()
    log = KeyLogger(tmp_path / "keys.jsonl", clock=clock)
    clock.t = 101.5
    log.on_press(Key("tab"))
    clock.t = 103.25
    log.on_press(Key("enter"))
    log.close()
    assert _lines(tmp_path / "keys.jsonl") == [
        {"t": 1.5, "key": "Tab", "modifiers": []}, {"t": 3.25, "key": "Enter", "modifiers": []}]


def test_typed_characters_are_never_stored_by_default(tmp_path):
    """Privacy: a system-wide logger would otherwise capture passwords."""
    log = KeyLogger(tmp_path / "keys.jsonl", clock=Clock())
    for ch in "hunter2":
        log.on_press(Char(ch))
        log.on_release(Char(ch))
    log.on_press(Key("tab"))
    log.close()
    text = (tmp_path / "keys.jsonl").read_text()
    assert "hunter" not in text and "h" not in {d["key"] for d in _lines(tmp_path / "keys.jsonl")}
    assert len(_lines(tmp_path / "keys.jsonl")) == 1


def test_text_logging_is_an_explicit_opt_in(tmp_path):
    log = KeyLogger(tmp_path / "keys.jsonl", clock=Clock(), log_text=True)
    log.on_press(Char("a"))
    log.close()
    assert [d["key"] for d in _lines(tmp_path / "keys.jsonl")] == ["a"]


def test_modifiers_label_a_key_but_are_not_events_themselves(tmp_path):
    log = KeyLogger(tmp_path / "keys.jsonl", clock=Clock())
    log.on_press(Key("shift"))
    log.on_press(Key("tab"))                    # Shift+Tab: still a focus move
    log.on_release(Key("shift"))
    log.on_press(Key("tab"))
    log.close()
    out = _lines(tmp_path / "keys.jsonl")
    assert [(d["key"], d["modifiers"]) for d in out] == [("Tab", ["Shift"]), ("Tab", [])]


def test_stop_key_stops_the_listener_and_is_not_logged(tmp_path):
    log = KeyLogger(tmp_path / "keys.jsonl", clock=Clock(), stop_key="F10")
    log.on_press(Key("tab"))
    assert log.on_press(Key("f10")) is False and log.stopped
    assert [d["key"] for d in _lines(tmp_path / "keys.jsonl")] == ["Tab"]


def test_marker_flashes_are_logged_as_f9_and_load_cleanly_into_the_audit(tmp_path):
    clock = Clock()
    log = KeyLogger(tmp_path / "keys.jsonl", clock=clock)
    clock.t = 101.7
    log.log_marker()
    log.close()
    row = _lines(tmp_path / "keys.jsonl")[0]
    assert row["key"] == "F9" and row["source"] == "marker"
    k = load_keys(tmp_path / "keys.jsonl")[0]                    # the audit's own loader accepts the extra field
    assert k.key == "F9" and abs(k.t - 1.7) < 1e-9


def test_marker_schedule_is_irregular_and_reproducible_with_a_seed():
    s = marker_schedule(4, random.Random(1))
    gaps = [round(b - a, 3) for a, b in zip(s, s[1:])]
    assert len(s) == 4 and s[0] == 1.5 and all(0.7 <= g <= 1.6 for g in gaps) and len(set(gaps)) == len(gaps)
    assert marker_schedule(4, random.Random(1)) == s


def test_presses_are_counted_separately_from_calibration_flashes(tmp_path):
    log = KeyLogger(tmp_path / "keys.jsonl", clock=Clock())
    for _ in range(4):
        log.log_marker()
    log.on_press(Key("tab"))
    log.on_press(Key("tab"))
    assert (log.count, log.markers, log.presses) == (6, 4, 2)
    log.close()


def test_permission_check_reports_each_permission_or_unknown():
    assert check_permissions(lambda: True, lambda: True) == {"accessibility": True, "input_monitoring": True}
    assert check_permissions(lambda: False, lambda: True) == {"accessibility": False, "input_monitoring": True}
    boom = lambda: (_ for _ in ()).throw(RuntimeError("no such API"))
    assert check_permissions(boom, lambda: False) == {"accessibility": None, "input_monitoring": False}


def test_permission_check_does_not_crash_where_the_mac_apis_do_not_exist():
    r = check_permissions()      # on Linux CI neither API exists: must return unknowns, not raise
    assert set(r) == {"accessibility", "input_monitoring"}


def _pty():
    import unittest
    try:
        import os  # noqa: F401
        import pty
        import termios  # noqa: F401
        return pty.openpty()
    except (ImportError, OSError) as e:
        raise unittest.SkipTest(f"no pseudo-terminal here: {e}")


def test_terminal_echo_is_off_while_logging_and_restored_after():
    import termios
    master, slave = _pty()
    before = termios.tcgetattr(slave)[3]
    with QuietTerminal(slave):
        inside = termios.tcgetattr(slave)[3]
        assert not inside & termios.ECHO and not inside & termios.ICANON
        assert inside & termios.ISIG                      # Ctrl+C must keep working
    # Compare the flags that matter. macOS also sets a bookkeeping bit (PENDIN, 0x20000000) whenever the line mode changes.
    mask = termios.ECHO | termios.ICANON | termios.ISIG
    assert termios.tcgetattr(slave)[3] & mask == before & mask


def test_keys_typed_into_the_terminal_during_logging_are_discarded_not_run_by_the_shell_afterwards():
    """Regression: the first real run echoed '^[[A' junk, and anything typed stayed queued for the shell."""
    import os
    import select
    master, slave = _pty()
    with QuietTerminal(slave):
        os.write(master, b"rm -rf important\n")          # typed while the logger was running
    assert not select.select([slave], [], [], 0.3)[0]     # nothing is left for the shell to read


def test_quiet_terminal_is_harmless_when_there_is_no_terminal():
    with QuietTerminal(fd=None):                           # e.g. stdin is a pipe
        pass


def test_marker_position_is_parsed_into_two_numbers():
    assert build_parser().parse_args(["--marker-pos", "20,300"]).marker_pos == (20, 300)
    assert build_parser().parse_args([]).marker_pos == (10, 60)


def test_a_placeholder_marker_position_is_rejected_up_front_with_a_clear_message():
    """Regression: 'X,Y' pasted literally crashed AFTER logging had started, with a bare ValueError."""
    import contextlib
    import io
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            build_parser().parse_args(["--marker-pos", "X,Y"])
        raise AssertionError("should have been rejected")
    except SystemExit as e:
        assert e.code == 2
    msg = err.getvalue()
    assert "two whole numbers" in msg and "'X,Y'" in msg and "placeholders" in msg