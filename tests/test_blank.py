"""A recording whose recorded area goes black (window moved to another Space, minimized, screen locked)."""
import json

import cv2
import numpy as np

from a11y_replay.audit import run_audit
from a11y_replay.cascade import run_focus_cascade
from a11y_replay.detectors.focus import detect_focus
from a11y_replay.events import KeyEvent
from a11y_replay.ingest.video import blank_stretches, is_blank_frame
from a11y_replay.vlm.client import MockClient
from tests.test_audit import FPS, H, KEYS, W, _frame


def _video(path, black=((2.0, 9.0),), seconds=12.0, ring_at=None):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(seconds * FPS)):
        t = i / FPS
        f = _frame(t, flash=False, toast=False, ring_at=ring_at)
        if any(a <= t < b for a, b in black):
            f = np.zeros((H, W, 3), np.uint8)
            f[200:204, 300:304] = 200                                # a mouse cursor is all that is left
        out.write(f)
    out.release()


def test_black_frames_are_recognized_and_a_normal_page_is_not():
    assert is_blank_frame(np.zeros((90, 160, 3), np.uint8))
    page = np.full((90, 160, 3), 245, np.uint8)
    assert not is_blank_frame(page)
    dark_ui = np.full((90, 160, 3), 30, np.uint8)
    dark_ui[10:40, 10:100] = 220
    assert not is_blank_frame(dark_ui)                               # a dark page with content is not blank


def test_black_stretches_are_found_with_their_bounds(tmp_path):
    _video(tmp_path / "v.avi", black=((2.0, 5.0), (7.0, 9.5)))
    stretches, _ = blank_stretches(tmp_path / "v.avi")
    assert len(stretches) == 2
    assert abs(stretches[0][0] - 2.0) < 0.15 and abs(stretches[0][1] - 5.0) < 0.15
    assert abs(stretches[1][0] - 7.0) < 0.15 and abs(stretches[1][1] - 9.5) < 0.15


def test_a_normal_recording_has_no_black_stretches(tmp_path):
    _video(tmp_path / "v.avi", black=())
    assert blank_stretches(tmp_path / "v.avi")[0] == []


def test_a_tab_press_on_a_black_screen_is_unreadable_not_not_visible(tmp_path):
    """Regression from a real recording: two black frames are identical, so 'nothing changed' was read as
    'no focus indicator appeared'."""
    _video(tmp_path / "v.avi")
    res = detect_focus(tmp_path / "v.avi", [KeyEvent(t=5.0, key="Tab"), KeyEvent(t=10.5, key="Tab")])
    assert [r.verdict for r in res] == ["unreadable", "not_visible"]     # the second is on the visible page, a real failure
    assert "black" in res[0].reasoning


def test_unreadable_presses_are_never_sent_to_a_model(tmp_path):
    _video(tmp_path / "v.avi")
    client = MockClient(['{"indicator_visible": "yes", "indicator_type": "x", "evidence": "x"}'])
    out = run_focus_cascade(tmp_path / "v.avi", [KeyEvent(t=5.0, key="Tab")], client=client)
    assert out.results[0].verdict == "unreadable" and client.meter.calls == 0


def test_the_audit_warns_about_a_black_recording_and_reports_unreadable_presses(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    _video(rec / "video.avi")
    (rec / "keys.jsonl").write_text("\n".join(json.dumps({"t": t, "key": k, "modifiers": []}) for t, k in KEYS))
    r = run_audit(rec, tmp_path / "out")
    assert r.warnings and "is black" in r.warnings[0] and "7 s" in r.warnings[0]
    focus = next(c["detail"] for c in r.checks if "2.4.7" in c["name"])
    assert "unreadable (the screen was black)" in focus
    assert [i for i in r.items if "screen was black" in i.title] and not [i for i in r.findings if i.criterion == "2.4.7"]
    page = (tmp_path / "out" / "report.html").read_text()
    assert 'role="alert"' in page and page.index("Warning.") < page.index('id="summary"') + 400
    assert json.loads((tmp_path / "out" / "report.json").read_text())["warnings"]


def test_a_normal_recording_gets_no_blank_warning(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    _video(rec / "video.avi", black=())
    (rec / "keys.jsonl").write_text("\n".join(json.dumps({"t": t, "key": k, "modifiers": []}) for t, k in KEYS))
    assert run_audit(rec, tmp_path / "out").warnings == []