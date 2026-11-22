"""Cascade tests on tiny synthetic videos with a scripted model."""
import cv2
import numpy as np

from a11y_replay.cascade import run_focus_cascade, to_findings
from a11y_replay.events import KeyEvent
from a11y_replay.vlm.client import MockClient

W, H, FPS = 640, 360, 25
YES = '{"indicator_visible": "yes", "indicator_type": "fill", "evidence": "dark fill"}'
NO = '{"indicator_visible": "no", "indicator_type": "", "evidence": "nothing changed"}'
UNSURE = '{"indicator_visible": "unsure", "indicator_type": "", "evidence": "cannot tell"}'


def _page():
    f = np.full((H, W, 3), 235, np.uint8)
    cv2.rectangle(f, (100, 100), (260, 140), (120, 120, 120), 1)
    return f


def _video(path, effect, at=1.0, seconds=3.0):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(seconds * FPS)):
        f = _page()
        if effect and i / FPS >= at:
            effect(f)
        out.write(f)
    out.release()


def _ring(f):
    cv2.rectangle(f, (96, 96), (264, 144), (0, 140, 255), 3)


def _highlight(f):  # a solid patch: classical layer cannot decide, so it escalates
    cv2.rectangle(f, (104, 106), (150, 134), (215, 120, 30), -1)


KEYS = [KeyEvent(t=1.0, key="Tab")]


def test_confident_classical_results_never_call_the_model(tmp_path):
    p = tmp_path / "ring.avi"; _video(p, _ring)
    client = MockClient([YES])
    out = run_focus_cascade(p, KEYS, client=client)
    assert out.results[0].verdict == "visible" and out.results[0].source == "classical"
    assert client.meter.calls == 0 and out.stats.escalated == 0 and out.stats.calls_per_minute == 0


def test_nothing_changed_is_a_classical_failure_without_a_model_call(tmp_path):
    p = tmp_path / "none.avi"; _video(p, None)
    client = MockClient([YES])
    out = run_focus_cascade(p, KEYS, client=client)
    assert out.results[0].verdict == "not_visible" and client.meter.calls == 0
    assert len(to_findings(out.results)) == 1


def test_ambiguous_press_is_escalated_and_model_yes_means_visible(tmp_path):
    p = tmp_path / "hl.avi"; _video(p, _highlight)
    client = MockClient([YES])
    out = run_focus_cascade(p, KEYS, client=client)
    r = out.results[0]
    assert r.source == "vlm" and r.verdict == "visible" and to_findings(out.results) == []
    assert out.stats.escalated == 1 and out.stats.vlm_calls == 1 and out.stats.calls_per_minute > 0
    assert out.stats.est_cost_usd is None  # mock model has no price


def test_model_no_becomes_a_finding_marked_as_model_decided(tmp_path):
    p = tmp_path / "hl2.avi"; _video(p, _highlight)
    out = run_focus_cascade(p, KEYS, client=MockClient([NO]))
    f = to_findings(out.results)
    assert len(f) == 1 and f[0].method == "vlm" and f[0].confidence is None and f[0].criterion == "2.4.7"


def test_model_unsure_stays_unresolved_and_is_never_a_finding(tmp_path):
    p = tmp_path / "hl3.avi"; _video(p, _highlight)
    out = run_focus_cascade(p, KEYS, client=MockClient([UNSURE]))
    assert out.results[0].verdict == "ambiguous" and out.stats.unresolved == 1
    assert to_findings(out.results) == []


def test_without_a_client_ambiguous_results_stay_classical(tmp_path):
    p = tmp_path / "hl4.avi"; _video(p, _highlight)
    out = run_focus_cascade(p, KEYS, client=None)
    assert out.results[0].verdict == "ambiguous" and out.results[0].source == "classical"
    assert out.stats.vlm_calls == 0


def test_calibration_key_is_ignored(tmp_path):
    p = tmp_path / "cal.avi"; _video(p, _highlight)
    client = MockClient([YES])
    out = run_focus_cascade(p, [KeyEvent(t=1.0, key="F9")], client=client)
    assert out.results == [] and client.meter.calls == 0