"""Focus detector tests: pure frame classification plus a tiny synthetic video."""
import cv2
import numpy as np

from a11y_replay.detectors.focus import classify_focus_change, detect_focus, to_findings
from a11y_replay.events import KeyEvent

W, H = 640, 360
BG = 235


def _page():
    f = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(f, (100, 100), (260, 140), (120, 120, 120), 1)  # an "input" box
    cv2.putText(f, "Name", (110, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    return f


def _ring(f, box=(100, 100, 260, 140), pad=4, thick=3, color=(0, 140, 255)):
    x0, y0, x1, y1 = box
    cv2.rectangle(f, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), color, thick)
    return f


def test_no_change_is_not_visible():
    c = classify_focus_change(_page(), _page())
    assert c.verdict == "not_visible" and c.confidence >= 0.8


def test_clear_ring_is_visible():
    c = classify_focus_change(_page(), _ring(_page()))
    assert c.verdict == "visible" and c.contrast and c.contrast >= 1.5


def test_moving_ring_between_elements_is_visible():
    a = _ring(_page(), box=(100, 100, 260, 140))
    b = _ring(_page(), box=(300, 200, 420, 240))
    assert classify_focus_change(a, b).verdict == "visible"


def test_text_selection_highlight_is_ambiguous_not_a_failure():
    after = _page()
    cv2.rectangle(after, (104, 106, ), (150, 134), (215, 120, 30), -1)  # filled blue patch
    c = classify_focus_change(_page(), after)
    assert c.verdict == "ambiguous" and c.confidence < 0.7


def test_underline_only_is_ambiguous():
    after = _page()
    cv2.line(after, (100, 150), (260, 150), (0, 0, 0), 3)
    assert classify_focus_change(_page(), after).verdict == "ambiguous"


def test_ring_below_diff_threshold_is_not_visible():
    c = classify_focus_change(_page(), _ring(_page(), color=(225, 225, 225)))
    assert c.verdict == "not_visible"


def test_low_contrast_ring_is_ambiguous():
    # visible to the pixel diff, but luminance contrast between before and after is small
    c = classify_focus_change(_page(), _ring(_page(), color=(205, 205, 205)))
    assert c.verdict == "ambiguous" and c.contrast is not None and c.contrast < 1.5


def _video(path, ring_from=None, fps=25, seconds=4.0):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for i in range(int(seconds * fps)):
        f = _page()
        if ring_from is not None and i / fps >= ring_from:
            _ring(f)
        out.write(f)
    out.release()


def test_detect_focus_on_video_and_offset(tmp_path):
    keys = [KeyEvent(t=1.1, key="Tab"), KeyEvent(t=0.5, key="F9")]  # F9 = calibration, ignored
    with_ring = tmp_path / "ring.avi"; _video(with_ring, ring_from=1.0)  # video leads log by 0.1 s
    res = detect_focus(with_ring, keys, offset=-0.1)
    assert len(res) == 1 and res[0].verdict == "visible" and res[0].key == "Tab"

    without = tmp_path / "none.avi"; _video(without, ring_from=None)
    res = detect_focus(without, keys, offset=-0.1)
    assert [r.verdict for r in res] == ["not_visible"]
    findings = to_findings(res)
    assert len(findings) == 1 and findings[0].criterion == "2.4.7" and findings[0].start == 1.1


def test_ambiguous_results_do_not_become_findings(tmp_path):
    p = tmp_path / "hl.avi"
    out = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 25, (W, H))
    for i in range(100):
        f = _page()
        if i / 25 >= 1.0:
            cv2.rectangle(f, (104, 106), (150, 134), (215, 120, 30), -1)
        out.write(f)
    out.release()
    res = detect_focus(p, [KeyEvent(t=1.0, key="Tab")], offset=0.0)
    assert res[0].verdict == "ambiguous" and to_findings(res) == []


def test_rings_a_few_pixels_apart_are_judged_separately():
    """Focus moving between nearby elements: the old ring vanishes, the new one appears. With a
    clear gap they are separate regions, so the appearing one is found."""
    a = _ring(_page(), box=(100, 100, 260, 140))
    b = _ring(_page(), box=(100, 160, 260, 200))  # about 8 px of gap between the two rings
    c = classify_focus_change(a, b)
    assert c.verdict == "visible" and c.contrast >= 1.5, c


def test_touching_rings_merge_and_are_escalated_never_judged_confidently():
    """When the old and new rings are only ~2 px apart the pixel mask merges them into one region
    whose direction (appeared vs disappeared) cannot be told. The safe answer is 'ambiguous'
    (escalate); a confident verdict either way would be a guess. Known limitation."""
    a = _ring(_page(), box=(100, 100, 260, 140))
    b = _ring(_page(), box=(100, 153, 260, 193))
    c = classify_focus_change(a, b)
    assert c.verdict == "ambiguous" and c.confidence < 0.7, c


def test_previous_indicator_disappearing_is_not_a_new_indicator():
    """Regression: focus moves to a control with NO indicator while the previous control's ring
    disappears. The screen changes, but nothing appeared on the newly focused element."""
    c = classify_focus_change(_ring(_page()), _page())
    assert c.verdict == "not_visible", c


def _dotted(f, box=(100, 100, 260, 140), pad=4):
    """A 1 px dotted black outline (every other pixel), like a thin dotted focus outline."""
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    for x in range(x0, x1 + 1, 2):
        f[y0, x] = f[y1, x] = (0, 0, 0)
    for y in range(y0, y1 + 1, 2):
        f[y, x0] = f[y, x1] = (0, 0, 0)
    return f


def test_thin_dotted_outline_appearing_is_visible():
    """The changed pixels are mostly light halo, so median-based direction and contrast tests see
    almost nothing. The extreme (darkest) pixels show a high-contrast dark outline appeared."""
    c = classify_focus_change(_page(), _dotted(_page()))
    assert c.verdict == "visible" and c.contrast > 5, c


def test_thin_dotted_outline_disappearing_is_not_a_new_indicator():
    c = classify_focus_change(_dotted(_page()), _page())
    assert c.verdict == "not_visible", c


def _stroke_fixture(highlight=True):
    """Pale-blue text highlight (macOS default selection color) over dark glyph strokes that do not
    change. The strokes are unchanged pixels enclosed by changed ones, so the changed region has many
    small holes. Drawn as bars so the hole share is controlled (text rendering varies by platform)."""
    f = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(f, (100, 100), (330, 146), (120, 120, 120), 1)
    if highlight:
        cv2.rectangle(f, (110, 108), (310, 138), (255, 208, 165), -1)
    for x in range(116, 306, 10):                       # 19 dark bars, unchanged between frames
        cv2.rectangle(f, (x, 112), (x + 5, 134), (30, 30, 30), -1)
    return f


def _hole_shares(before, after):
    g0 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    g1 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    mask = ((cv2.absdiff(g0, g1) > 25) * 255).astype(np.uint8)
    cs, h = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    i = max((i for i in range(len(cs)) if h[0][i][3] == -1), key=lambda i: cv2.contourArea(cs[i]))
    holes = [cv2.contourArea(cs[j]) for j in range(len(cs)) if h[0][j][3] == i]
    outer = cv2.contourArea(cs[i])
    return sum(holes) / outer, max(holes) / outer


def test_text_highlight_with_dark_text_staying_dark_is_not_a_ring():
    """Regression (seen on macOS recordings): the browser highlights the text but the glyphs stay
    dark. Their TOTAL area can reach 30-35% of the changed region, but each hole is tiny, whereas a
    ring encloses one large interior. Summing the holes mistook this for a ring."""
    before, after = _stroke_fixture(False), _stroke_fixture(True)
    total, largest = _hole_shares(before, after)
    assert total >= 0.30 and largest < 0.05, (total, largest)   # the fixture really has the failure mode
    c = classify_focus_change(before, after)
    assert c.verdict == "ambiguous" and "ring-like outline appeared" not in c.reasoning, c


def test_ring_around_a_button_with_text_still_counts_as_one_large_interior():
    f = _stroke_fixture(False)
    after = _ring(f.copy(), box=(100, 100, 330, 146))
    assert classify_focus_change(f, after).verdict == "visible"