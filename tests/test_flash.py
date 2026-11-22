"""Flash detector tests on synthetic videos (no browser needed)."""
import cv2
import numpy as np

from a11y_replay.detectors.flash import detect_flashes, find_transitions, relative_luminance

FPS = 25
W, H = 320, 240


def _video(path, frame_fn, seconds=4.0):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(seconds * FPS)):
        out.write(frame_fn(i, i / FPS))
    out.release()


def _gray(v):
    return np.full((H, W, 3), v, np.uint8)


def _toggle(period_frames, a=0, b=255, start_s=0.0, end_s=99.0, region=None):
    """Alternate between gray levels a and b every `period_frames` frames inside [start_s, end_s]."""
    def fn(i, t):
        f = _gray(a)
        if start_s <= t < end_s and (i // period_frames) % 2 == 1:
            if region is None:
                f[:] = b
            else:
                x0, y0, x1, y1 = region
                f[y0:y1, x0:x1] = b
        return f
    return fn


def test_relative_luminance_endpoints():
    assert relative_luminance(_gray(0)).max() == 0.0
    assert abs(relative_luminance(_gray(255)).min() - 1.0) < 1e-3


def test_static_video_has_no_findings(tmp_path):
    p = tmp_path / "s.avi"; _video(p, lambda i, t: _gray(128))
    assert detect_flashes(p) == []


def test_fast_full_frame_flashing_is_flagged(tmp_path):
    p = tmp_path / "fast.avi"; _video(p, _toggle(3))  # ~8.3 toggles/s = ~4.2 flashes/s
    found = detect_flashes(p)
    assert len(found) == 1 and found[0].criterion == "2.3.1"
    assert "flashes within one second" in found[0].reasoning


def test_slow_flashing_at_or_below_three_per_second_passes(tmp_path):
    p = tmp_path / "slow.avi"; _video(p, _toggle(6))  # ~4.2 toggles/s = ~2.1 flashes/s
    assert detect_flashes(p) == []


def test_large_patch_flagged_small_patch_ignored(tmp_path):
    big = tmp_path / "big.avi"; _video(big, _toggle(3, region=(60, 40, 260, 200)))   # ~42% of frame
    small = tmp_path / "small.avi"; _video(small, _toggle(3, region=(10, 10, 50, 50)))  # ~2% of frame
    assert len(detect_flashes(big)) == 1
    assert detect_flashes(small) == []


def test_low_contrast_flicker_is_not_a_flash(tmp_path):
    p = tmp_path / "dim.avi"; _video(p, _toggle(3, a=100, b=112))  # luminance swing far below 10%
    assert detect_flashes(p) == []


def test_bright_both_states_is_not_a_general_flash(tmp_path):
    # swing is > 10% but the darker image is >= 0.80 relative luminance
    p = tmp_path / "bright.avi"; _video(p, _toggle(3, a=236, b=255))
    assert detect_flashes(p) == []


def test_slow_fade_is_not_a_flash(tmp_path):
    p = tmp_path / "fade.avi"; _video(p, lambda i, t: _gray(int(255 * min(t / 3.5, 1.0))))
    assert detect_flashes(p) == []


def test_onset_time_and_offset_mapping(tmp_path):
    p = tmp_path / "burst.avi"; _video(p, _toggle(3, start_s=1.0, end_s=2.6), seconds=4.0)
    f0 = detect_flashes(p)[0]
    assert abs(f0.start - 1.0) <= 0.2 and f0.end <= 2.8
    f1 = detect_flashes(p, offset=-0.07)[0]
    assert abs((f1.start - f0.start) - 0.07) < 1e-6  # log_time = video_time - offset


def test_two_separate_bursts_make_two_findings(tmp_path):
    def fn(i, t):
        on = (0.5 <= t < 1.8) or (3.0 <= t < 4.3)
        return _toggle(3)(i, t) if on else _gray(0)
    p = tmp_path / "two.avi"; _video(p, fn, seconds=5.0)
    assert len(detect_flashes(p)) == 2


def test_find_transitions_counts_alternating_swings():
    lum = np.array([0, 1, 0, 1, 0], dtype=np.float32).reshape(5, 1, 1)
    assert find_transitions(lum, 0.1, 0.8)[:, 0, 0].sum() == 4


def test_rule_boundary_at_25fps(tmp_path):
    """Documents the boundary: flashes = transitions // 2, so a 1 s window with 6-7 transitions
    counts as 3 flashes (passes) and 8-9 transitions counts as 4 (flagged). A stricter reading
    that counts every transition above 6 would flag the 4-frame case too."""
    four = tmp_path / "p4.avi"; _video(four, _toggle(4), seconds=3.0)    # ~6.25 toggles/s
    three = tmp_path / "p3.avi"; _video(three, _toggle(3), seconds=3.0)  # ~8.33 toggles/s
    assert detect_flashes(four) == []
    assert len(detect_flashes(three)) == 1


def _scrolling_text(px_per_s, pitch=12, ink=4, word=16, gap=16):
    """Lines of 'text' (words with gaps) scrolling up, about 17% ink like real text. Proportions matter: the regional
    average only stays put if a grid cell holds several lines. Here a cell (grid 4) is 90 px tall over 12 px spacing
    (7.5 lines); a real 720p screen has ~30 lines, so ~7 per cell. A cell holding only 1 to 3 lines would still flicker."""
    def fn(i, t):
        f = _gray(255)
        off = int(t * px_per_s)
        for y in range(-pitch, H + pitch, pitch):
            yy = y - (off % pitch)
            for x in range(40, W - 40, word + gap):
                f[max(yy, 0):max(yy + ink, 0), x:x + word] = 30
        return f
    return fn


def test_scrolling_text_is_not_flashing(tmp_path):
    """Regression from the first real recording: judged pixel by pixel, ordinary scrolling text (a pixel
    alternates between ink and paper several times a second) was reported as 'up to 11 flashes within one
    second over 82% of the frame'. A real flash changes the average brightness of a REGION, not just single
    pixels, and scrolling text does not."""
    for speed in (60, 120, 300, 600):          # 5, 10, 25, 50 lines per second passing any pixel
        p = tmp_path / f"scroll{speed}.avi"; _video(p, _scrolling_text(speed))
        assert detect_flashes(p) == [], speed


def test_a_real_flash_is_still_found_when_text_is_on_screen(tmp_path):
    def fn(i, t):
        f = _scrolling_text(300)(i, t)
        if 1.0 <= t < 3.0 and (i // 3) % 2 == 1:
            f[:] = 0                                                    # whole screen goes black every few frames
        return f
    p = tmp_path / "text_and_flash.avi"; _video(p, fn)
    assert len(detect_flashes(p)) == 1


def test_known_limit_fine_pattern_flicker_is_not_detected(tmp_path):
    """Documented miss: a checkerboard that inverts every few frames flashes at the pixel level but its regional
    average stays constant, so this detector (which needs regional agreement) does not report it."""
    def fn(i, t):
        f = _gray(0)
        yy, xx = np.mgrid[0:H, 0:W]
        board = ((yy // 8 + xx // 8) % 2 == 0)
        on = (int(t * FPS) // 3) % 2 == 1
        f[board ^ on] = 255
        return f
    p = tmp_path / "checker.avi"; _video(p, fn)
    assert detect_flashes(p) == []


def test_known_limit_huge_bands_scrolling_fast_can_still_be_reported(tmp_path):
    """Documented: the regional test needs a region to hold several lines. Very large, high-contrast bands whose spacing
    is 1 to 2 cells (here 120 px in a 90 px cell), scrolling fast, still alternate at the regional level. Whether that
    counts as flashing depends on how one reads the criterion; the tool reports it."""
    p = tmp_path / "huge.avi"; _video(p, _scrolling_text(600, pitch=120, ink=60, word=60, gap=0))
    assert len(detect_flashes(p)) >= 1