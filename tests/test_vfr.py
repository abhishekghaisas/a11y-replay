"""Variable-frame-rate recordings (what a screen recorder produces for a still page). Needs ffmpeg and ffprobe."""
import json
import shutil
import subprocess
import unittest

import cv2

from a11y_replay.audit import run_audit
from a11y_replay.ingest.video import frame_times, is_variable_frame_rate, normalize_to_cfr, probe
from a11y_replay.segment import detect_changes
from tests.test_audit import FPS, KEYS, H, W, _frame

SECONDS = 15.0


def _need_ffmpeg():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise unittest.SkipTest("ffmpeg/ffprobe not available")


def _constant_video(path):
    """15 s at 25 fps: four marker flashes (1.5-4.5 s), a still page, a big blip at the very end so a recorder
    that only writes changed frames still writes a frame near 15 s."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(SECONDS * FPS)):
        t = i / FPS
        f = _frame(t, flash=False, toast=False)
        if t >= 14.7:
            f[0:120, 0:160] = 0
        out.write(f)
    out.release()


def _variable_copy(src, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "mpdecimate", "-vsync", "vfr",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)], check=True)


def test_a_screen_recording_style_file_is_detected_as_variable_rate(tmp_path):
    _need_ffmpeg()
    _constant_video(tmp_path / "const.avi")
    _variable_copy(tmp_path / "const.avi", tmp_path / "vfr.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "const.avi"), "-r", "25",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "cfr.mp4")], check=True)
    assert is_variable_frame_rate(tmp_path / "vfr.mp4")
    assert not is_variable_frame_rate(tmp_path / "cfr.mp4")
    assert not is_variable_frame_rate(tmp_path / "const.avi")


def test_detection_still_works_without_ffprobe_by_reading_frame_timestamps(tmp_path, monkeypatch=None):
    _need_ffmpeg()
    _constant_video(tmp_path / "const.avi")
    _variable_copy(tmp_path / "const.avi", tmp_path / "vfr.mp4")
    real = shutil.which
    shutil.which = lambda name, *a, **k: None if name == "ffprobe" else real(name, *a, **k)
    try:
        assert is_variable_frame_rate(tmp_path / "vfr.mp4")
        assert not is_variable_frame_rate(tmp_path / "const.avi")
    finally:
        shutil.which = real


def test_the_raw_file_misreports_time_and_the_converted_copy_does_not(tmp_path):
    _need_ffmpeg()
    _constant_video(tmp_path / "const.avi")
    _variable_copy(tmp_path / "const.avi", tmp_path / "vfr.mp4")
    raw = probe(tmp_path / "vfr.mp4")
    assert abs(raw.duration - SECONDS) > 3                       # frame count / fps is nowhere near the real 15 s
    assert normalize_to_cfr(tmp_path / "vfr.mp4", tmp_path / "fixed.mp4")
    fixed = probe(tmp_path / "fixed.mp4")
    assert abs(fixed.fps - 25) < 0.5 and abs(fixed.duration - SECONDS) < 0.3
    ts = frame_times(tmp_path / "fixed.mp4")
    assert abs(ts[-1] - SECONDS) < 0.3 and max(b - a for a, b in zip(ts, ts[1:])) < 0.1
    first = detect_changes(tmp_path / "fixed.mp4")[0]
    assert abs(first.t - 1.5) < 0.15, first                      # the first marker flash is where it really was


def test_an_audit_of_a_variable_rate_recording_lines_up_and_analyzes_every_press(tmp_path):
    _need_ffmpeg()
    rec = tmp_path / "rec"
    rec.mkdir()
    _constant_video(rec / "src.avi")
    _variable_copy(rec / "src.avi", rec / "video.mp4")
    (rec / "src.avi").unlink()
    (rec / "keys.jsonl").write_text("\n".join(json.dumps({"t": t, "key": k, "modifiers": []}) for t, k in KEYS))
    r = run_audit(rec, tmp_path / "out")
    assert r.sync["trusted"] and abs(r.sync["offset_s"]) < 0.2, r.sync
    focus = next(c["detail"] for c in r.checks if "2.4.7" in c["name"])
    assert "fell outside" not in focus and focus.startswith("1 Tab press(es)"), focus
    assert [i.criterion for i in r.findings] == ["2.4.7"] and abs(r.findings[0].start - 6.0) < 0.3
    assert "variable frame rate" in r.sync["note"]


def test_when_ffmpeg_is_missing_the_audit_warns_instead_of_pretending(tmp_path):
    _need_ffmpeg()
    rec = tmp_path / "rec"
    rec.mkdir()
    _constant_video(rec / "src.avi")
    _variable_copy(rec / "src.avi", rec / "video.mp4")
    (rec / "src.avi").unlink()
    (rec / "keys.jsonl").write_text("\n".join(json.dumps({"t": t, "key": k, "modifiers": []}) for t, k in KEYS))
    real = shutil.which
    shutil.which = lambda name, *a, **k: None if name in ("ffmpeg",) else real(name, *a, **k)
    try:
        r = run_audit(rec, tmp_path / "out")
    finally:
        shutil.which = real
    assert "could not be converted" in r.sync["note"] and "timing may be wrong" in r.sync["note"]