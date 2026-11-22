import contextlib
import io
import os
import time

import cv2

from a11y_replay.diagnose import analysis_video, contact_sheet, main
from tests.test_audit import _recording


def test_diagnose_lists_changes_and_keys_and_writes_a_contact_sheet(tmp_path):
    rec = _recording(tmp_path / "rec", flash=False, toast=False)         # marker flashes at 1.5, 2.5, 3.5, 4.5 s
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([str(rec), "--from", "0", "--to", "6"])
    out = buf.getvalue()
    assert "Screen changes the audit sees" in out and "1.5" in out and "4.5" in out
    assert "calibration flash" not in out                               # these keys carry no source field, so no label
    assert "F9" in out and "Contact sheet written" in out
    sheet = cv2.imread(str(rec / "audit_report" / "diagnose_0-6s.png"))
    assert sheet is not None and sheet.shape[0] > 100 and sheet.shape[1] >= 4 * 100


def test_the_contact_sheet_has_one_labeled_tile_per_step(tmp_path):
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    img = contact_sheet(next(rec.glob("video.*")), 0.0, 2.0, step=0.5, cols=3, width=160)
    # 5 tiles (0, 0.5, 1, 1.5, 2.0) in 3 columns -> 2 rows
    assert img.shape[1] == 3 * 160 and img.shape[0] >= 2 * 80


def test_a_converted_copy_older_than_the_video_is_not_reused(tmp_path):
    """Recording again into the same folder must not show frames from the previous recording."""
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    video = next(rec.glob("video.*"))
    copy = rec / "audit_report" / "analysis_video.mp4"
    copy.parent.mkdir()
    copy.write_bytes(b"x" * 2000)
    old = time.time() - 3600
    os.utime(copy, (old, old))                                          # the copy predates the video
    assert analysis_video(rec, tmp_path / "work") == video
    now = time.time() + 5
    os.utime(copy, (now, now))                                          # a copy made after the video is reused
    assert analysis_video(rec, tmp_path / "work") == copy