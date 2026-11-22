import shutil

from a11y_replay.doctor import check_environment, main


def _by_name(results, needle):
    return next(r for r in results if needle in r[0])


def test_everything_present_reports_ready(tmp_path):
    (tmp_path / "ex1").mkdir()
    (tmp_path / "ex1" / "video.mp4").write_bytes(b"x")
    res = check_environment(tmp_path)
    assert _by_name(res, "OpenCV")[1] and _by_name(res, "NumPy")[1]
    assert _by_name(res, "Example recordings")[1] and "1 found" in _by_name(res, "Example recordings")[2]


def test_a_missing_ffmpeg_is_reported_as_required_and_blocks_readiness(capsys=None):
    real = shutil.which
    shutil.which = lambda name, *a, **k: None if name in ("ffmpeg", "ffprobe") else real(name, *a, **k)
    try:
        res = check_environment()
        assert not _by_name(res, "ffmpeg")[1] and _by_name(res, "ffmpeg")[3] is True
        assert "packages.txt" in _by_name(res, "ffmpeg")[2]
        assert main([]) == 1
    finally:
        shutil.which = real


def test_missing_examples_are_optional_not_blocking(tmp_path):
    res = check_environment(tmp_path / "does_not_exist")
    row = _by_name(res, "Example recordings")
    assert not row[1] and row[3] is False