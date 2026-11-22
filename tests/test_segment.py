"""Segmentation tests on tiny synthetic videos (no browser needed)."""
import cv2
import numpy as np

from a11y_replay.ingest.video import crop, iter_frames, probe
from a11y_replay.segment import detect_changes

FPS = 10
W, H = 320, 240


def _write(path, frames):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for f in frames:
        out.write(f)
    out.release()


def _base():
    return np.full((H, W, 3), 200, np.uint8)


def test_probe_and_iter(tmp_path):
    p = tmp_path / "v.avi"
    _write(p, [_base() for _ in range(20)])
    info = probe(p)
    assert (info.width, info.height) == (W, H) and round(info.fps) == FPS
    assert sum(1 for _ in iter_frames(p)) == 20


def test_static_video_has_no_events(tmp_path):
    p = tmp_path / "static.avi"
    _write(p, [_base() for _ in range(30)])
    assert detect_changes(p) == []


def test_sensor_noise_is_ignored(tmp_path):
    rng = np.random.default_rng(0)
    frames = [np.clip(_base() + rng.normal(0, 3, (H, W, 3)), 0, 255).astype(np.uint8) for _ in range(30)]
    p = tmp_path / "noise.avi"
    _write(p, frames)
    assert detect_changes(p) == []


def test_square_appearing_and_disappearing_are_two_events(tmp_path):
    frames = []
    for i in range(30):
        f = _base()
        if 10 <= i < 15:  # visible from t=1.0s; gone again at t=1.5s
            cv2.rectangle(f, (100, 80), (180, 140), (0, 0, 0), -1)
        frames.append(f)
    p = tmp_path / "square.avi"
    _write(p, frames)
    events = detect_changes(p)
    assert len(events) == 2  # appearance and disappearance are separate episodes
    appear, vanish = events
    assert abs(appear.t - 1.0) <= 0.11 and abs(vanish.t - 1.5) <= 0.11
    x, y, w, h = appear.bbox
    assert 90 <= x <= 105 and 70 <= y <= 85 and 60 <= w <= 90 and 50 <= h <= 75


def test_continuous_flicker_merges_into_one_episode(tmp_path):
    frames = []
    for i in range(30):
        f = _base()
        if 10 <= i < 20 and i % 2 == 0:  # toggles every frame for 1 s
            cv2.rectangle(f, (100, 80), (180, 140), (0, 0, 0), -1)
        frames.append(f)
    p = tmp_path / "flicker.avi"
    _write(p, frames)
    events = detect_changes(p)
    assert len(events) == 1 and events[0].n_frames >= 8


def test_crop_clamps_to_frame():
    f = np.zeros((50, 100, 3), np.uint8)
    assert crop(f, (90, 40, 30, 30), pad=5).shape[:2] == (15, 15)