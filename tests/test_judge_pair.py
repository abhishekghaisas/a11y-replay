import cv2
import numpy as np

from a11y_replay.judge_pair import judge_files
from a11y_replay.vlm.client import MockClient

YES = '{"indicator_visible": "yes", "indicator_type": "outline", "evidence": "orange ring on Cancel"}'


def _save(path, v):
    cv2.imwrite(str(path), np.full((80, 200, 3), v, np.uint8))


def test_judge_files_asks_the_focus_question_about_two_saved_images(tmp_path):
    _save(tmp_path / "b.png", 220)
    _save(tmp_path / "a.png", 120)
    client = MockClient([YES])
    r = judge_files(tmp_path / "b.png", tmp_path / "a.png", client)
    assert r.answer == "yes" and r.parse_ok and client.meter.calls == 1
    kinds = [c["type"] for c in client.requests[0]]
    assert kinds.count("image") == 2                         # exactly the before and after pictures


def test_a_missing_image_gives_a_clear_error(tmp_path):
    _save(tmp_path / "b.png", 200)
    try:
        judge_files(tmp_path / "b.png", tmp_path / "nope.png", MockClient([YES]))
        raise AssertionError("should have failed")
    except FileNotFoundError as e:
        assert "nope.png" in str(e)


def test_the_shipped_example_pair_is_two_readable_images_of_the_same_size():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "docs" / "examples"
    b, a = cv2.imread(str(root / "touching_rings_before.png")), cv2.imread(str(root / "touching_rings_after.png"))
    assert b is not None and a is not None and b.shape == a.shape and b.shape[1] > 400
    diff = np.abs(b.astype(int) - a.astype(int)).sum(axis=2) > 60
    assert 0.02 < diff.mean() < 0.4                          # the ring moved: a real, localized change between the two