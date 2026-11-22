import shutil
from pathlib import Path

from a11y_replay.verify_examples import EXPECT, main, verify

SHIPPED = Path(__file__).resolve().parents[1] / "demo_examples"


def test_the_shipped_examples_all_behave_as_their_names_say():
    results = verify(SHIPPED)
    assert {r[0] for r in results} == set(EXPECT)
    assert all(r[1] is True for r in results), [(r[0], r[3]) for r in results if r[1] is not True]


def test_two_swapped_folders_are_caught_with_a_hint_about_which_they_look_like(tmp_path):
    copy = tmp_path / "demo_examples"
    shutil.copytree(SHIPPED, copy)
    a, b = copy / "focus_page_with_bug", copy / "focus_page_clean"
    tmp = copy / "_t"
    a.rename(tmp); b.rename(a); tmp.rename(b)                                   # swap the two folders' contents
    res = {r[0]: r for r in verify(copy)}
    assert res["focus_page_with_bug"][1] is False and "focus_page_clean" in res["focus_page_with_bug"][3]
    assert res["focus_page_clean"][1] is False and "focus_page_with_bug" in res["focus_page_clean"][3]
    assert res["flashing_banner_with_bug"][1] is True                           # the others are unaffected
    assert main([str(copy)]) == 1


def test_a_missing_folder_and_an_unknown_folder_are_reported_not_crashed_on(tmp_path):
    copy = tmp_path / "demo_examples"
    shutil.copytree(SHIPPED, copy)
    shutil.rmtree(copy / "flashing_banner_clean")
    shutil.copytree(copy / "focus_page_clean", copy / "my_own_extra")
    res = {r[0]: r for r in verify(copy)}
    assert res["flashing_banner_clean"][1] is False and "not found" in res["flashing_banner_clean"][3]
    assert res["my_own_extra"][1] is None