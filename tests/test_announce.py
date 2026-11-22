"""Seen-versus-heard tests on tiny synthetic videos. Needs the Tesseract program for OCR; the
OCR-dependent tests are skipped (not passed) if it is missing."""
import unittest

import cv2
import numpy as np

from dataclasses import replace

from a11y_replay.config import DEFAULT
from a11y_replay.detectors.announce import (Observation, content_conflict, detect_announcements, match_observations,
                                            new_words, normalize, numbers_in, similarity, to_findings, word_coverage)
from a11y_replay.events import KeyEvent, Utterance

W, H, FPS = 640, 360, 25


def _need_tesseract():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:  # missing module or missing program
        raise unittest.SkipTest(f"Tesseract not available: {e}")


def _video(path, items, seconds=9.0):
    """items: (text or None, (x, y), start_s, end_s); text=None draws a colored square instead."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(seconds * FPS)):
        t = i / FPS
        f = np.full((H, W, 3), 255, np.uint8)
        for text, (x, y), a, b in items:
            if a <= t < b:
                if text is None:
                    cv2.rectangle(f, (x, y), (x + 60, y + 40), (0, 120, 255), -1)
                else:
                    cv2.putText(f, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
        out.write(f)
    out.release()


MSG = ("Changes saved", (150, 200))


def _items(start=3.0, end=6.0, extra=()):
    return [(MSG[0], MSG[1], start, end), *extra]


# ---------------------------------------------------------------- pure text helpers (no OCR)
def test_similarity_basics():
    assert similarity("Changes saved", "Changes saved") == 100.0
    assert similarity("Changes saved", "Alert, changes saved") >= 95      # spoken role prefix
    assert similarity("Changes saved", "chaneges savd") >= 80             # recognition typos
    assert similarity("Changes saved", "Upload failed") < 60
    assert similarity("", "x") == 0.0


def test_new_words_are_returned_as_written_on_screen():
    """The report must say what the screen says: digits stay digits, case is kept."""
    assert new_words("", "3 Results FOUND") == "3 Results FOUND"
    assert new_words("3 results", "3 results found") == "found"
    assert new_words("three results", "3 results found") == "found"   # equal after normalizing


def test_new_words_ignores_text_that_merely_moved():
    assert new_words("quarterly report", "upload failed file too large quarterly report") == "upload failed file too large"
    assert new_words("Quarterly Report", "Upload failed quarterly report") == "Upload failed"
    assert new_words("a b", "a b") == ""
    assert normalize("  Alert,  Upload FAILED: file!! ") == "alert upload failed file"


def test_numbers_compare_equal_whether_written_as_digits_or_words():
    assert normalize("3 results found") == "three results found"
    assert normalize("Step 21 of 100") == "step twenty one of one hundred"
    assert similarity("3 of 12", "Three of twelve.") == 100.0
    assert similarity("3 results found", "3 results found") == 100.0
    assert normalize("year 2024") == "year 2024"       # beyond 999 left alone


# ---------------------------------------------------------------- content guards (no OCR needed)
def test_numbers_are_parsed_from_digits_and_words():
    assert numbers_in("3 results found") == [3] == numbers_in("Three results found.")
    assert numbers_in("Step 21 of 100") == [21, 100] == numbers_in("step twenty-one of a hundred")
    assert numbers_in("one hundred twenty three items") == [123]
    assert numbers_in("three five") == [3, 5]
    assert numbers_in("13 results") == [13] and numbers_in("no numbers here") == []


def test_content_conflicts_catch_what_character_similarity_cannot():
    assert similarity("3 results found", "13 results found") >= 80          # why the guard exists
    assert "number 3" in content_conflict("3 results found", "13 results found")
    assert "number 3" in content_conflict("3 results found", "No results found")
    assert content_conflict("3 results found", "Three results found.") is None
    assert content_conflict("Upload failed file too large", "Alert, upload failed, file too large.") is None
    assert "negation" in content_conflict("Your changes were saved successfully", "Your changes were not saved successfully")
    assert content_conflict("Changes saved", "Change is saved.") is None    # a plain misrecognition is still accepted


def _ob(text, t=10.0):
    return Observation(t=t, bbox=(0, 0, 10, 10), kind="message", text=text)


def test_a_different_number_or_a_flipped_negation_does_not_announce_the_message():
    for wrong in ("13 results found", "No results found"):
        r = match_observations([_ob("3 results found")], [Utterance(10.6, 11.6, wrong)])[0]
        assert r.verdict == "unannounced" and "rejected" in r.reasoning, (wrong, r)
    off = replace(DEFAULT, announce_content_guards=False)
    assert match_observations([_ob("3 results found")], [Utterance(10.6, 11.6, "13 results found")], cfg=off)[0].verdict == "announced"


def test_a_right_announcement_is_still_found_among_wrong_ones():
    utts = [Utterance(10.3, 11.0, "Link, Terms"), Utterance(10.9, 11.9, "No results found"),
            Utterance(12.0, 13.0, "Three results found.")]
    r = match_observations([_ob("3 results found")], utts)[0]
    assert r.verdict == "announced" and r.matched_text == "Three results found."


def test_a_garbled_number_is_rejected_like_a_wrong_one_a_known_cost():
    r = match_observations([_ob("3 results found")], [Utterance(10.5, 11.5, "tree results found")])[0]
    assert r.verdict == "unannounced"


# ---------------------------------------------------------------- partial announcements
def test_word_coverage_separates_a_misrecognized_word_from_a_different_word():
    assert word_coverage("Upload failed file too large", "Alert, upload failed") == 0.4          # a fragment
    assert word_coverage("Upload failed file too large", "Alert. Upload, failed. File to large.") == 1.0     # "to" ~ "too"
    assert word_coverage("Changes saved", "Change is saved.") == 1.0                              # inflection garbled
    assert word_coverage("Changes saved", "Changes saving") == 0.5                                # key word differs
    assert word_coverage("3 results found", "Three results found.") == 1.0


def test_every_real_whisper_output_seen_so_far_still_counts_as_announced():
    """Regression: these are the actual strings faster-whisper `base` produced on the benchmark's synthetic voice."""
    real = [("Changes saved", "Change is saved."),
            ("Upload failed file too large", "Alert, Upload, failed. File to large."),
            ("3 results found", "3 results found"), ("3 results found", "Three results found.")]
    for screen, heard in real:
        r = match_observations([_ob(screen)], [Utterance(10.3, 11.3, heard)])[0]
        assert r.verdict == "announced", (screen, heard, r.reasoning)


def test_a_fragment_or_a_different_key_word_goes_to_review_not_to_announced():
    for screen, heard in (("Upload failed file too large", "Upload failed"), ("Changes saved", "Changes saving")):
        r = match_observations([_ob(screen)], [Utterance(10.3, 11.3, heard)])[0]
        assert r.verdict == "needs_review" and "Only part of the message" in r.reasoning, (heard, r)
        assert to_findings([r]) == []                       # a person decides; it is not a finding
    off = replace(DEFAULT, announce_content_guards=False)
    assert match_observations([_ob("Changes saved")], [Utterance(10.3, 11.3, "Changes saving")], cfg=off)[0].verdict == "announced"


# ---------------------------------------------------------------- end to end
def test_spoken_message_is_announced(tmp_path):
    _need_tesseract()
    p = tmp_path / "a.avi"; _video(p, _items())
    res = detect_announcements(p, [], [Utterance(3.6, 4.4, "Changes saved")])
    assert [r.verdict for r in res] == ["announced"] and "changes saved" in res[0].text.lower()
    assert to_findings(res) == []


def test_silent_message_is_a_4_1_3_finding(tmp_path):
    _need_tesseract()
    p = tmp_path / "s.avi"; _video(p, _items())
    res = detect_announcements(p, [], [])
    assert [r.verdict for r in res] == ["unannounced"]
    f = to_findings(res)
    assert len(f) == 1 and f[0].criterion == "4.1.3" and abs(f[0].start - 3.0) < 0.2
    assert "Nothing was spoken" in res[0].reasoning


def test_message_disappearing_is_not_a_second_event(tmp_path):
    _need_tesseract()
    p = tmp_path / "d.avi"; _video(p, _items(3.0, 5.0))
    assert len(detect_announcements(p, [], [])) == 1   # the appearance only, not the dismissal


def test_announcement_too_late_does_not_count(tmp_path):
    _need_tesseract()
    p = tmp_path / "l.avi"; _video(p, _items())
    res = detect_announcements(p, [], [Utterance(7.6, 8.2, "Changes saved")])  # 4.6 s late
    assert res[0].verdict == "unannounced"


def test_announcement_of_different_text_does_not_count(tmp_path):
    _need_tesseract()
    p = tmp_path / "w.avi"; _video(p, _items())
    res = detect_announcements(p, [], [Utterance(3.5, 4.0, "Item deleted")])
    assert res[0].verdict == "unannounced" and res[0].nearest_text == "Item deleted"


def test_one_utterance_cannot_announce_two_messages(tmp_path):
    _need_tesseract()
    p = tmp_path / "t.avi"; _video(p, [(MSG[0], MSG[1], 1.5, 3.5), (MSG[0], MSG[1], 6.0, 8.0)], seconds=10)
    res = detect_announcements(p, [], [Utterance(2.0, 2.6, "Changes saved")])
    assert [r.verdict for r in res] == ["announced", "unannounced"]


def test_non_text_change_is_needs_review_never_a_finding(tmp_path):
    _need_tesseract()
    p = tmp_path / "n.avi"; _video(p, [(None, (300, 100), 3.0, 6.0)])
    res = detect_announcements(p, [], [])
    assert [r.verdict for r in res] == ["needs_review"] and to_findings(res) == []


def test_focus_key_and_calibration_key_changes_are_excluded(tmp_path):
    _need_tesseract()
    p = tmp_path / "k.avi"; _video(p, [(None, (300, 100), 3.0, 6.0)])
    assert detect_announcements(p, [KeyEvent(t=3.0, key="Tab")], []) == []
    assert detect_announcements(p, [KeyEvent(t=3.0, key="F9")], []) == []
    assert len(detect_announcements(p, [KeyEvent(t=3.0, key="Enter")], [])) == 1  # user-caused still counts


def test_page_load_changes_at_the_start_are_ignored(tmp_path):
    _need_tesseract()
    p = tmp_path / "pl.avi"; _video(p, _items(0.3, 3.0))
    assert detect_announcements(p, [], []) == []


def test_video_offset_is_mapped_to_the_log_clock(tmp_path):
    _need_tesseract()
    p = tmp_path / "o.avi"; _video(p, _items(3.0, 6.0))   # appears at video 3.0 = log 3.1 (video leads by 0.1)
    res = detect_announcements(p, [], [Utterance(3.3, 3.9, "Changes saved")], offset=-0.1)
    assert res[0].verdict == "announced" and abs(res[0].t - 3.1) < 0.1