"""End-to-end audit tests on a small synthetic recording (no browser, no network)."""
import json
import unittest

import cv2
import numpy as np

from a11y_replay.audit import AuditReport, ReportItem, audit_uploads, run_audit
from a11y_replay.report import render_html
from a11y_replay.vlm.client import MockClient

W, H, FPS = 640, 360, 25
KEYS = [(1.5, "F9"), (2.5, "F9"), (3.5, "F9"), (4.5, "F9"), (6.0, "Tab")]   # F9 flashes a marker: gives clock sync


def _need_tesseract():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise unittest.SkipTest(f"Tesseract not available: {e}")


def _frame(t, flash=True, toast=True, highlight_at=None, ring_at=None):
    f = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(f, (100, 100), (300, 140), (120, 120, 120), 1)
    if any(k <= t < k + 0.2 and key == "F9" for k, key in KEYS):
        cv2.rectangle(f, (W - 70, H - 70), (W - 20, H - 20), (150, 30, 220), -1)
    if toast and 8.0 <= t < 11.0:
        cv2.putText(f, "Changes saved", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2)
    if flash and 12.0 <= t < 13.6 and (int(t * FPS) // 3) % 2 == 1:
        f[:] = 0
    if ring_at is not None and t >= ring_at:
        cv2.rectangle(f, (96, 96), (304, 144), (0, 140, 255), 3)
    if highlight_at is not None and t >= highlight_at:
        cv2.rectangle(f, (104, 106), (180, 134), (215, 120, 30), -1)
    return f


def _recording(path, with_keys=True, transcript=None, flash=True, toast=True, highlight_at=None, ring_at=None, seconds=15.0):
    path.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(seconds * FPS)):
        out.write(_frame(i / FPS, flash, toast, highlight_at, ring_at))
    out.release()
    if with_keys:
        (path / "keys.jsonl").write_text("\n".join(json.dumps({"t": t, "key": k, "modifiers": []}) for t, k in KEYS))
    if transcript is not None:
        (path / "utterances.json").write_text(json.dumps(transcript))
    return path


def test_audit_finds_all_three_kinds_of_problem_and_writes_a_report(tmp_path):
    _need_tesseract()
    rec = _recording(tmp_path / "rec", transcript=[])          # empty transcript: nothing was announced
    r = run_audit(rec, tmp_path / "out")
    assert {i.criterion for i in r.findings} == {"2.4.7", "4.1.3", "2.3.1"}
    assert r.sync["trusted"] and abs(r.sync["offset_s"]) < 0.15
    assert not [i for i in r.items if i.status == "needs_review" and abs(i.start - 12.3) < 1.0], \
        "a flashing episode must not also be listed as an unlabeled change"
    assert (tmp_path / "out" / "report.html").exists() and (tmp_path / "out" / "report.json").exists()
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert len(data["items"]) == len(r.items) and data["sync"]["pairs"] >= 3
    for it in r.findings:                                       # every finding carries a picture
        assert it.evidence and (tmp_path / "out" / it.evidence[0]["file"]).exists()
    names = [c["name"] for c in r.checks if c["status"] == "not_implemented"]
    assert any("2.2.2" in n for n in names) and any("2.1.2" in n for n in names)


def test_announced_message_is_not_a_finding(tmp_path):
    _need_tesseract()
    rec = _recording(tmp_path / "rec", transcript=[{"start": 8.6, "end": 9.4, "text": "Changes saved"}])
    r = run_audit(rec, tmp_path / "out")
    assert "4.1.3" not in {i.criterion for i in r.findings}


def test_audit_ignores_benchmark_answer_keys(tmp_path):
    """The audit must use only video, key log and transcript. A labels.json that lies must change nothing."""
    _need_tesseract()
    rec = _recording(tmp_path / "rec", transcript=[])
    base = [(i.criterion, round(i.start, 1)) for i in run_audit(rec, tmp_path / "o1").items]
    (rec / "labels.json").write_text(json.dumps({
        "labels": [], "qa": {"ok": True, "offset_s": 4.0}, "status_checks": [], "duration_s": 99}))
    again = [(i.criterion, round(i.start, 1)) for i in run_audit(rec, tmp_path / "o2").items]
    assert base == again


def test_missing_inputs_skip_checks_and_say_why(tmp_path):
    rec = _recording(tmp_path / "rec", with_keys=False, transcript=None, toast=False)
    r = run_audit(rec, tmp_path / "out")
    by = {c["name"]: c for c in r.checks}
    focus = next(c for n, c in by.items() if "2.4.7" in n)
    ann = next(c for n, c in by.items() if "4.1.3" in n)
    assert focus["status"] == "skipped" and "keys.jsonl" in focus["detail"]
    assert ann["status"] == "skipped" and "transcript" in ann["detail"]
    assert r.sync["trusted"] is False and "No key log" in r.sync["note"]
    assert any(i.criterion == "2.3.1" for i in r.findings)       # the flash check needs neither


def test_unclear_focus_presses_need_review_without_a_model_and_are_decided_with_one(tmp_path):
    rec = _recording(tmp_path / "rec", flash=False, toast=False, highlight_at=5.9)   # a solid patch after Tab
    r = run_audit(rec, tmp_path / "plain")
    assert [i.status for i in r.items if i.criterion == "2.4.7"] == ["needs_review"]
    no = MockClient(lambda c: '{"indicator_visible": "no", "indicator_type": "", "evidence": "nothing visible"}',
                    model="claude-sonnet-5-5")
    r2 = run_audit(rec, tmp_path / "model", client=no)
    f = [i for i in r2.items if i.criterion == "2.4.7"]
    assert [i.status for i in f] == ["finding"] and f[0].method == "vlm" and f[0].confidence is None
    assert r2.stats["vlm_calls"] == 1
    yes = MockClient(lambda c: '{"indicator_visible": "yes", "indicator_type": "fill", "evidence": "dark fill"}')
    assert [i for i in run_audit(rec, tmp_path / "yes", client=yes).items if i.criterion == "2.4.7"] == []


def test_report_html_escapes_content_and_is_self_describing(tmp_path):
    item = ReportItem("x-1", "finding", "4.1.3", '<script>alert("x")</script>', 3.0, 6.0, "classical",
                      "<b>bold</b> & more", None, [])
    rep = AuditReport("rec<1>", "2026-01-01 00:00", "0.1.0", {"offset_s": 0.0, "trusted": True, "pairs": 4, "spread_s": 0.0,
                      "note": "ok"}, [{"name": "n", "status": "ran", "detail": "d"}], [item], {"video_seconds": 1})
    page = render_html(rep, tmp_path)
    assert "<script>alert" not in page and "&lt;script&gt;" in page and "&lt;b&gt;bold" in page
    assert 'lang="en"' in page and "screening aid" in page and "<caption>" in page and 'scope="col"' in page
    assert "not a compliance test" in page


def test_audit_uploads_gathers_files_and_returns_the_report_folder(tmp_path):
    rec = _recording(tmp_path / "src", flash=True, toast=False)
    out = audit_uploads(rec / "video.avi", rec / "keys.jsonl", None, None, work_dir=tmp_path / "work")
    assert (out / "report.html").exists()
    assert "2.3.1" in (out / "report.json").read_text()


def test_audit_finds_the_clock_offset_when_the_logger_was_started_seconds_late(tmp_path):
    """Real sessions: the person starts the screen recorder and the key logger by hand, so the key log's
    t=0 is seconds away from the video's. The audit must search for that offset, not give up."""
    rec = _recording(tmp_path / "rec", transcript=None, toast=False, flash=False)
    lines = [json.loads(x) for x in (rec / "keys.jsonl").read_text().splitlines()]
    shifted = [{**d, "t": round(d["t"] - 7.3, 3)} for d in lines]               # the logger started 7.3 s after the video
    (rec / "keys.jsonl").write_text("\n".join(json.dumps(d) for d in shifted))
    r = run_audit(rec, tmp_path / "out")
    assert r.sync["trusted"] and r.sync["searched"] and abs(r.sync["offset_s"] - 7.3) < 0.4, r.sync
    assert "started about" in r.sync["note"]


def test_a_transcript_on_the_video_clock_is_shifted_onto_the_log_clock(tmp_path):
    _need_tesseract()
    rec = _recording(tmp_path / "rec", transcript=[{"start": 8.6, "end": 9.4, "text": "Changes saved"}])
    r = run_audit(rec, tmp_path / "log", transcript_clock="log")
    off = r.sync["offset_s"]
    # same speech, but with times written on a clock that runs 2.5 s ahead of the video
    (rec / "utterances.json").write_text(json.dumps([{"start": 8.6 + 2.5, "end": 9.4 + 2.5, "text": "Changes saved"}]))
    assert "4.1.3" in {i.criterion for i in run_audit(rec, tmp_path / "wrong", transcript_clock="log").findings}
    # claiming it is on the video clock does not fix a 2.5 s skew either: only a matching clock does
    (rec / "utterances.json").write_text(json.dumps([{"start": 8.6 + off, "end": 9.4 + off, "text": "Changes saved"}]))
    ok = run_audit(rec, tmp_path / "video", transcript_clock="video")
    assert "4.1.3" not in {i.criterion for i in ok.findings}


# ---------------------------------------------------------------- scoring against human labels
def _labeled(tmp_path, issues, complete=True, independent=True):
    from eval.eval_real import evaluate_recording
    rec = _recording(tmp_path / "rec", transcript=[])
    (rec / "human_labels.json").write_text(json.dumps(
        {"labeler": "T. Ester", "independent": independent, "complete": complete, "issues": issues}))
    return evaluate_recording(rec, tmp_path / "out")


GOOD = [{"t": 6.0, "criterion": "2.4.7", "note": "no focus ring"},
        {"t": 8.0, "criterion": "4.1.3", "note": "toast not read"},
        {"t": 12.1, "criterion": "2.3.1", "note": "flashing"}]


def test_scoring_matches_findings_to_human_labels_on_the_video_clock(tmp_path):
    _need_tesseract()
    r = _labeled(tmp_path, GOOD)
    for c in ("2.4.7", "4.1.3", "2.3.1"):
        d = r["per_criterion"][c]
        assert (d["tp"], d["fn"], d["fp"]) == (1, 0, 0), (c, d)
    assert r["independent"] and r["complete"]


def test_a_label_the_audit_cannot_find_is_a_miss_and_an_unlabeled_finding_a_false_alarm(tmp_path):
    _need_tesseract()
    r = _labeled(tmp_path, [GOOD[0], GOOD[1], {"t": 40.0, "criterion": "2.4.7", "note": "something else"}])
    d = r["per_criterion"]["2.4.7"]
    assert (d["tp"], d["fn"]) == (1, 1) and d["missed_labels"][0]["t"] == 40.0
    assert r["per_criterion"]["2.3.1"]["fp"] == 1        # the flash was real but the labeler did not list it
    assert r["per_criterion"]["2.3.1"]["extra_findings"]


def test_incomplete_labels_make_extra_findings_unconfirmed_not_false_alarms(tmp_path):
    _need_tesseract()
    r = _labeled(tmp_path, GOOD[:2], complete=False)
    d = r["per_criterion"]["2.3.1"]
    assert d["fp"] is None and d["unconfirmed"] == 1


def test_a_finding_far_from_the_label_in_time_does_not_match(tmp_path):
    _need_tesseract()
    r = _labeled(tmp_path, [{"t": 6.0 + 5.0, "criterion": "2.4.7", "note": "wrong time"}] + GOOD[1:])
    d = r["per_criterion"]["2.4.7"]
    assert d["tp"] == 0 and d["fn"] == 1 and d["fp"] == 1


def test_a_partly_spoken_message_is_listed_for_review_with_its_own_title(tmp_path):
    _need_tesseract()
    rec = _recording(tmp_path / "rec", transcript=[{"start": 8.5, "end": 8.9, "text": "Saved"}], flash=False)
    r = run_audit(rec, tmp_path / "out")
    review = [i for i in r.needs_review if i.criterion == "4.1.3"]
    assert review and "partly announced" in review[0].title
    assert not [i for i in r.findings if i.criterion == "4.1.3"]


def test_the_summary_says_up_front_when_checks_were_skipped(tmp_path):
    """Regression from reading a real report: '0 findings' was at the top and 'two of three checks never ran' was
    in a table at the bottom, so a skim looked like a clean bill of health."""
    rec = _recording(tmp_path / "rec", with_keys=False, transcript=None, toast=False, flash=False)
    run_audit(rec, tmp_path / "out")
    page = (tmp_path / "out" / "report.html").read_text()
    top = page[:page.index('id="findings"')]
    assert "Only 1 of 3 available checks ran" in top and "does not mean the recording is clean" in top
    assert "needs keys.jsonl" in top and "transcript" in top            # and WHY, before any findings


def test_the_summary_is_plain_when_every_check_ran(tmp_path):
    _need_tesseract()
    rec = _recording(tmp_path / "rec", transcript=[])
    run_audit(rec, tmp_path / "out")
    top = (tmp_path / "out" / "report.html").read_text().split('id="findings"')[0]
    assert "All 3 available checks ran" in top and "Only" not in top


def test_passes_are_hidden_by_default_and_listed_with_evidence_on_request(tmp_path):
    """A report that only shows findings hides what the tool judged fine; --show-passes makes passes checkable."""
    rec = _recording(tmp_path / "rec", flash=False, toast=False, ring_at=5.9)       # a clear ring appears after Tab
    plain = run_audit(rec, tmp_path / "plain")
    assert plain.items == [] and "judged fine" not in (tmp_path / "plain" / "report.html").read_text().lower()
    shown = run_audit(rec, tmp_path / "shown", include_passes=True)
    assert [i.status for i in shown.items] == ["pass"] and shown.findings == [] and shown.needs_review == []
    p = shown.passes[0]
    assert p.criterion == "2.4.7" and p.evidence and (tmp_path / "shown" / p.evidence[0]["file"]).exists()
    page = (tmp_path / "shown" / "report.html").read_text()
    assert "Judged fine (1)" in page and "as critically as the findings" in page


def _rewrite_keys(rec, lines):
    (rec / "keys.jsonl").write_text("\n".join(json.dumps({"t": t, "key": k, "modifiers": []}) for t, k in lines))


def test_presses_that_fall_outside_the_video_are_reported_not_silently_dropped(tmp_path):
    """Regression from a real run: 'Focus visible: 1 Tab press(es)' with no hint that others had been skipped."""
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    _rewrite_keys(rec, KEYS + [(500.0, "Tab"), (501.0, "Tab")])
    r = run_audit(rec, tmp_path / "out")
    detail = next(c["detail"] for c in r.checks if "2.4.7" in c["name"])
    assert "3 Tab press(es) in the key log, but only 1 could be analyzed" in detail and "fell outside the video" in detail


def test_untrusted_clock_alignment_downgrades_focus_findings_to_review(tmp_path):
    """If the key log cannot be lined up with the video, the frames compared may be the wrong ones, so a 'focus not visible'
    result is a guess. It must not be reported as a finding."""
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    _rewrite_keys(rec, [(1.5, "F9"), (2.5, "F9"), (6.0, "Tab")])             # only 2 usable pairs: sync not trusted
    r = run_audit(rec, tmp_path / "out")
    assert r.sync["trusted"] is False
    assert [i for i in r.findings if i.criterion == "2.4.7"] == []
    review = [i for i in r.needs_review if i.criterion == "2.4.7"]
    assert review and "clock alignment is not trusted" in review[0].title and "may not be the ones" in review[0].reasoning
    assert "downgraded" in next(c["detail"] for c in r.checks if "2.4.7" in c["name"])


def test_when_sync_is_not_trusted_the_console_lists_each_pair(tmp_path):
    import contextlib
    import io

    from a11y_replay.audit import main
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    _rewrite_keys(rec, [(1.5, "F9"), (2.5, "F9"), (6.0, "Tab")])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([str(rec), "--out", str(tmp_path / "out")])
    out = buf.getvalue()
    assert "NOT TRUSTED" in out and "pairs (key @ log time" in out and "F9@1.5s" in out and "F9@2.5s" in out
    assert json.loads((tmp_path / "out" / "report.json").read_text())["sync"]["detail"][0]["key"] == "F9"


def test_a_big_unrelated_change_near_a_flash_time_is_not_counted_as_a_calibration_flash(tmp_path):
    """Regression from a real recording: two F9 timestamps were 'matched' to window-sized changes (28% and 22% of the frame),
    so the report claimed 2 calibration flashes although no marker was in the video."""
    import contextlib
    import io

    from a11y_replay.audit import main
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    out = cv2.VideoWriter(str(rec / "video2.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(15 * FPS)):                                  # no marker at all; instead two near-full-frame changes
        t = i / FPS
        f = np.full((H, W, 3), 255, np.uint8)
        if 1.6 <= t < 1.9 or 5.3 <= t < 5.6:
            f[40:340, 30:610] = (60, 60, 60)
        out.write(f)
    out.release()
    (rec / "video.avi").unlink()
    (rec / "video2.avi").rename(rec / "video.avi")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([str(rec), "--out", str(tmp_path / "out")])
    r = json.loads((tmp_path / "out" / "report.json").read_text())["sync"]
    assert r["calibration_pairs"] == 0 and any(d["change_frac"] > 0.06 for d in r["detail"])
    assert "was NOT counted as a flash" in buf.getvalue()


def test_trusted_alignment_still_gives_a_real_finding(tmp_path):
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    r = run_audit(rec, tmp_path / "out")
    assert r.sync["trusted"] and [i.criterion for i in r.findings] == ["2.4.7"]


def test_the_console_shows_the_clock_sync_line(tmp_path, capsys=None):
    import contextlib
    import io

    from a11y_replay.audit import main
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([str(rec), "--out", str(tmp_path / "out")])
    out = buf.getvalue()
    assert "Clock sync: offset" in out and "calibration flashes" in out and "spread" in out and "trusted" in out


def test_calibration_flashes_matched_means_no_circularity_warning(tmp_path):
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    r = run_audit(rec, tmp_path / "out")
    assert r.sync["trusted"] and r.sync["calibration_pairs"] >= 3
    assert "partly circular" not in r.sync["note"]


def test_alignment_without_calibration_flashes_is_flagged_as_partly_circular(tmp_path):
    """Regression from a real recording: the video started 11 s after the logger, so the flashes were not recorded and the clocks
    were aligned only from the Tab presses that were then being tested."""
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    lines = [json.loads(x) for x in (rec / "keys.jsonl").read_text().splitlines()]
    keep = [d for d in lines if d["key"] != "F9"]                              # the logger saw no flashes in the video
    extra = [{"t": 7.0 + i, "key": "Enter", "modifiers": []} for i in range(4)]
    (rec / "keys.jsonl").write_text("\n".join(json.dumps(d) for d in keep + extra))
    # make four visible changes at those Enter times so alignment is possible from them alone
    out = cv2.VideoWriter(str(rec / "video2.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(int(15 * FPS)):
        t = i / FPS
        f = _frame(t, flash=False, toast=False)
        for e in extra:
            if e["t"] <= t < e["t"] + 0.5:
                cv2.rectangle(f, (400, 200), (520, 300), (0, 0, 0), -1)
        out.write(f)
    out.release()
    (rec / "video.avi").unlink()
    (rec / "video2.avi").rename(rec / "video.avi")
    r = run_audit(rec, tmp_path / "out")
    assert r.sync["trusted"] and r.sync["calibration_pairs"] == 0
    assert "partly circular" in r.sync["note"] and "None of the calibration flashes" in r.sync["note"]


def _age(path, seconds):
    import os
    import time
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_the_newest_video_is_analyzed_and_the_others_are_named_in_a_warning(tmp_path):
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    import shutil
    shutil.copy(rec / "video.avi", rec / "video.mov")                        # an older recording left in the folder
    _age(rec / "video.mov", 3600)
    r = run_audit(rec, tmp_path / "out")
    assert r.stats["video_file"] == "video.avi"
    assert r.warnings and "Several video files" in r.warnings[0] and "video.mov" in r.warnings[0]


def test_evidence_pictures_from_an_earlier_run_are_removed(tmp_path):
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    (tmp_path / "out" / "evidence").mkdir(parents=True)
    (tmp_path / "out" / "evidence" / "old_run.png").write_bytes(b"stale")
    run_audit(rec, tmp_path / "out")
    assert not (tmp_path / "out" / "evidence" / "old_run.png").exists()


def test_the_console_says_which_video_file_and_when_it_was_saved(tmp_path):
    import contextlib
    import io

    from a11y_replay.audit import main
    rec = _recording(tmp_path / "rec", flash=False, toast=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([str(rec), "--out", str(tmp_path / "out")])
    assert "Video: video.avi, 15.0 s, last modified 20" in buf.getvalue()