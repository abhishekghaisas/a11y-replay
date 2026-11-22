"""Demo logic tests (no Gradio), plus a wiring check of app.py against a stand-in for Gradio."""
import importlib
import inspect
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

from a11y_replay.demo import HEADERS, check_upload, find_examples, run_demo
from tests.test_audit import _recording


class _NoKeyNoDotenv:
    """No API key in the environment, and the demo's .env loading switched off: a developer's own .env holds a real key and the
    demo reloads it on every run, which silently put it back and made these tests depend on whose machine they ran on."""

    def __enter__(self):
        import a11y_replay.demo as demo_module
        self.module, self.real = demo_module, demo_module._load_env
        self.old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        demo_module._load_env = lambda: None
        return self

    def __exit__(self, *exc):
        self.module._load_env = self.real
        if self.old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self.old_key
        return False


def test_find_examples_lists_only_folders_that_contain_a_video(tmp_path):
    (tmp_path / "recordings" / "a").mkdir(parents=True)
    (tmp_path / "recordings" / "a" / "video.mp4").write_bytes(b"x")
    (tmp_path / "recordings" / "empty").mkdir()
    (tmp_path / "recordings" / "notes.txt").write_text("hi")
    found = find_examples([str(tmp_path / "recordings")])
    assert list(found) == [f"{tmp_path / 'recordings'}/a"]


def test_running_an_example_gives_a_summary_rows_evidence_and_a_report(tmp_path):
    rec = _recording(tmp_path / "rec", toast=False)          # flash + a Tab with no visible effect; no transcript
    r = run_demo("ex", examples={"ex": str(rec)})
    assert r.ok and "finding(s)" in r.summary and "available checks ran" in r.summary
    assert {row[1] for row in r.rows} >= {"WCAG 2.3.1", "WCAG 2.4.7"} and all(len(row) == len(HEADERS) for row in r.rows)
    assert r.evidence and all(Path(p).exists() for p, _ in r.evidence)
    assert Path(r.report_html).exists()
    assert "not a compliance test" in r.summary and "Skipped" in r.summary          # honesty is part of the UI


def test_the_demo_never_writes_into_the_example_recording_folder(tmp_path):
    rec = _recording(tmp_path / "rec", toast=False)
    run_demo("ex", examples={"ex": str(rec)})
    assert not (rec / "audit_report").exists()


def test_uploaded_files_work_like_a_folder(tmp_path):
    rec = _recording(tmp_path / "rec", toast=False)
    r = run_demo(None, video=str(rec / "video.avi"), keys=str(rec / "keys.jsonl"), examples={})
    assert r.ok and any(row[1] == "WCAG 2.3.1" for row in r.rows)


def test_nothing_chosen_gives_a_plain_instruction_not_a_crash():
    r = run_demo(None, examples={})
    assert not r.ok and "Choose an example" in r.summary


def test_a_model_without_an_api_key_is_refused_up_front(tmp_path):
    rec = _recording(tmp_path / "rec", toast=False)
    with _NoKeyNoDotenv():
        r = run_demo("ex", model="claude-sonnet-5-5", examples={"ex": str(rec)})
    assert not r.ok and "ANTHROPIC_API_KEY" in r.summary and "none" in r.summary


def test_a_video_that_will_not_open_explains_how_to_convert_it(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "video.mp4").write_bytes(b"not a video at all")
    r = run_demo("ex", examples={"ex": str(bad)})
    assert not r.ok and "ffmpeg" in r.summary


def test_progress_is_reported_from_start_to_finish(tmp_path):
    rec = _recording(tmp_path / "rec", toast=False)
    seen = []
    run_demo("ex", examples={"ex": str(rec)}, progress=lambda f, m: seen.append((f, m)))
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs) and fracs[0] < 0.1 and fracs[-1] == 1.0 and len(set(m for _, m in seen)) >= 5


def test_show_passes_is_passed_through(tmp_path):
    rec = _recording(tmp_path / "rec", flash=False, toast=False, ring_at=5.9)
    assert run_demo("ex", examples={"ex": str(rec)}).rows == []
    shown = run_demo("ex", examples={"ex": str(rec)}, show_passes=True)
    assert [row[2] for row in shown.rows] == ["Judged fine"]


# ---------------------------------------------------------------- app.py wiring against a stand-in for Gradio
def _fake_gradio():
    mod = types.ModuleType("gradio")
    registry, progress_calls = [], []

    class Comp:
        def __init__(self, *a, **k):
            self.kwargs = k

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

        def click(self, fn, inputs=None, outputs=None, **k):
            registry.append((fn, inputs, outputs))

    for name in ("Blocks", "Row", "Column", "Markdown", "File", "Dropdown", "Radio", "Checkbox", "Button",
                 "Dataframe", "Gallery"):
        setattr(mod, name, type(name, (Comp,), {}))

    class Progress:
        def __call__(self, frac, desc=None):
            progress_calls.append((frac, desc))

    mod.Progress = Progress
    return mod, registry, progress_calls


def test_app_wiring_connects_every_input_to_the_handler_and_returns_every_output(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    fake, registry, progress_calls = _fake_gradio()
    saved = {k: sys.modules.get(k) for k in ("gradio", "app")}
    sys.modules["gradio"] = fake
    sys.modules.pop("app", None)
    try:
        app = importlib.import_module("app")
        app.find_examples = lambda *a, **k: {"ex": str(rec)}
        rec = _recording(tmp_path / "rec", toast=False)
        app.build_demo()
        (fn, inputs, outputs), = registry
        params = [p for p in inspect.signature(fn).parameters.values() if p.name != "progress"]
        assert len(inputs) == len(params) == 7 and len(outputs) == 4
        summary, rows, gallery, report = fn("ex", None, None, None, "none", "log", False)
        assert "finding(s)" in summary and rows and gallery and Path(report).exists()
        assert progress_calls and progress_calls[-1][0] == 1.0
        # a failure comes back in the same shape, with a message
        s2, r2, g2, rep2 = fn("", None, None, None, "none", "log", False)
        assert "Could not run" in s2 and r2 == [] and g2 == [] and rep2 is None
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------- hosted use: no leftovers, bounded uploads
def _demo_dirs():
    import glob
    return set(glob.glob(tempfile.gettempdir() + "/a11y_demo_*"))


def test_a_hosted_run_returns_everything_in_memory_and_leaves_nothing_on_disk(tmp_path):
    rec = _recording(tmp_path / "rec", toast=False)
    before = _demo_dirs()
    r = run_demo(None, video=str(rec / "video.avi"), keys=str(rec / "keys.jsonl"), keep_files=False, examples={})
    assert r.ok and r.report_bytes.startswith(b"<!doctype html") and r.evidence == [] and r.report_dir is None
    assert r.evidence_bytes and all(png[:8] == b"\x89PNG\r\n\x1a\n" for png, _ in r.evidence_bytes)
    assert _demo_dirs() == before                                   # the uploaded video and the report are gone


def test_failed_runs_leave_nothing_behind_either(tmp_path):
    before = _demo_dirs()
    assert not run_demo(None, examples={}).ok                       # nothing chosen
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert not run_demo(None, video=str(bad), examples={}).ok       # unreadable upload
    rec = _recording(tmp_path / "rec", toast=False)
    with _NoKeyNoDotenv():
        refused = run_demo("ex", model="claude-sonnet-5-5", examples={"ex": str(rec)})   # a model with no key is refused up front
    assert not refused.ok and "ANTHROPIC_API_KEY" in refused.summary
    assert _demo_dirs() == before


def test_local_runs_still_keep_their_files_for_the_ui_to_show(tmp_path):
    rec = _recording(tmp_path / "rec", toast=False)
    r = run_demo("ex", examples={"ex": str(rec)})
    try:
        assert r.ok and Path(r.report_html).exists() and r.evidence and r.report_bytes is None
    finally:
        shutil.rmtree(Path(r.report_dir).parent, ignore_errors=True)


def test_check_upload_rejects_oversize_overlong_and_unreadable_files(tmp_path):
    rec = _recording(tmp_path / "rec", toast=False)                 # a 15 s clip
    video = rec / "video.avi"
    assert check_upload(video, max_mb=100, max_seconds=100) is None
    assert "accepts up to" in check_upload(video, max_mb=0.001, max_seconds=100)
    assert "15 s long" in check_upload(video, max_mb=100, max_seconds=5)
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"x" * 5000)
    assert "could not be read" in check_upload(junk, max_mb=100, max_seconds=100)