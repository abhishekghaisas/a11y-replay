"""Drive streamlit_app.py against a stand-in for Streamlit (the real library is not available here). This catches my own
mistakes (wrong arguments, results lost on rerun, temp files left behind, the model option leaking into a hosted demo);
it cannot catch a difference from the real library."""
import glob
import importlib
import shutil
import sys
import tempfile
import types
from pathlib import Path

from tests.test_audit import _recording

ROOT = Path(__file__).resolve().parents[1]


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def progress(self, v):
        return None

    def write(self, *a, **k):
        return None

    def empty(self):
        return None


class Stop(BaseException):
    """What st.stop() raises in the real library: it ends the script run."""


def _fake(**cfg):
    st = types.ModuleType("streamlit")
    st.__path__ = []
    st.calls, st.cfg, st.session_state, st.sidebar = [], cfg, {}, _Ctx()
    rec = lambda name: (lambda *a, **k: st.calls.append((name, a, k)))
    for name in ("set_page_config", "title", "header", "warning", "caption", "markdown", "error", "info", "write",
                 "dataframe", "image", "download_button"):
        setattr(st, name, rec(name))
    st.radio = lambda label, options, **k: cfg.get("mode", options[0]) if label == "Source" else options[0]
    st.selectbox = lambda label, options, **k: (st.calls.append(("selectbox", (label,), {})), cfg.get("example", options[0]))[1]
    st.file_uploader = lambda label, **k: cfg.get("uploads", {}).get(label.split(" (")[0].split(",")[0])
    st.checkbox = lambda label, value=False, **k: cfg.get("passes", False)
    st.button = lambda label, **k: cfg.get("go", True)
    st.progress = lambda v=0, **k: _Ctx()
    st.empty = lambda: _Ctx()
    st.columns = lambda n: [_Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda labels: [_Ctx() for _ in labels]
    st.expander = lambda label, **k: _Ctx()

    def stop():
        raise Stop()
    st.stop = stop
    comp, v1 = types.ModuleType("streamlit.components"), types.ModuleType("streamlit.components.v1")
    comp.__path__ = []
    v1.html = lambda body, **k: st.calls.append(("components.html", (body,), k))
    st.components = comp
    return st, comp, v1


class _Upload:
    def __init__(self, path):
        self.name, self._data = Path(path).name, Path(path).read_bytes()

    def getbuffer(self):
        return self._data


def _run(tmp_path, monkey=None, **cfg):
    sys.path.insert(0, str(ROOT))
    st, comp, v1 = _fake(**cfg)
    saved = {k: sys.modules.get(k) for k in ("streamlit", "streamlit.components", "streamlit.components.v1", "streamlit_app")}
    sys.modules.update({"streamlit": st, "streamlit.components": comp, "streamlit.components.v1": v1})
    sys.modules.pop("streamlit_app", None)
    try:
        app = importlib.import_module("streamlit_app")
        ex = tmp_path / "demo_examples"
        ex.mkdir(exist_ok=True)
        if not (ex / "clip").exists():
            shutil.copytree(_recording(tmp_path / "rec", toast=False), ex / "clip")
        app.EXAMPLES_DIR = ex
        if monkey:
            monkey(app)
        try:
            app.main()
        except Stop:
            pass
        return st, app
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)


def _dirs():
    return set(glob.glob(tempfile.gettempdir() + "/a11y_demo_*"))


def _calls(st, name):
    return [c for c in st.calls if c[0] == name]


def test_running_an_example_shows_findings_evidence_and_the_report_and_keeps_nothing(tmp_path):
    before = _dirs()
    st, app = _run(tmp_path)
    r = st.session_state["result"]
    assert r.ok and r.report_bytes.startswith(b"<!doctype html")
    assert _calls(st, "dataframe") and set(_calls(st, "dataframe")[0][1][0][0]) == set(app.HEADERS)
    assert _calls(st, "image") and _calls(st, "components.html") and _calls(st, "download_button")
    assert _calls(st, "download_button")[0][2]["mime"] == "text/html"
    assert _dirs() == before                                           # no temp folder survives the run


def test_the_language_model_is_not_offered_or_used_on_a_hosted_demo(tmp_path):
    seen = {}

    def spy(app):
        real = app.run_demo
        app.ENABLE_MODEL = False
        app.run_demo = lambda *a, **k: (seen.setdefault("model", a[4]), real(*a, **k))[1]
    st, app = _run(tmp_path, monkey=spy)
    assert seen["model"] == "none"
    assert not [c for c in _calls(st, "selectbox") if "Language model" in c[1][0]]


def test_the_model_selector_appears_only_when_explicitly_enabled(tmp_path):
    def enable(app):
        app.ENABLE_MODEL = True
    st, app = _run(tmp_path, monkey=enable, go=False)
    assert [c for c in _calls(st, "selectbox") if "Language model" in c[1][0]]


def test_an_uploaded_recording_is_analyzed_and_every_temporary_file_is_removed(tmp_path):
    rec = _recording(tmp_path / "up", toast=False)
    saved = []

    def spy(app):
        real = app._save_upload
        app._save_upload = lambda up, suffix: (saved.append(real(up, suffix)), saved[-1])[1]
    before = _dirs()
    st, app = _run(tmp_path, monkey=spy, mode="Upload my own",
                   uploads={"Screen recording": _Upload(rec / "video.avi"), "Key log": _Upload(rec / "keys.jsonl")})
    result = st.session_state["result"]
    assert result.ok and {r[1] for r in result.rows} >= {"WCAG 2.3.1", "WCAG 2.4.7"}      # the flash AND the Tab press, from the upload
    assert len(saved) == 2 and not any(Path(p).exists() for p in saved)
    assert _dirs() == before


def test_an_oversize_upload_is_refused_before_any_analysis_and_cleaned_up(tmp_path):
    rec = _recording(tmp_path / "up", toast=False)
    saved = []

    def tiny_limit(app):
        app.MAX_MB = 0.0001
        real = app._save_upload
        app._save_upload = lambda up, suffix: (saved.append(real(up, suffix)), saved[-1])[1]
    st, app = _run(tmp_path, monkey=tiny_limit, mode="Upload my own", uploads={"Screen recording": _Upload(rec / "video.avi")})
    assert [c for c in _calls(st, "error") if "accepts up to" in c[1][0]]
    assert st.session_state.get("result") is None and saved and not any(Path(p).exists() for p in saved)


def test_pressing_run_without_choosing_a_file_gives_a_message_not_a_crash(tmp_path):
    st, app = _run(tmp_path, mode="Upload my own", uploads={})
    assert [c for c in _calls(st, "error") if "Choose a screen recording" in c[1][0]]


def test_a_failed_run_shows_the_message_and_leaves_nothing_behind(tmp_path):
    bad = tmp_path / "demo_examples" / "broken"
    bad.mkdir(parents=True)
    (bad / "video.mp4").write_bytes(b"not a video")
    before = _dirs()
    st, app = _run(tmp_path, example="broken")
    assert not st.session_state["result"].ok and [c for c in _calls(st, "error") if "Could not run" in c[1][0]]
    assert _dirs() == before


def test_results_survive_a_rerun_because_streamlit_reruns_the_script_on_every_click(tmp_path):
    """Pressing Download (or anything) reruns the script with the button reading False; the results must still be shown."""
    sys.path.insert(0, str(ROOT))
    st, comp, v1 = _fake()
    saved = {k: sys.modules.get(k) for k in ("streamlit", "streamlit.components", "streamlit.components.v1", "streamlit_app")}
    sys.modules.update({"streamlit": st, "streamlit.components": comp, "streamlit.components.v1": v1})
    sys.modules.pop("streamlit_app", None)
    try:
        app = importlib.import_module("streamlit_app")
        ex = tmp_path / "demo_examples"
        shutil.copytree(_recording(tmp_path / "rec", toast=False), ex / "clip")
        app.EXAMPLES_DIR = ex
        app.main()                                   # first run: button pressed
        st.calls.clear()
        st.cfg["go"] = False
        app.main()                                   # rerun: button not pressed, results still in session_state
        assert _calls(st, "download_button") and _calls(st, "components.html")
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)