"""Streamlit demo for A11y Replay.

    pip install streamlit
    streamlit run streamlit_app.py

Deploys to Streamlit Community Cloud as is (see docs/deploy_streamlit.md). All behavior lives in a11y_replay/demo.py
(tested without a UI); this file only builds the page.

Hosted-demo safeguards:
  * recordings are processed from a temporary file and every temporary file is deleted before the results are shown;
  * the language-model option is OFF unless the environment variable A11Y_DEMO_ENABLE_MODEL=1 is set (never set it on a public
    site: anyone could spend your API key);
  * uploads are limited in size and length.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from a11y_replay.demo import DISCLAIMER, HEADERS, check_upload, find_examples, run_demo

ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "demo_examples"
MAX_MB = 60.0
MAX_SECONDS = 180.0
ENABLE_MODEL = os.environ.get("A11Y_DEMO_ENABLE_MODEL") == "1"

PRIVACY = ("Do not upload recordings that show private or confidential information. Your files are written to a temporary "
           "folder, analyzed, and deleted before the results appear; this app does not store them. The hosting platform's own "
           "logging is outside this app's control.")


def _save_upload(up, suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(up.getbuffer())
    f.close()
    return f.name


def _show_results(st, components, result):
    if not result.ok:
        st.error(result.summary)
        return
    st.markdown(result.summary)
    tab_table, tab_evidence, tab_report = st.tabs(["Findings", "Evidence frames", "Full report"])
    with tab_table:
        if result.rows:
            st.dataframe([dict(zip(HEADERS, row)) for row in result.rows])
        else:
            st.info("No findings and nothing needing review. Read the line above about which checks ran.")
    with tab_evidence:
        if not result.evidence_bytes:
            st.write("No evidence frames for this recording.")
        for i in range(0, len(result.evidence_bytes), 2):
            cols = st.columns(2)
            for col, (png, caption) in zip(cols, result.evidence_bytes[i:i + 2]):
                with col:
                    st.image(png, caption=caption)
    with tab_report:
        components.html(result.report_bytes.decode("utf-8"), height=900, scrolling=True)
    st.download_button("Download report.html", data=result.report_bytes, file_name="a11y-replay-report.html", mime="text/html")


def main():
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="A11y Replay", layout="wide")
    st.title("A11y Replay: audit a screen recording")
    st.warning(DISCLAIMER)
    st.caption(PRIVACY)

    examples = {Path(p).name: p for p in find_examples([str(EXAMPLES_DIR)]).values()}   # show folder names, not paths
    with st.sidebar:
        st.header("Recording")
        mode = st.radio("Source", ["Use an example", "Upload my own"])
        example = video = keys = transcript = None
        if mode == "Use an example":
            if examples:
                example = st.selectbox("Example", list(examples))
                st.caption("Synthetic test pages recorded for this project; the folder's README says what each should show.")
            else:
                st.error("No example recordings were found in demo_examples/.")
        else:
            video = st.file_uploader(f"Screen recording (up to {MAX_MB:.0f} MB and {MAX_SECONDS:.0f} s)",
                                     type=["mp4", "mov", "webm", "avi", "mkv"])
            keys = st.file_uploader("Key log, keys.jsonl (optional, enables the focus check)", type=["jsonl", "txt"])
            transcript = st.file_uploader("Transcript, utterances.json (optional, enables the status-message check)",
                                          type=["json"])
        clock = st.radio("Transcript times are on the clock of the", ["key log", "video"], horizontal=True)
        passes = st.checkbox("Also list what was judged fine, to check it", value=False)
        if ENABLE_MODEL:
            model = st.selectbox("Language model for unclear focus cases", ["none", "claude-sonnet-5-5"])
        else:
            model = "none"
            st.caption("Language-model judging is turned off in this hosted demo.")
        go = st.button("Run audit", type="primary")
        with st.expander("Environment check"):
            from a11y_replay.doctor import check_environment
            for name, ok, detail, required in check_environment(EXAMPLES_DIR):
                st.write(f"{'OK' if ok else ('MISSING' if required else 'missing, optional')}: {name}. {detail}")

    if go:
        temp_files = []
        try:
            if mode == "Upload my own":
                if video is None:
                    st.error("Choose a screen recording to upload, or switch to an example.")
                    st.stop()
                v_path = _save_upload(video, Path(video.name).suffix or ".mp4")
                temp_files.append(v_path)
                problem = check_upload(v_path, MAX_MB, MAX_SECONDS)
                if problem:
                    st.error(problem)
                    st.stop()
                k_path = _save_upload(keys, ".jsonl") if keys is not None else None
                t_path = _save_upload(transcript, ".json") if transcript is not None else None
                temp_files += [p for p in (k_path, t_path) if p]
            else:
                v_path = k_path = t_path = None
            bar, note = st.progress(0), st.empty()

            def on_progress(frac, message):
                bar.progress(int(max(0.0, min(1.0, frac)) * 100))
                note.write(message)

            st.session_state["result"] = run_demo(
                example, v_path, k_path, t_path, model, "video" if clock == "video" else "log", bool(passes),
                progress=on_progress, examples=examples, keep_files=False)
            bar.empty()
            note.empty()
        finally:
            for p in temp_files:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    if st.session_state.get("result") is not None:
        _show_results(st, components, st.session_state["result"])
    else:
        st.info("Pick an example in the sidebar and press **Run audit**.")


if __name__ == "__main__":      # Streamlit runs the script as __main__
    main()