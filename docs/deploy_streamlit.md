# Putting the demo on Streamlit

`streamlit_app.py` is the entry point. It runs the same audit as the command line, from a web page: pick one of five bundled
example recordings, or upload a short recording (with an optional key log and transcript), and read the findings, evidence frames
and the full report.

**What I checked and what I did not.** The app's logic is tested without a browser, and the page script was driven against a
stand-in for Streamlit. **Nobody has run it under the real Streamlit library or deployed it yet**, and I have not seen the hosting
platform's current limits or screens. Treat the steps below as a plan to verify, not a record of something that worked.

## What is different from running it on your own machine

| | Local | Hosted |
|---|---|---|
| Language model for unclear focus cases | optional (`A11Y_DEMO_ENABLE_MODEL=1` and `ANTHROPIC_API_KEY` in `.env`) | **off, and the selector is not shown** |
| Uploaded recording | copied to a temporary folder | same, and the folder is **deleted before the results appear** |
| Upload limits | none | 60 MB and 180 s (checked by the app) |
| Examples | your own `recordings/` | the five synthetic ones in `demo_examples/` |

Never set `A11Y_DEMO_ENABLE_MODEL=1` on a public site, and never put an API key in the repository or the hosting secrets of a public
app: anyone could spend it.

## 1. Check it locally first

```
pip install streamlit
python -m a11y_replay.doctor demo_examples      # every required line should say OK
python -m a11y_replay.verify_examples            # does each example BEHAVE as its folder name says? (works for any files)
python -m a11y_replay.check_examples            # only if you use the exact files shipped here: are they where they belong? (--fix repairs)
python -m pytest tests -q                        # or run the tests the way you have been
streamlit run streamlit_app.py                   # opens http://localhost:8501
```

In the page: open **Environment check** in the sidebar (all OK), run each example, and compare with the table in
`demo_examples/README.md`. Then upload one short clip of your own.

## 2. Check what you are about to publish

```
python -m a11y_replay.prepush      # lists everything git would commit; any BLOCK line means: do not push
git status
```

You should see `streamlit_app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`, `demo_examples/` (11 files) and the
`a11y_replay/` package. You should **not** see `recordings/`, `.env`, `benchmark/data/` or any report with your own screen in it.
If `demo_examples/` is missing from the list, the `.gitignore` exceptions did not apply.

## 3. Deploy (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. Go to share.streamlit.io, sign in with GitHub, and create a new app.
3. Choose the repository and branch, and set the main file to `streamlit_app.py`.
4. In the advanced settings choose Python 3.12 (the code is written for 3.9 and newer). Add no secrets.
5. Deploy. The first build installs `requirements.txt` and the system packages in `packages.txt`.

The hosting platform's own documentation is the authority on its steps, memory and CPU limits, and what happens to idle apps. If the
platform shuts apps down after a period of inactivity, the first visit afterwards will be slow.

## 4. After it is live

- Sidebar, **Environment check**: OpenCV, NumPy, ffmpeg, ffprobe and Tesseract should all say OK.
- Run all five examples and compare with `demo_examples/README.md`.
- Upload a clip under 60 s (and optionally its `keys.jsonl`). Confirm the results appear.
- Confirm no language-model selector is visible.

## 5. If it fails

| Symptom | Likely cause |
|---|---|
| "ffmpeg program: MISSING" in the environment check | `packages.txt` is not at the repository root |
| Import error mentioning `libGL` | `opencv-python` was installed instead of `opencv-python-headless` (check `requirements.txt`) |
| "Tesseract OCR: missing (optional)" | `tesseract-ocr` is not in `packages.txt`; only the status-message check is affected |
| "No example recordings were found" | `demo_examples/` was not committed (see step 2) |
| An upload is refused | it is over 60 MB or 180 s; the message says which |
| Results look wrong for your own recording | read `docs/recording_guide.md`: most surprises came from the recording, not the tool |