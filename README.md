# A11y Replay

**Audit a screen recording for accessibility problems that only show up over time.** Give it a video of someone
using an interface (plus their key presses, and a transcript or audio of what a screen reader said) and it
reports findings with timestamps, the WCAG 2.1 criterion, evidence frames and the reasoning.

> **A screening aid, not a compliance test.** It checks three criteria from pixels and speech. A clean report
> does not mean a page is accessible, and every finding should be confirmed by a person.
>
> **Status: working prototype, validated only on synthetic recordings made by its author.** Nothing has been
> tested on real-world recordings or labeled by anyone else. Read "What the numbers do and don't show" below
> before quoting any result.

> **Read `docs/RESULTS.md` first.** It lists what was tested at what level of evidence, the eight things real recordings broke
> that no test had, and the open issues, including one unexplained result on a real recording.

## What it checks

| Check | WCAG | How | Status |
|---|---|---|---|
| Keyboard focus visible after Tab | 2.4.7 | frame differencing per key press, then a ring/appeared-or-disappeared test; unclear cases go to a language model | works on the benchmark |
| Flashing content | 2.3.1 | luminance swings that must also flash across a whole region; general-flash approximation | works on the benchmark; first real recording exposed false positives (scrolling), fix not yet re-checked on it |
| Status message seen but not heard | 4.1.3 | OCR of new on-screen text, matched against a transcript | works with a simulated transcript and with real speech recognition of one synthetic voice |
| Pause / stop / hide moving content | 2.2.2 | not built | **not implemented** |
| No keyboard trap | 2.1.2 | not built | **not implemented** |

Not covered at all: anything in the page's code (labels, roles, the accessibility tree), and every criterion not listed.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,benchmark,vlm,record]" && playwright install chromium
brew install tesseract ffmpeg            # OCR and video conversion (macOS)

# make a few synthetic recordings, then audit one
python -m benchmark.generate --n 3
python -m a11y_replay.audit benchmark/data/status_messages_mixed_000
open benchmark/data/status_messages_mixed_000/audit_report/report.html

python app.py                            # optional demo UI (Gradio wiring is untested)
pytest                                   # 238 tests
```

An audited folder contains `video.mp4|webm`, and optionally `keys.jsonl` (enables clock sync and the focus
check), `utterances.json` (a transcript, enables the status-message check) or `audio.wav` (recognized with
`--transcribe`). Add `--model claude-sonnet-5-5` to let a language model judge the focus cases the classical layer
cannot (needs `ANTHROPIC_API_KEY`; costs money; off by default). The audit reads **only** those inputs, never the
benchmark's answer keys; a test enforces that.

## How it works

1. **Clock sync.** The key log is the master clock. Each key press is paired with the visual change it caused to
   estimate the video-to-log offset per recording; if it cannot be trusted the report says so.
2. **Classical first.** Deterministic detectors decide what they can. Each is documented with its limits in
   `docs/methodology.md`.
3. **Model only for the residue.** Cases a ring test cannot judge (an underline, a fill, a faint ring, a text
   highlight) are sent to a language model with a cropped before/after pair and one closed question. Every call is
   counted. Model answers carry no calibrated confidence and are labeled as model decisions.
4. **Report.** A self-contained, accessible HTML file plus JSON.

## Demo UI

**Streamlit (for a hosted demo):** `pip install streamlit && streamlit run streamlit_app.py`. It offers five bundled synthetic
examples (`demo_examples/`) or an upload, shows findings, evidence frames and the full report, turns the language model off, limits
uploads, and deletes every temporary file before showing results. Deployment steps and what is and is not verified:
`docs/deploy_streamlit.md`. **It has not yet been run under the real Streamlit library.**

**Gradio (local):** `pip install gradio && python app.py` opens a small web UI: pick an example recording or upload one (optionally with its key log
and transcript), run the audit, and see the summary, a findings table, evidence frames and the downloadable report. All logic is in
`a11y_replay/demo.py` and is tested; `app.py` only builds widgets, and its wiring was checked against a stand-in for Gradio, **not
against the real library**. `python app.py --selftest` checks it on your machine without opening a browser. A walkthrough is in
`docs/demo_script.md`.

## Recording real sessions

The tooling for real-world evidence exists; the evidence itself does not yet. `python -m a11y_replay.recorder` logs
navigation keys (never typed text, by default) and flashes a calibration square so the key log and a screen recording
can be lined up even when started seconds apart; `python -m eval.eval_real recordings/*` scores the audit against labels
written by someone else. **Not yet tried by anyone**: the key logger's system hooks (pynput, tkinter) are untested, and no
real recordings or independent labels have been collected. Start with `docs/recording_guide.md`.

## What the numbers do and don't show

Results below are from this author's own machine and benchmark. Treat them as development checks.

- **Focus, 53 Tab presses, 9 test samples:** all 19 real failures found, no confident mistakes, about a quarter of
  presses escalated. The classifier was changed repeatedly after failures on these same samples (every change is listed in
  `docs/methodology.md`), so this is not an unbiased accuracy figure. One of those failures was found only because a
  second machine rendered text selection differently.
- **Language model on the escalated presses:** Sonnet 5.5, Haiku 5.5 and Opus 5.5 all agreed with the expected verdict on
  every clear case, so the benchmark could not tell them apart. The model was more generous than my label on a faint
  ring. Results were not repeated to measure run-to-run noise.
- **Status messages, 9 messages, 3 silent:** correct with a simulated transcript and with real Whisper `base` output of
  one synthetic voice (6 spoken lines), after fixing Whisper merging two lines 11 s apart. One short message matched at
  84.6% against an 80% threshold, so the margin is thin. Recall cannot fail in this benchmark by construction.
- **Wrong speech near a silent message** (7 recorded samples, 11 silent messages, an adversarial set I built around the
  matcher's known weak spots): without content guards only 3 were flagged; with them 6 are flagged and 5 more go to review,
  none passes. The guards were written after seeing those examples, so this is a design check, not robustness evidence.
- **Flashing:** the benchmark cases are easy (full-width black and white). The first real recording (a Streamlit app, whole
  screen) produced 14 false flashing findings from scrolling text and window switching; the detector now requires the
  brightness of a whole region to flash (a checkerboard pattern is therefore no longer detected). Fixed on synthetic
  scrolling text; **not yet re-run on that recording.**
- **Real focus check:** on a clean page, 6 of 6 Tab presses judged visible after fixing a frame-rate bug (the clocks were aligned from the Tab presses themselves, so this is partly circular). An earlier run on the buggy page used the old time mapping and is void until re-run.
- **First real recording, clock sync:** worked on a hand-started recording (offset 1.2 s, 15 key/change pairs, spread 0.08 s).

## Related work and gaps

Based on a limited web search in November 2026, not a systematic review; verify before claiming novelty.

| Existing | What it does |
|---|---|
| [AXNav](https://machinelearning.apple.com/research/axnav) (Apple) | LLM-driven accessibility test runs on a device, producing a video with flagged issues |
| [ScreenAudit](https://arxiv.org/html/2504.02110v1) | LLM audit of mobile screens from UI metadata and screen-reader transcripts |
| axe-core, Lighthouse, WAVE | static scans of a page's DOM or accessibility tree |

| Gap I tried to fill | Where it stands |
|---|---|
| Audit a recording someone already made, with no instrumented device | built; synthetic recordings only |
| Time-dependent issues static scanners cannot see | 3 of 5 planned checks built |
| Seen-versus-heard on one timeline (OCR vs transcript) | built; real speech recognition tested on one synthetic voice |
| A labeled benchmark with exact timestamps | built (4 synthetic apps, quality-gated); recording and labeling tooling built; **no real-world set and no independent labelers yet** |
| Use a model only where classical methods cannot decide, and measure it | built and instrumented; the model comparison is small and saturated |

## Repository layout

```
a11y_replay/    audit.py (end-to-end), demo.py (UI logic), report.py, recorder.py (key logger), detectors/ (flash, focus, announce), ingest/, vlm/, cascade.py, sync.py
benchmark/      test apps, generate.py (records video + keys + labels, quality-gated), tts.py (macOS speech)
eval/           eval_focus, eval_flash, eval_announce, eval_vlm (bake-off), eval_real (human labels), shared helpers
docs/           recording_guide.md, labeling_guide.md (benchmark ground truth), methodology.md (methods, limits, every post-hoc change)
tests/          238 tests; OCR tests skip if Tesseract is missing
```

## Next

Collect real recordings and independent labels (`docs/recording_guide.md`); a benchmark case where a silent message sits next to
unrelated speech (so recall can fail and the 80% threshold can be set); the 2.2.2 motion detector; repeated model calls to
measure noise; matched model settings across models.

## License

TBD.