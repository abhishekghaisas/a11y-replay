"""Demo UI for A11y Replay (Gradio).

    pip install gradio
    python app.py                 # opens http://127.0.0.1:7860
    python app.py --selftest      # checks the app builds and the audit runs, without opening a browser

All behavior lives in a11y_replay/demo.py (tested without Gradio). This file only builds the widgets. It was checked
against a stand-in for Gradio, NOT against the real library, so if your Gradio version rejects something, --selftest
says what, and the command-line audit does the same work.
"""
from __future__ import annotations

import argparse
import inspect

from a11y_replay.demo import DISCLAIMER, HEADERS, find_examples, run_demo

INTRO = f"""# A11y Replay: audit a screen recording

{DISCLAIMER}

**To use:** pick an example recording, or upload a screen recording. Adding the key log (`keys.jsonl`, from
`python -m a11y_replay.recorder`) turns on the keyboard-focus check; a transcript (`utterances.json`) turns on the
status-message check. Without them those checks are skipped, and the results say so.
"""


def _kw(cls, **kw):
    """Pass only the keyword arguments this Gradio version's component accepts."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def build_demo():
    import gradio as gr

    examples = find_examples()

    def handle(example, video, keys, transcript, model, clock, passes, progress=gr.Progress()):
        result = run_demo(example or None, video, keys, transcript, model, clock, bool(passes),
                          progress=lambda frac, msg: progress(frac, desc=msg), examples=examples)
        return result.summary, result.rows, result.evidence, result.report_html

    with gr.Blocks(title="A11y Replay") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                example = gr.Dropdown(**_kw(gr.Dropdown, choices=[""] + list(examples), value="",
                                            label="Example recording (leave blank to upload your own)"))
                video = gr.File(**_kw(gr.File, label="Screen recording (.mp4, .mov or .webm)", type="filepath"))
                keys = gr.File(**_kw(gr.File, label="Key log, keys.jsonl (optional)", type="filepath"))
                transcript = gr.File(**_kw(gr.File, label="Transcript, utterances.json (optional)", type="filepath"))
                model = gr.Dropdown(**_kw(gr.Dropdown, choices=["none", "claude-sonnet-5-5"], value="none",
                                          label="Language model for unclear focus cases (needs ANTHROPIC_API_KEY)"))
                clock = gr.Radio(**_kw(gr.Radio, choices=["log", "video"], value="log",
                                       label="Transcript times are on the clock of the: key log / video"))
                passes = gr.Checkbox(**_kw(gr.Checkbox, label="Also list what was judged fine, to check it", value=False))
                run = gr.Button("Run audit", variant="primary")
            with gr.Column():
                summary = gr.Markdown("Results appear here.")
                table = gr.Dataframe(**_kw(gr.Dataframe, headers=HEADERS, interactive=False, wrap=True, label="Findings"))
                gallery = gr.Gallery(**_kw(gr.Gallery, label="Evidence frames", columns=2))
                download = gr.File(**_kw(gr.File, label="Download report.html"))
        run.click(handle, inputs=[example, video, keys, transcript, model, clock, passes],
                  outputs=[summary, table, gallery, download])
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--selftest", action="store_true", help="build the UI and audit one example, without opening a browser")
    a = ap.parse_args(argv)
    if a.selftest:
        import gradio as gr
        print(f"gradio {getattr(gr, '__version__', '?')}: building the interface ...")
        build_demo()
        print("  interface built.")
        ex = find_examples()
        if not ex:
            print("  no example recordings found under recordings/ or benchmark/data/; skipping the audit check.")
            return
        label = next(iter(ex))
        print(f"  auditing {label} ...")
        r = run_demo(label, examples=ex, progress=lambda f, m: print(f"    {f:4.0%}  {m}"))
        print("  OK" if r.ok else f"  FAILED: {r.summary}")
        print(r.summary.splitlines()[0] if r.ok else "")
        return
    build_demo().launch(server_port=a.port)


if __name__ == "__main__":
    main()