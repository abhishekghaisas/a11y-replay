"""Ask the language model the focus question about two saved images, with no recording needed.

    python -m a11y_replay.judge_pair BEFORE.png AFTER.png [--model claude-sonnet-5-5] [--effort medium]

Use it to try the model on a real before/after pair you saved (for example a crop copied out of a report), or on the example
pair in docs/examples/ (a focus ring moving between two touching buttons, from a real recording: the correct answer is
"yes", an indicator is visible on the newly focused button). Needs ANTHROPIC_API_KEY in .env; costs a fraction of a cent.

It asks exactly what the audit asks for an unclear press, so what you see is what the escalation would have decided. The
images are used whole: for a real escalation the audit sends a padded crop around the changed region.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from a11y_replay.config import DEFAULT, Config
from a11y_replay.vlm.client import AnthropicClient, VLMClient
from a11y_replay.vlm.pricing import estimate_cost
from a11y_replay.vlm.tasks import JudgeResult, judge_focus


def judge_files(before: str | Path, after: str | Path, client: VLMClient, key: str = "Tab",
                cfg: Config = DEFAULT) -> JudgeResult:
    imgs = []
    for p in (before, after):
        im = cv2.imread(str(p))
        if im is None:
            raise FileNotFoundError(f"Could not read the image: {p}")
        imgs.append(im)
    return judge_focus(client, imgs[0], imgs[1], None, key, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--model", default=DEFAULT.vlm_model)
    ap.add_argument("--effort", default=None)
    a = ap.parse_args(argv)
    client = AnthropicClient(model=a.model, effort=a.effort)
    try:
        r = judge_files(a.before, a.after, client)
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(str(e))
    cost = estimate_cost(a.model, r.input_tokens, r.output_tokens)
    print(f"model:      {r.model}")
    print(f"answer:     {r.answer}   (yes = an indicator is visible on the newly focused element)")
    print(f"indicator:  {r.indicator_type or '-'}")
    print(f"evidence:   {r.evidence or '-'}")
    print(f"valid JSON: {r.parse_ok}, attempts: {r.attempts}")
    print(f"tokens:     {r.input_tokens} in, {r.output_tokens} out" + (f", about ${cost:.4f}" if cost is not None else ""))


if __name__ == "__main__":
    main()