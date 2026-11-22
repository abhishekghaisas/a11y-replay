"""Evaluate the focus detector (WCAG 2.4.7) on generated samples, scoring all three verdicts.

Usage (from the repo root):
    python -m eval.eval_focus benchmark/data/*
    python -m eval.eval_focus benchmark/data/* --detail   # also list every press whose verdict differs from the expected one

Expected verdict per Tab press:
  * focus_styles samples: from "focus_checks" in labels.json (visible / not_visible / ambiguous)
  * focus_outline samples: every press in a bug sample is not_visible, in a clean sample visible
The detector answers visible / not_visible / ambiguous (ambiguous = would go to the VLM).

What is reported:
  * a per-style table (predicted verdict counts for each style)
  * a confusion matrix over all presses
  * confident errors: the detector said visible/not_visible and was wrong, or was confident on a
    case whose ground truth is ambiguous. These are the dangerous ones.
  * escalation rate: share of presses the classical layer could not decide
  * failure detection (positive = not_visible): precision and recall, where an escalated failure
    counts as a miss until the VLM step exists

Caveat: the styles and their expected verdicts were written by the same person who wrote the
detector. Treat results as a development check, not as accuracy evidence.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from a11y_replay.config import DEFAULT
from a11y_replay.detectors.focus import detect_focus
from a11y_replay.ingest.keys import load_keys
from eval.common import sample_offset
from eval.metrics import prf

VERDICTS = ("visible", "not_visible", "ambiguous")


def check_for(meta: dict, key_t: float):
    """Ground truth for the Tab press at key_t: (style, expected, tab_index, previous_style) or None."""
    checks = meta.get("focus_checks")
    if checks:
        c = min(checks, key=lambda c: abs(c["t"] - key_t))
        if abs(c["t"] - key_t) >= 0.05:
            return None
        prev = next((x["style"] for x in checks if x["tab_index"] == c["tab_index"] - 1), None)
        return c["style"], c["expected"], c["tab_index"], prev
    if meta.get("app") == "focus_outline":
        style, exp = ("legacy_bug", "not_visible") if meta.get("variant") == "bug" else ("legacy_clean", "visible")
        return style, exp, None, None
    return None


def main(argv: list[str]) -> None:
    detail = "--detail" in argv
    paths = [a for a in argv if a != "--detail"]
    details = []
    samples = [Path(p) for p in paths if (Path(p) / "labels.json").exists() and (Path(p) / "keys.jsonl").exists()]
    rows = []  # (style, expected, predicted)
    untrusted = []
    for s in samples:
        meta = json.loads((s / "labels.json").read_text())
        if meta.get("app") not in ("focus_outline", "focus_styles"):
            continue
        video = next(s.glob("video.*"))
        offset, ok = sample_offset(s, meta, video)
        if not ok:
            untrusted.append(s.name)
        for r in detect_focus(video, load_keys(s / "keys.jsonl"), offset=offset, cfg=DEFAULT):
            e = check_for(meta, r.key_t)
            if e:
                rows.append((e[0], e[1], r.verdict))
                if r.verdict != e[1]:
                    details.append((s.name, e, r))
    if not rows:
        raise SystemExit("No focus samples found. Pass sample directories, e.g. benchmark/data/*")

    by_style = defaultdict(Counter)
    expected_of = {}
    for style, exp, pred in rows:
        by_style[style][pred] += 1
        expected_of[style] = exp
    print(f"{'style':18} {'expected':12} {'n':>3} {'visible':>8} {'not_vis':>8} {'ambig':>6}")
    for style in sorted(by_style, key=lambda s: (VERDICTS.index(expected_of[s]), s)):
        c = by_style[style]
        print(f"{style:18} {expected_of[style]:12} {sum(c.values()):3d} {c['visible']:8d} "
              f"{c['not_visible']:8d} {c['ambiguous']:6d}")

    conf = Counter((e, p) for _, e, p in rows)
    print("\nConfusion matrix (rows = expected, columns = predicted)")
    print(f"{'':14}" + "".join(f"{v:>13}" for v in VERDICTS))
    for e in VERDICTS:
        print(f"{e:14}" + "".join(f"{conf[(e, p)]:13d}" for p in VERDICTS))

    n = len(rows)
    escalated = sum(1 for _, _, p in rows if p == "ambiguous")
    wrong_confident = sum(1 for _, e, p in rows if p != "ambiguous" and e != "ambiguous" and p != e)
    overconfident = sum(1 for _, e, p in rows if e == "ambiguous" and p != "ambiguous")
    tp = conf[("not_visible", "not_visible")]
    fp = sum(1 for _, e, p in rows if p == "not_visible" and e != "not_visible")
    fn = sum(1 for _, e, p in rows if e == "not_visible" and p != "not_visible")
    pr, rc, f1 = prf(tp, fp, fn)
    print(f"\n{n} Tab presses from {len(samples)} samples")
    print(f"  escalated to VLM (ambiguous):      {escalated} ({escalated / n:.0%})")
    print(f"  confident but WRONG:               {wrong_confident}")
    print(f"  confident on an ambiguous case:    {overconfident}")
    print(f"  failure detection: TP={tp} FP={fp} FN={fn}  precision={pr:.2f} recall={rc:.2f} F1={f1:.2f}")
    if untrusted:
        print("  Clock sync NOT trusted for:", ", ".join(untrusted))
    if detail:
        print("\nEscalated or wrong presses:")
        for name, (style, exp, idx, prev), r in details:
            where = f"tab {idx}" if idx else "tab ?"
            print(f"  {name}  {where}  style={style}  after={prev}  expected={exp}  got={r.verdict}"
                  f"  conf={r.confidence}\n      {r.reasoning}")


if __name__ == "__main__":
    main(sys.argv[1:])