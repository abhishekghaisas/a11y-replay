"""Bake-off: how well do VLMs judge the focus presses the classical layer cannot decide?

Usage (from the repo root; needs ANTHROPIC_API_KEY in .env for Claude models):
    python -m eval.eval_vlm benchmark/data/* --models claude-sonnet-5-5
    python -m eval.eval_vlm benchmark/data/* --models claude-sonnet-5-5,claude-haiku-5-5,claude-opus-5-5
    python -m eval.eval_vlm benchmark/data/* --models claude-sonnet-5-5,ollama:qwen3-vl:8b --scope all
    python -m eval.eval_vlm benchmark/data/* --mock          # dry run: no API, fake answers, pipeline check

--scope residue (default): judge only the presses the classical layer left ambiguous.
--scope all:               judge every press too, so the model alone can be compared with the hybrid
                           and so the "no confident mistakes on real failures" rule has data (the
                           classical layer already resolves almost all real failures).

Responses are cached in eval/results/vlm_cache/<model>/, keyed by model settings, prompt version and
images, so reruns are free and repeatable. Delete the cache to force fresh calls.

DECISION RULE (fixed before any real results existed): pick the cheapest model that makes NO confident
mistake on presses expected to be failures AND agrees with the expected "visible" on at least 90% of the
presses expected to be visible. If Opus 5.5 beats the default by more than a press or two, switch.

Caveats: the benchmark styles and their expected verdicts were written by the person who wrote the
detector; the prompt describes indicators generically, which overlaps in kind with some styles; the
set is small. A pass here is a development signal, not accuracy evidence.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from a11y_replay.cascade import ANSWER_TO_VERDICT, judge_press
from a11y_replay.config import DEFAULT
from a11y_replay.detectors.focus import detect_focus
from a11y_replay.ingest.keys import load_keys
from a11y_replay.ingest.video import probe
from a11y_replay.vlm.client import AnthropicClient, CachedClient, MockClient, OllamaClient
from a11y_replay.vlm.pricing import estimate_cost
from a11y_replay.vlm.prompts import PROMPT_VERSION
from eval.common import sample_offset
from eval.eval_focus import check_for
from eval.metrics import prf

VERDICTS = ("visible", "not_visible", "ambiguous")
MOCK_ANSWER = '{"indicator_visible": "unsure", "indicator_type": "", "evidence": "dry run"}'


def make_client(spec: str, cache_root: Path, effort, thinking, mock: bool):
    slug = spec.replace(":", "_").replace("/", "_")
    if mock:
        return MockClient(lambda c: MOCK_ANSWER, model=f"mock:{spec}")
    inner = (OllamaClient(spec[len("ollama:"):]) if spec.startswith("ollama:")
             else AnthropicClient(model=spec, effort=effort, thinking=thinking))
    return CachedClient(inner, cache_root / slug)


def collect(samples):
    rows = []
    for s in samples:
        meta = json.loads((s / "labels.json").read_text())
        if meta.get("app") not in ("focus_outline", "focus_styles") or not (s / "keys.jsonl").exists():
            continue
        video = next(s.glob("video.*"))
        offset, _ = sample_offset(s, meta, video)
        fps = probe(video).fps or 25.0
        for r in detect_focus(video, load_keys(s / "keys.jsonl"), offset=offset, cfg=DEFAULT):
            gt = check_for(meta, r.key_t)
            if gt:
                rows.append({"sample": s.name, "video": video, "fps": fps, "style": gt[0],
                             "expected": gt[1], "classical": r.verdict, "result": r, "answers": {}})
    return rows


def hybrid_verdict(row, spec):
    if row["classical"] != "ambiguous":
        return row["classical"]
    j = row["answers"].get(spec)
    return ANSWER_TO_VERDICT[j.answer] if j else "ambiguous"


def confusion(pairs):
    c = Counter(pairs)
    out = [f"{'':14}" + "".join(f"{v:>13}" for v in VERDICTS)]
    for e in VERDICTS:
        out.append(f"{e:14}" + "".join(f"{c[(e, p)]:13d}" for p in VERDICTS))
    return "\n".join(out)


def report_model(spec, client, rows, scope):
    judged = [r for r in rows if spec in r["answers"]]
    ans = lambda r: r["answers"][spec].answer
    print(f"\n{'=' * 78}\nMODEL {spec}   (prompt {PROMPT_VERSION})   judged presses: {len(judged)}")
    if not judged:
        print("  nothing to judge")
        return
    by = {e: Counter(ans(r) for r in judged if r["expected"] == e) for e in VERDICTS}
    print(f"\n  {'expected':12} {'n':>3} {'yes':>5} {'no':>5} {'unsure':>7}")
    for e in VERDICTS:
        n = sum(by[e].values())
        print(f"  {e:12} {n:3d} {by[e]['yes']:5d} {by[e]['no']:5d} {by[e]['unsure']:7d}")

    styles = sorted({r["style"] for r in judged}, key=lambda st: (VERDICTS.index(next(r["expected"] for r in judged if r["style"] == st)), st))
    print(f"\n  {'style':18} {'expected':12} {'n':>3} {'yes':>5} {'no':>5} {'unsure':>7}")
    for st in styles:
        rs = [r for r in judged if r["style"] == st]
        c = Counter(ans(r) for r in rs)
        print(f"  {st:18} {rs[0]['expected']:12} {len(rs):3d} {c['yes']:5d} {c['no']:5d} {c['unsure']:7d}")

    nv, nf = sum(by["visible"].values()), sum(by["not_visible"].values())
    agree = by["visible"]["yes"] / nv if nv else None
    conf_mistakes = by["not_visible"]["yes"]
    wrong_no = by["visible"]["no"]
    print(f"\n  agreement on expected-visible (said yes): {by['visible']['yes']}/{nv}"
          + (f" = {agree:.0%}" if nv else ""))
    print(f"  said NO on an expected-visible press (false alarm if reported): {wrong_no}")
    print(f"  said YES on an expected-failure press (confident mistake):      {conf_mistakes}/{nf}")
    print(f"  ambiguous-by-design presses: committed to yes/no on "
          f"{by['ambiguous']['yes'] + by['ambiguous']['no']}/{sum(by['ambiguous'].values())}")
    bad = [r for r in judged if not r["answers"][spec].parse_ok]
    print(f"  replies that were not valid JSON after one retry: {len(bad)}")
    rule_ok = (conf_mistakes == 0 if nf else None), (agree is not None and agree >= 0.9) if nv else None
    note = [] if nf else ["no expected-failure presses were judged (use --scope all)"]
    print(f"  decision rule: failures-no-mistake={rule_ok[0]}  visible>=90%={rule_ok[1]}"
          + (f"   [{'; '.join(note)}]" if note else ""))
    if len(judged) < 20:
        print(f"  WARNING: only {len(judged)} judged presses; treat as anecdotal")

    m = client.meter
    cost = estimate_cost(client.model, m.input_tokens, m.output_tokens)
    calls_real = m.calls - m.cache_hits
    print(f"\n  calls {m.calls} (cache hits {m.cache_hits}, new {calls_real}); tokens in/out "
          f"{m.input_tokens}/{m.output_tokens}; mean latency {m.latency_s / max(m.calls, 1):.1f}s; "
          + (f"estimated cost for these calls ${cost:.4f}" if cost is not None else "cost: n/a (dry run or unknown price)"))

    pairs = [(r["expected"], hybrid_verdict(r, spec)) for r in rows]
    tp = sum(1 for e, p in pairs if e == "not_visible" and p == "not_visible")
    fp = sum(1 for e, p in pairs if e != "not_visible" and p == "not_visible")
    fn = sum(1 for e, p in pairs if e == "not_visible" and p != "not_visible")
    wrong = sum(1 for e, p in pairs if p != "ambiguous" and e != "ambiguous" and p != e)
    pr, rc, f1 = prf(tp, fp, fn)
    print(f"\n  HYBRID (classical where confident, this model for the residue), all {len(rows)} presses")
    print("  " + confusion(pairs).replace("\n", "\n  "))
    print(f"  confident but wrong: {wrong}   failures: TP={tp} FP={fp} FN={fn}  precision={pr:.2f} recall={rc:.2f}")
    if scope == "all":
        pairs_m = [(r["expected"], ANSWER_TO_VERDICT[ans(r)]) for r in judged]
        print(f"\n  MODEL ALONE (no classical layer), {len(pairs_m)} presses")
        print("  " + confusion(pairs_m).replace("\n", "\n  "))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+")
    ap.add_argument("--models", default=DEFAULT.vlm_model, help="comma list; ollama models as ollama:<name>")
    ap.add_argument("--scope", choices=("residue", "all"), default="residue")
    ap.add_argument("--effort", default=None, help="override effort (Claude), e.g. low|medium|high")
    ap.add_argument("--thinking", default=None, choices=("between_tools", "adaptive"))
    ap.add_argument("--cache-dir", type=Path, default=Path("eval/results/vlm_cache"))
    ap.add_argument("--out", type=Path, default=None, help="write per-press results as JSON lines")
    ap.add_argument("--max-calls", type=int, default=300, help="safety cap on model calls")
    ap.add_argument("--mock", action="store_true", help="dry run with fake answers; no API calls")
    a = ap.parse_args(argv)

    samples = [Path(p) for p in a.samples if (Path(p) / "labels.json").exists()]
    rows = collect(samples)
    if not rows:
        raise SystemExit("No focus samples found.")
    specs = [m.strip() for m in a.models.split(",") if m.strip()]
    todo = [r for r in rows if a.scope == "all" or r["classical"] == "ambiguous"]
    planned = len(todo) * len(specs)
    print(f"{len(rows)} labeled presses from {len(samples)} samples; {len(todo)} to judge x {len(specs)} "
          f"model(s) = up to {planned} calls (cached ones are free)" + ("  [DRY RUN]" if a.mock else ""))
    if planned > a.max_calls:
        raise SystemExit(f"{planned} calls exceeds --max-calls {a.max_calls}; raise it if intended.")

    clients = {s: make_client(s, a.cache_dir, a.effort, a.thinking, a.mock) for s in specs}
    for spec, client in clients.items():
        for r in todo:
            r["answers"][spec] = judge_press(client, r["video"], r["result"], r["fps"], DEFAULT)
    for spec, client in clients.items():
        report_model(spec, client, rows, a.scope)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps({
                    "sample": r["sample"], "style": r["style"], "expected": r["expected"],
                    "classical": r["classical"], "prompt": PROMPT_VERSION, "when": time.strftime("%Y-%m-%d %H:%M"),
                    "answers": {s: {"answer": j.answer, "type": j.indicator_type, "evidence": j.evidence,
                                    "parse_ok": j.parse_ok, "raw": j.raw_text} for s, j in r["answers"].items() if j},
                }) + "\n")
        print(f"\nper-press results written to {a.out}")


if __name__ == "__main__":
    main()