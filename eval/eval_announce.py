"""Evaluate the seen-versus-heard detector (WCAG 4.1.3) on generated status-message samples.

Usage (from the repo root; needs the Tesseract program: brew install tesseract):
    python -m eval.eval_announce benchmark/data/*
    python -m eval.eval_announce benchmark/data/* --noise 0.15 --delay 1.0   # degrade the transcript
    python -m eval.eval_announce benchmark/data/* --sweep-noise 0,0.05,0.1,0.2,0.3 --seeds 5
    python -m eval.eval_announce benchmark/data/* --transcript asr     # real audio -> Whisper (see below)

--transcript asr --show-asr additionally prints every true line next to what the recognizer heard.

--transcript asr uses `audio.wav` (make it with `python -m benchmark.tts benchmark/data/*`, macOS
only) transcribed by faster-whisper (`pip install faster-whisper`; first use downloads a model). It
also reports how well the recognizer did against the true lines: word error rate, timing error, missed
and invented lines. The audio is clean synthetic speech from one voice, an optimistic case.

Ground truth (labels.json "status_checks"): each message is announced or silent. A silent one is a
failure. The positive class is "unannounced".

The transcript here is a SIMULATED, error-free list of utterances written by the benchmark
generator. --noise replaces/deletes that fraction of characters to imitate recognition errors, and
--delay shifts every utterance later, to see where matching breaks. These are crude stand-ins for
real speech recognition and real screen-reader timing, not measurements of them.

Caveats: the pages and messages were written by the detector's author, and there are only a handful
of messages per sample. A good score is a development check, not accuracy evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from dataclasses import replace

from a11y_replay.config import DEFAULT
from a11y_replay.detectors.announce import match_observations, observe_changes, similarity
from a11y_replay.events import Utterance
from a11y_replay.ingest.keys import load_keys
from eval.common import sample_offset, word_error_rate
from eval.metrics import prf

TOL = 1.0  # seconds: a detection counts as a given message if its onset is this close


def corrupt(text: str, p: float, rng: random.Random) -> str:
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < p:
            if rng.random() < 0.5:
                out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))   # substitution
            # else: deletion
        else:
            out.append(ch)
    return "".join(out)


def load(samples, transcript="simulated", whisper_model="base"):
    data = []
    for s in samples:
        meta = json.loads((s / "labels.json").read_text())
        if not str(meta.get("app", "")).startswith("status_messages") or not (s / "utterances.json").exists():
            continue
        video = next(s.glob("video.*"))
        offset, ok = sample_offset(s, meta, video)
        obs = observe_changes(video, load_keys(s / "keys.jsonl"), offset=offset, cfg=DEFAULT)
        utts = [Utterance(**u) for u in json.loads((s / "utterances.json").read_text())]
        asr = None
        if transcript == "asr":
            if not (s / "audio.wav").exists():
                raise SystemExit(f"{s.name}: no audio.wav; run  python -m benchmark.tts benchmark/data/*")
            from a11y_replay.ingest.audio import load_or_transcribe
            asr = load_or_transcribe(s, whisper_model)
        data.append({"name": s.name, "meta": meta, "obs": obs, "utts": utts, "asr": asr, "sync_ok": ok})
    return data


def asr_quality(data):
    """Compare recognizer output with the true lines: word error rate, start-time error, missed, invented."""
    wers, terrs, missed, invented, n = [], [], 0, 0, 0
    for d in data:
        pool = list(d["asr"])
        for u in d["utts"]:
            n += 1
            near = [a for a in pool if abs(a.start - u.start) <= 2.5]
            if not near:
                missed += 1
                continue
            a = min(near, key=lambda a: abs(a.start - u.start))
            pool.remove(a)
            wers.append(word_error_rate(u.text, a.text))
            terrs.append(a.start - u.start)
        invented += len(pool)
    return dict(n=n, wer=wers, terr=terrs, missed=missed, invented=invented)


def show_transcripts(data):
    """Print, per sample, the true lines and what the recognizer heard, in time order."""
    for d in data:
        rows = [(u.start, "TRUE", u.text) for u in d["utts"]] + [(a.start, "ASR ", a.text) for a in d["asr"]]
        print(f"\n  {d['name']}")
        for start, who, text in sorted(rows):
            print(f"    {start:7.2f}s  {who}  {text!r}")
        print("    on-screen messages (time, true text):",
              [(round(c["t"], 2), c["text"], "announced" if c["announced"] else "SILENT") for c in d["meta"]["status_checks"]])


def score(data, noise=0.0, delay=0.0, seed=0, verbose=False, cfg=DEFAULT):
    rng = random.Random(seed)
    tp = fp = fn = tn = missed = spurious = review_only = 0
    ocr_sims, by_trigger, by_cat = [], {}, {}
    for d in data:
        source = d["asr"] if d.get("asr") is not None else d["utts"]
        utts = [Utterance(u.start + delay, u.end + delay, corrupt(u.text, noise, rng)) for u in source]
        results = match_observations(d["obs"], utts, cfg=cfg)
        taken = set()
        for c in d["meta"]["status_checks"]:
            cand = [(abs(r.t - c["t"]), i) for i, r in enumerate(results) if i not in taken and abs(r.t - c["t"]) <= TOL]
            if not cand:
                missed += 1
                if not c["announced"]:
                    fn += 1
                continue
            _, i = min(cand)
            taken.add(i)
            r = results[i]
            if r.verdict == "needs_review":
                review_only += 1
            pred_silent = r.verdict == "unannounced"
            if r.text:
                ocr_sims.append(similarity(r.text, c["text"]))
            truth_silent = not c["announced"]
            tp += pred_silent and truth_silent
            fp += pred_silent and not truth_silent
            fn += (not pred_silent) and truth_silent
            tn += (not pred_silent) and not truth_silent
            ok = pred_silent == truth_silent
            for dd in c.get("distractors", []):
                if dd["category"] != "unrelated":
                    b2 = by_cat.setdefault(dd["category"], [0, 0, dd["text"], 0])
                    b2[0] += pred_silent; b2[1] += 1; b2[3] += (r.verdict == "needs_review")
            b = by_trigger.setdefault(c["trigger"], [0, 0]); b[0] += ok; b[1] += 1
            if verbose and not ok:
                print(f"    WRONG {d['name']} t={c['t']} {c['kind']} announced={c['announced']} -> {r.verdict}: {r.reasoning}")
        spurious += sum(1 for i, r in enumerate(results) if i not in taken)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, missed=missed, spurious=spurious, review_only=review_only,
                ocr=ocr_sims, by_trigger=by_trigger, by_cat=by_cat)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+")
    ap.add_argument("--noise", type=float, default=0.0, help="fraction of letters corrupted in the transcript")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds added to every utterance start")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-guards", action="store_true", help="turn off the number/negation guards (to measure what they add)")
    ap.add_argument("--transcript", choices=("simulated", "asr"), default="simulated")
    ap.add_argument("--show-asr", action="store_true", help="with --transcript asr: list true lines vs what was heard")
    ap.add_argument("--whisper-model", default="base", help="faster-whisper model size, e.g. tiny, base, small")
    ap.add_argument("--sweep-noise", default=None, help="comma list of noise levels; averages over --seeds")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)

    samples = [Path(p) for p in a.samples if (Path(p) / "labels.json").exists()]
    data = load(samples, a.transcript, a.whisper_model)
    if not data:
        raise SystemExit("No status_messages samples with utterances.json found.")
    n_msgs = sum(len(d["meta"]["status_checks"]) for d in data)
    n_silent = sum(1 for d in data for c in d["meta"]["status_checks"] if not c["announced"])
    print(f"{len(data)} samples, {n_msgs} messages ({n_silent} silent = failures, {n_msgs - n_silent} announced)")
    bad = [d["name"] for d in data if not d["sync_ok"]]
    if bad:
        print("Clock sync NOT trusted for:", ", ".join(bad))

    if a.sweep_noise:
        print(f"\n{'noise':>6} {'precision':>10} {'recall':>8} {'F1':>6} {'false alarms':>13} {'misses':>7}   (mean over {a.seeds} seeds)")
        for lvl in [float(x) for x in a.sweep_noise.split(",")]:
            runs = [score(data, lvl, a.delay, s) for s in range(a.seeds)]
            P = statistics.mean(prf(r["tp"], r["fp"], r["fn"])[0] for r in runs)
            R = statistics.mean(prf(r["tp"], r["fp"], r["fn"])[1] for r in runs)
            F = statistics.mean(prf(r["tp"], r["fp"], r["fn"])[2] for r in runs)
            print(f"{lvl:6.2f} {P:10.2f} {R:8.2f} {F:6.2f} {statistics.mean(r['fp'] for r in runs):13.1f} {statistics.mean(r['fn'] for r in runs):7.1f}")
        return

    if a.transcript == "asr":
        q = asr_quality(data)
        print(f"\nSpeech recognition vs the true lines ({a.whisper_model}): {q['n']} lines, "
              f"missed {q['missed']}, invented {q['invented']}")
        if q["wer"]:
            print(f"  word error rate: mean {statistics.mean(q['wer']):.2f}, worst {max(q['wer']):.2f}   "
                  f"start-time error: mean {statistics.mean(q['terr']):+.2f}s, worst {max(q['terr'], key=abs):+.2f}s")

    if a.transcript == "asr" and a.show_asr:
        show_transcripts(data)

    cfg = replace(DEFAULT, announce_content_guards=not a.no_guards)
    r = score(data, a.noise, a.delay, a.seed, verbose=True, cfg=cfg)
    p, rc, f1 = prf(r["tp"], r["fp"], r["fn"])
    print(f"\ntranscript: {a.transcript}   extra degradation: noise={a.noise} delay={a.delay}s seed={a.seed}")
    print(f"  silent messages flagged (TP): {r['tp']}   false alarms on announced ones (FP): {r['fp']}   "
          f"silent ones missed (FN): {r['fn']}   announced ones passed (TN): {r['tn']}")
    print(f"  precision={p:.2f} recall={rc:.2f} F1={f1:.2f}")
    print(f"  messages with no detection at all: {r['missed']}   spurious detections: {r['spurious']}   "
          f"needs_review: {r['review_only']}")
    if r["ocr"]:
        print(f"  OCR text vs true text similarity: mean {statistics.mean(r['ocr']):.0f}, min {min(r['ocr']):.0f}")
    if r["by_cat"]:
        print(f"  SILENT messages with wrong speech nearby (correct = still flagged as silent), guards {'OFF' if a.no_guards else 'on'}:")
        for cat, (flagged, n, example, reviewed) in sorted(r["by_cat"].items()):
            print(f"      {cat:13} {flagged}/{n} flagged as silent, {reviewed} more sent to review   e.g. spoken {example!r}")
    for trig, (ok, n) in sorted(r["by_trigger"].items()):
        print(f"  trigger={trig:5}: {ok}/{n} correct")


if __name__ == "__main__":
    main()