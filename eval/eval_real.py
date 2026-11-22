"""Score the audit against HUMAN labels on real recordings.

Usage (from the repo root):
    python -m eval.eval_real recordings/*
    python -m eval.eval_real recordings/* --model claude-sonnet-5-5 --transcribe --transcript-clock video

Each recording folder holds what the audit needs (video, keys.jsonl, optional transcript or audio) plus
`human_labels.json`, written by a person who watched the video WITHOUT seeing the audit's output:

    {"labeler": "Jane Doe, accessibility tester", "independent": true, "complete": true,
     "issues": [{"t": 12.4, "criterion": "2.4.7", "note": "focus disappears on the Pay button"}]}

`t` is seconds on the VIDEO's own timeline (what a player shows). `independent` says the labeler is not the
tool's author; `complete` says they reviewed the whole recording, which is the only case where an unmatched
finding can be called a false alarm. If either is false, the output says what that means.

A finding matches a label of the same criterion within a tolerance (focus 0.75 s, the others 2 s). Items the audit
could only mark "needs review" count as flagged-for-review, not as findings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from a11y_replay.audit import run_audit

TOL = {"2.4.7": 0.75, "4.1.3": 2.0, "2.3.1": 2.0}
DEFAULT_TOL = 2.0


def _match(items: list, labels: list, tol: float):
    """Greedy nearest matching. items/labels are lists of times. Returns (pairs, unmatched_items, unmatched_labels)."""
    free = list(range(len(items)))
    pairs, missed = [], []
    for li, lt in sorted(enumerate(labels), key=lambda x: x[1]):
        cand = [(abs(items[i] - lt), i) for i in free if abs(items[i] - lt) <= tol]
        if cand:
            _, i = min(cand)
            free.remove(i)
            pairs.append((i, li))
        else:
            missed.append(li)
    return pairs, free, missed


def compare(report, labels: dict) -> dict:
    off = report.sync["offset_s"]
    issues = labels.get("issues", [])
    criteria = sorted({i["criterion"] for i in issues} | {i.criterion for i in report.items})
    per = {}
    for c in criteria:
        lab = [i for i in issues if i["criterion"] == c]
        fnd = [i for i in report.findings if i.criterion == c]
        rev = [i for i in report.needs_review if i.criterion == c]
        tol = TOL.get(c, DEFAULT_TOL)
        # audit times are on the key-log clock; humans label the video, so convert
        pairs, extra, missed = _match([i.start + off for i in fnd], [l["t"] for l in lab], tol)
        # of the labels the findings missed, which did the audit at least flag for review?
        rev_pairs, _, still_missed = _match([i.start + off for i in rev], [lab[m]["t"] for m in missed], tol)
        per[c] = {
            "tp": len(pairs),
            "fp": len(extra) if labels.get("complete") else None,
            "unconfirmed": 0 if labels.get("complete") else len(extra),
            "fn": len(missed) - len(rev_pairs),
            "review_only": len(rev_pairs),
            "missed_labels": [lab[missed[k]] for k in still_missed],
            "extra_findings": [{"t": round(fnd[i].start + off, 2), "title": fnd[i].title} for i in extra],
        }
    return {"recording": report.recording, "labeler": labels.get("labeler", "?"), "independent": bool(labels.get("independent")),
            "complete": bool(labels.get("complete")), "sync_trusted": report.sync["trusted"], "offset_s": off, "per_criterion": per}


def evaluate_recording(rec_dir, out_dir=None, **audit_kwargs) -> dict:
    rec = Path(rec_dir)
    labels = json.loads((rec / "human_labels.json").read_text())
    return compare(run_audit(rec, out_dir, **audit_kwargs), labels)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--model", default=None)
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--transcript-clock", choices=("log", "video"), default="log")
    a = ap.parse_args(argv)
    client = None
    if a.model:
        from a11y_replay.vlm.client import AnthropicClient, CachedClient
        client = CachedClient(AnthropicClient(model=a.model), Path("eval/results/vlm_cache") / a.model)
    recs = [Path(r) for r in a.recordings if (Path(r) / "human_labels.json").exists()]
    if not recs:
        raise SystemExit("No recordings with human_labels.json found.")
    totals, warnings = {}, []
    for rec in recs:
        r = evaluate_recording(rec, client=client, transcribe=a.transcribe, transcript_clock=a.transcript_clock)
        print(f"\n{r['recording']}  (labeler: {r['labeler']}; independent={r['independent']}; complete={r['complete']}; "
              f"sync trusted={r['sync_trusted']}, offset {r['offset_s']:+.2f}s)")
        if not r["independent"]:
            warnings.append(f"{r['recording']}: labels are NOT independent")
        if not r["complete"]:
            warnings.append(f"{r['recording']}: labels not marked complete, so unmatched findings are 'unconfirmed', not false alarms")
        if not r["sync_trusted"]:
            warnings.append(f"{r['recording']}: clock sync not trusted; timing-based matches are unreliable")
        for c, d in r["per_criterion"].items():
            fp = "n/a" if d["fp"] is None else d["fp"]
            print(f"  WCAG {c}: found {d['tp']}, missed {d['fn']} (+{d['review_only']} only flagged for review), false alarms {fp}"
                  + (f", unconfirmed {d['unconfirmed']}" if d["unconfirmed"] else ""))
            for m in d["missed_labels"]:
                print(f"      MISSED  t={m['t']}  {m.get('note', '')}")
            for e in d["extra_findings"]:
                print(f"      EXTRA   t={e['t']}  {e['title']}")
            t = totals.setdefault(c, {"tp": 0, "fn": 0, "fp": 0, "review": 0, "fp_known": True})
            t["tp"] += d["tp"]; t["fn"] += d["fn"]; t["review"] += d["review_only"]
            if d["fp"] is None:
                t["fp_known"] = False
            else:
                t["fp"] += d["fp"]
    print(f"\nTotals over {len(recs)} recording(s)")
    for c, t in sorted(totals.items()):
        rec_ = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else float("nan")
        prec = (t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else float("nan")) if t["fp_known"] else None
        print(f"  WCAG {c}: found {t['tp']}, missed {t['fn']} (recall {rec_:.2f}), flagged-only {t['review']}"
              + (f", false alarms {t['fp']} (precision {prec:.2f})" if prec is not None else ", precision n/a (labels incomplete)"))
    for w in sorted(set(warnings)):
        print("  WARNING:", w)
    print("\nThese numbers describe these recordings and these labelers only.")


if __name__ == "__main__":
    main()