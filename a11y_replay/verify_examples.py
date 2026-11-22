"""Check that each demo example BEHAVES as its folder name says, whoever made the files.

    python -m a11y_replay.verify_examples [demo_examples]

Runs every example through the same code the demo uses and compares the counts of findings with what the example is meant to show
(for instance: `focus_page_with_bug` must give at least five focus findings and judge no press fine). If a folder fails, it also says
whether the result looks like one of the OTHER examples, which usually means files were put in the wrong folder.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from a11y_replay.demo import find_examples, run_demo

# (criterion, status shown in the demo's table, minimum, maximum). Status: Finding | Needs review | Judged fine.
EXPECT = {
    "focus_page_with_bug": ("every Tab press reported as a finding", [("2.4.7", "Finding", 5, 99), ("2.4.7", "Judged fine", 0, 0)]),
    "focus_page_clean": ("focus rings judged fine, none reported as failures", [("2.4.7", "Finding", 0, 0), ("2.4.7", "Judged fine", 4, 99)]),
    "flashing_banner_with_bug": ("one flashing finding", [("2.3.1", "Finding", 1, 1)]),
    "flashing_banner_clean": ("no findings at all", [("2.3.1", "Finding", 0, 0), ("2.4.7", "Finding", 0, 0), ("4.1.3", "Finding", 0, 0)]),
    "status_messages_two_silent": ("two unannounced messages, one announced", [("4.1.3", "Finding", 2, 2), ("4.1.3", "Judged fine", 1, 1)]),
}


def counts_of(rows) -> Counter:
    return Counter((r[1].replace("WCAG ", ""), r[2]) for r in rows)


def satisfies(counts: Counter, rules) -> bool:
    return all(lo <= counts.get((crit, status), 0) <= hi for crit, status, lo, hi in rules)


def verify(examples_dir: str | Path = "demo_examples") -> list:
    """List of (name, passed, observed counts, message). Never raises."""
    found = {Path(p).name: p for p in find_examples([str(examples_dir)]).values()}
    out = []
    for name in sorted(found):
        if name not in EXPECT:
            out.append((name, None, Counter(), "no expectation defined for this folder name; skipped"))
            continue
        r = run_demo(None if not found else name, examples=found, show_passes=True, keep_files=False)
        if not r.ok:
            out.append((name, False, Counter(), r.summary))
            continue
        c = counts_of(r.rows)
        meant, rules = EXPECT[name]
        if satisfies(c, rules):
            out.append((name, True, c, f"as expected: {meant}"))
        else:
            looks_like = [n for n, (_, rs) in EXPECT.items() if n != name and satisfies(c, rs)]
            hint = f" This result looks like: {', '.join(looks_like)}." if looks_like else ""
            out.append((name, False, c, f"expected: {meant}.{hint}"))
    for name in EXPECT:
        if name not in found:
            out.append((name, False, Counter(), "folder not found"))
    return out


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    results = verify(args[0] if args else "demo_examples")
    bad = 0
    for name, passed, c, msg in results:
        tag = "PASS " if passed else ("SKIP " if passed is None else "FAIL ")
        seen = ", ".join(f"{n} x {crit} {status}" for (crit, status), n in sorted(c.items())) or "no items"
        print(f"{tag} {name:28} {seen}\n      {msg}")
        bad += passed is False
    print(f"\n{sum(1 for r in results if r[1])} of {len(EXPECT)} examples behave as expected." if not bad else
          f"\n{bad} example(s) do NOT behave as expected.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())