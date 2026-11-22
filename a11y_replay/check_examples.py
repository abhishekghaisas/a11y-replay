"""Check, and optionally repair, the demo example files against their checksums.

    python -m a11y_replay.check_examples            # report only
    python -m a11y_replay.check_examples --fix      # move files that are in the wrong place

Needs demo_examples/CHECKSUMS.sha256 (one "<sha256>  <path>" line per file, paths relative to the repository root).
For every file under demo_examples/ it says whether the file is where it belongs, or is a known example file in the WRONG place
(for example `video.mp4` of one example saved into another example's folder, or saved as `video (1).mp4`), or matches no known
file at all (modified, regenerated, or something else). --fix only ever moves known files to their correct place, and never
overwrites a file it cannot account for.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

SKIP = {"README.md", "CHECKSUMS.sha256", ".DS_Store"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checksums(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, _, rel = line.partition("  ")
            out[rel.strip()] = digest.strip()
    return out


def scan(root: Path, checksums: dict, examples_dir: str = "demo_examples") -> dict:
    """{'ok': [...], 'misplaced': [(found_rel, belongs_rel)], 'unknown': [...], 'missing': [...]} (paths relative to root)."""
    by_hash = {}
    for rel, digest in checksums.items():
        by_hash.setdefault(digest, []).append(rel)
    ok, misplaced, unknown = [], [], []
    base = Path(root) / examples_dir
    for p in sorted(base.rglob("*")) if base.exists() else []:
        if not p.is_file() or p.name in SKIP:
            continue
        rel = p.relative_to(root).as_posix()
        digest = sha256_of(p)
        if checksums.get(rel) == digest:
            ok.append(rel)
        elif digest in by_hash:
            misplaced.append((rel, by_hash[digest][0]))
        else:
            unknown.append(rel)
    present_ok = set(ok) | {b for _, b in misplaced}
    missing = [rel for rel in checksums if rel not in present_ok]
    return {"ok": ok, "misplaced": misplaced, "unknown": unknown, "missing": missing}


def fix(root: Path, result: dict) -> list:
    """Move every misplaced known file to its correct path. Two phases, so a swap never overwrites anything. Returns the
    moves made and the moves refused (a destination holding a file that is not one of the files being moved)."""
    root = Path(root)
    sources = {a for a, _ in result["misplaced"]}
    moves, refused = [], []
    plan = []
    for found, belongs in result["misplaced"]:
        dest = root / belongs
        if dest.exists() and belongs not in sources:
            refused.append((found, belongs))           # something we cannot account for sits there; leave everything alone
        else:
            plan.append((found, belongs))
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        staged = []
        for i, (found, belongs) in enumerate(plan):
            t = Path(tmp) / str(i)
            shutil.move(str(root / found), t)
            staged.append((t, belongs, found))
        for t, belongs, found in staged:
            (root / belongs).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(t), root / belongs)
            moves.append((found, belongs))
    return moves, refused


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="the repository root (default: current folder)")
    ap.add_argument("--checksums", default="demo_examples/CHECKSUMS.sha256")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    cs_path = root / a.checksums
    if not cs_path.exists():
        print(f"Cannot find {cs_path}.")
        return 2
    checksums = parse_checksums(cs_path)
    r = scan(root, checksums)
    print(f"{len(r['ok'])} of {len(checksums)} files are in the right place.")
    for found, belongs in r["misplaced"]:
        print(f"  WRONG PLACE: {found}\n               has the contents of {belongs}")
    for rel in r["unknown"]:
        print(f"  UNKNOWN:     {rel} matches none of the expected files (modified, regenerated, or not part of this set)")
    for rel in r["missing"]:
        print(f"  MISSING:     {rel} (no file anywhere under demo_examples/ has these contents)")
    if r["misplaced"] and not a.fix:
        print("\nRun again with --fix to move the misplaced files to where they belong.")
    if a.fix and r["misplaced"]:
        moves, refused = fix(root, r)
        for found, belongs in moves:
            print(f"  moved {found} -> {belongs}")
        for found, belongs in refused:
            print(f"  NOT MOVED: {found} (the destination {belongs} holds a file I cannot account for)")
        r = scan(root, checksums)
        print(f"\nAfter fixing: {len(r['ok'])} of {len(checksums)} files are in the right place.")
    return 0 if len(r["ok"]) == len(checksums) else 1


if __name__ == "__main__":
    raise SystemExit(main())