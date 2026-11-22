"""Check what git is about to publish, BEFORE the first push.

    python -m a11y_replay.prepush

Looks at every file git would commit (tracked files, plus untracked files that are not ignored) and reports:
  BLOCK  things that must never be published: real recordings, generated reports (they embed frames of your screen),
         .env or key files, API keys and tokens inside files, macOS litter.
  WARN   things worth a look: files over 5 MB, your home-folder path inside a file, a missing LICENSE.
Exit code 0 means no BLOCK. This is a safety net, not a guarantee: it cannot recognize private content it has no rule for,
such as a screenshot of your own desktop that was named something innocent.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

BIG = 5 * 1024 * 1024
TEXT_LIMIT = 2 * 1024 * 1024
VIDEO_AUDIO = {".mov", ".avi", ".mkv", ".m4a", ".wav", ".webm", ".mp4"}
SECRET_PATTERNS = [
    ("an Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("a GitHub token", re.compile(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("an AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("an API key assigned to ANTHROPIC_API_KEY",
     re.compile(r"""ANTHROPIC_API_KEY\s*[:=]\s*['"]?(?!your|<|\$|xxx|\.\.\.|sk-ant-\.\.\.)[A-Za-z0-9_\-]{24,}""", re.I)),
]
HOME_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+/|C:\\Users\\[A-Za-z0-9._ -]+\\")
EXPECTED = ["README.md", "streamlit_app.py", "requirements.txt", "packages.txt"]


def git(args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True)


def candidate_files(root: Path) -> list:
    r = git(["ls-files", "-z", "--cached", "--others", "--exclude-standard"], root)
    return sorted({p for p in r.stdout.decode("utf-8", "replace").split("\0") if p and (root / p).is_file()})


def classify(rel: str):
    """BLOCK reason for a path, or None. Pure path rules."""
    p = PurePosixPath(rel)
    parts, name, ext = p.parts, p.name, p.suffix.lower()
    if name == ".gitkeep":                       # an intentional empty placeholder that keeps a folder in the repository
        return None
    if name == ".DS_Store":
        return "macOS litter"
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "an environment file (it normally holds your API key)"
    if name in ("secrets.toml", "id_rsa", "id_ed25519") or ext in (".pem", ".key", ".p12"):
        return "a secrets or key file"
    if "audit_report" in parts:
        return "a generated report (it embeds frames of whatever was on screen)"
    if parts[0] in ("recordings",) or parts[:2] == ("benchmark", "data"):
        return "a real recording or generated benchmark data"
    if ext in VIDEO_AUDIO and not (parts[0] == "demo_examples" and ext == ".mp4") and not (parts[:2] == ("docs", "examples")):
        return "a video or audio file outside demo_examples/"
    return None


def scan_text(root: Path, rel: str):
    p = root / rel
    try:
        size = p.stat().st_size
        if size > TEXT_LIMIT:
            return [], []
        data = p.read_bytes()
    except OSError:
        return [], []
    if b"\0" in data[:8000]:
        return [], []
    text = data.decode("utf-8", "replace")
    secrets = [what for what, rx in SECRET_PATTERNS if rx.search(text)]
    homes = bool(HOME_PATH.search(text))
    return secrets, homes


def check(root: str | Path = ".") -> dict:
    root = Path(root)
    if git(["rev-parse", "--is-inside-work-tree"], root).returncode != 0:
        return {"error": "This folder is not a git repository yet. Run `git init -b main`, then run this check again."}
    files = candidate_files(root)
    block, warn = [], []
    for rel in files:
        reason = classify(rel)
        if reason:
            block.append((rel, f"is {reason}"))
        secrets, homes = scan_text(root, rel)
        for what in secrets or []:
            block.append((rel, f"contains what looks like {what}"))
        if homes:
            warn.append((rel, "contains a path from your own computer (/Users/<name>/...)"))
        size = (root / rel).stat().st_size
        if size > BIG:
            warn.append((rel, f"is {size / 1e6:.1f} MB"))
    for need in EXPECTED:
        if need not in files:
            warn.append((need, "is expected in the repository but would not be committed"))
    examples = sorted({PurePosixPath(f).parts[1] for f in files if f.startswith("demo_examples/") and f.endswith("video.mp4")})
    if len(examples) < 5:
        warn.append(("demo_examples/", f"only {len(examples)} example video(s) would be committed; the hosted demo expects 5"))
    if not any(PurePosixPath(f).name.upper().startswith("LICENSE") for f in files):
        warn.append(("LICENSE", "no license file: without one, others have no right to reuse the code"))
    return {"files": files, "block": block, "warn": warn}


def main(argv=None) -> int:
    root = (argv or sys.argv[1:] or ["."])[0]
    r = check(root)
    if "error" in r:
        print(r["error"])
        return 2
    print(f"{len(r['files'])} files would be committed.")
    for rel, why in r["block"]:
        print(f"  BLOCK  {rel} {why}")
    for rel, why in r["warn"]:
        print(f"  WARN   {rel} {why}")
    if r["block"]:
        print("\nDo NOT push yet. Fix every BLOCK line (usually: add the path to .gitignore and `git rm --cached` it).")
        return 1
    print("\nNo BLOCK lines." + (" Read the WARN lines, then it is reasonable to commit." if r["warn"] else " Looks clean."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())