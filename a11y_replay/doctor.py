"""Check that this machine can run the audit, and say what is missing.

    python -m a11y_replay.doctor

Used before deploying (does the hosting environment have ffmpeg and Tesseract?) and shown inside the hosted demo, because
"it works on my laptop" is the usual way a deployment fails.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def check_environment(examples_dir: str | Path | None = None) -> list:
    """List of (name, ok, detail, required). `required` False means only one check (status messages) needs it."""
    out = []

    def add(name, ok, detail, required=True):
        out.append((name, bool(ok), detail, required))

    try:
        import cv2
        add("OpenCV (opencv-python-headless on a server)", True, f"version {cv2.__version__}")
    except Exception as e:
        add("OpenCV (opencv-python-headless on a server)", False, f"cannot import: {e}")
    try:
        import numpy
        add("NumPy", True, f"version {numpy.__version__}")
    except Exception as e:
        add("NumPy", False, f"cannot import: {e}")
    for tool, why in (("ffmpeg", "converts variable-frame-rate screen recordings"), ("ffprobe", "detects variable frame rate")):
        path = shutil.which(tool)
        add(f"{tool} program", path, path or f"not found on PATH ({why}; on Streamlit Cloud list it in packages.txt)")
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        add("Tesseract OCR", True, f"version {version}", required=False)
    except Exception as e:
        add("Tesseract OCR", False, f"not working ({type(e).__name__}); the status-message check will not run. "
            "Install the program (apt: tesseract-ocr, brew: tesseract) and `pip install pytesseract`", required=False)
    if examples_dir is not None:
        n = sum(1 for d in Path(examples_dir).glob("*") if d.is_dir() and any(d.glob("video.*"))) if Path(examples_dir).exists() else 0
        add("Example recordings", n > 0, f"{n} found in {examples_dir}", required=False)
    return out


def main(argv=None) -> int:
    ex = (argv or sys.argv[1:] or [None])[0]
    results = check_environment(ex)
    worst = 0
    for name, ok, detail, required in results:
        tag = "OK     " if ok else ("MISSING" if required else "missing (optional)")
        print(f"[{tag}] {name}: {detail}")
        if not ok and required:
            worst = 1
    print("\nReady." if worst == 0 else "\nSomething required is missing; the audit cannot run here.")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())