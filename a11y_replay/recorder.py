"""Key logger for recording sessions: writes keys.jsonl in the format the audit reads.

    python -m a11y_replay.recorder --out recordings/checkout/keys.jsonl

HOW TO USE (details in docs/recording_guide.md):
  1. Start your screen recording with "Record Selected Portion" (NOT "Record Selected Window", which cannot capture the
     square) and a box over the browser, below the menu bar, that includes the spot given by --marker-pos.
  2. Run this. A small square appears at the top-left of the screen and flashes four times at irregular
     intervals. Each flash is logged as an "F9" event. The audit uses these, plus any key press that visibly
     changes the screen, to line the key log up with the video, even if you started the two seconds apart.
  3. Do the session. Press F10 to stop (or Ctrl+C in the terminal).

PRIVACY. A system-wide key logger can see everything typed in every app, passwords included. By default this
records ONLY navigation keys (Tab, Enter, Space, Escape, arrows, Page/Home/End, F-keys) and holds Shift/Ctrl/Alt/Cmd
only to label them (Shift+Tab). Typed characters are never written. `--log-text` turns text logging on and should
be used only when you are certain nothing sensitive is typed.

NOT RUN in the development environment (no display, no keyboard, no pynput). The logging core below is tested with a
fake clock and fake key objects; the pynput listener and the tkinter marker are thin wrappers you must try yourself.
macOS asks for Input Monitoring (and possibly Accessibility) permission for your terminal the first time.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

SPECIAL = {"tab": "Tab", "enter": "Enter", "return": "Enter", "space": "Space", "esc": "Escape",
           "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
           "page_up": "PageUp", "page_down": "PageDown", "home": "Home", "end": "End"}
MODIFIERS = {"shift": "Shift", "shift_l": "Shift", "shift_r": "Shift",
             "ctrl": "Control", "ctrl_l": "Control", "ctrl_r": "Control",
             "alt": "Alt", "alt_l": "Alt", "alt_r": "Alt", "alt_gr": "Alt",
             "cmd": "Meta", "cmd_l": "Meta", "cmd_r": "Meta"}
NAVIGATION = set(SPECIAL.values()) | {f"F{i}" for i in range(1, 21)}


def describe(key):
    """Classify a pynput-style key object: (name, kind) with kind in modifier|navigation|text|other.
    Works on any object with a `.name` (special keys) or `.char` (characters)."""
    name = getattr(key, "name", None)
    if name:
        if name in MODIFIERS:
            return MODIFIERS[name], "modifier"
        n = SPECIAL.get(name) or (name.upper() if re.fullmatch(r"f\d{1,2}", name) else name)
        return n, ("navigation" if n in NAVIGATION else "other")
    ch = getattr(key, "char", None)
    return (ch, "text") if ch else ("Unknown", "other")


class KeyLogger:
    """Appends one JSON line per logged key press. Times are seconds since this object was created."""

    def __init__(self, out_path, clock=time.monotonic, log_text: bool = False, stop_key: str = "F10"):
        self.clock, self.log_text, self.stop_key = clock, log_text, stop_key
        self.t0 = clock()
        self.held: set = set()
        self.stopped = False
        self.count = 0          # every line written
        self.markers = 0        # of which calibration flashes (not real key presses)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(out_path, "w", encoding="utf-8", buffering=1)

    def now(self) -> float:
        return self.clock() - self.t0

    def _write(self, name: str, modifiers: list, **extra) -> None:
        self._f.write(json.dumps({"t": round(self.now(), 3), "key": name, "modifiers": modifiers, **extra}) + "\n")
        self.count += 1

    def on_press(self, key):
        name, kind = describe(key)
        if kind == "modifier":
            self.held.add(name)
            return None
        if name == self.stop_key:
            self.close()
            return False                       # tells pynput's listener to stop
        if kind == "navigation" or (kind == "text" and self.log_text):
            self._write(name, sorted(self.held))
        return None                            # typed characters are dropped here and never stored

    def on_release(self, key):
        name, kind = describe(key)
        if kind == "modifier":
            self.held.discard(name)

    @property
    def presses(self) -> int:
        """Real key presses captured (calibration flashes excluded)."""
        return self.count - self.markers

    def log_marker(self) -> None:
        """Record a calibration flash as an F9 event (source: marker)."""
        self.markers += 1
        self._write("F9", [], source="marker")

    def close(self) -> None:
        self.stopped = True
        if not self._f.closed:
            self._f.close()


class QuietTerminal:
    """While logging, turn off the terminal's echo and, on exit, throw away anything typed into it.

    Without this, keys pressed while the terminal is in front are echoed as junk ("^[[A") and stay in the
    input queue, so when the logger ends the SHELL receives them: type a command and press Enter during a
    session and it runs afterwards. Ctrl+C still works (signals are left on). `fd` is a test hook.
    """

    def __init__(self, fd=None):
        self.fd = fd if fd is not None else (sys.stdin.fileno() if sys.stdin and sys.stdin.isatty() else None)
        self.saved = None

    def __enter__(self):
        if self.fd is not None:
            try:
                import termios
                self.saved = termios.tcgetattr(self.fd)
                new = termios.tcgetattr(self.fd)
                new[3] &= ~(termios.ECHO | termios.ICANON)
                termios.tcsetattr(self.fd, termios.TCSADRAIN, new)
            except Exception:
                self.saved = None            # no termios (Windows) or not a terminal: carry on without
        return self

    def __exit__(self, *exc):
        if self.fd is not None and self.saved is not None:
            import termios
            termios.tcflush(self.fd, termios.TCIFLUSH)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False


def check_permissions(accessibility=None, listen=None) -> dict:
    """Can this process watch the keyboard? macOS needs BOTH Accessibility and Input Monitoring for the
    terminal (or editor) running this. Returns {"accessibility": bool|None, "input_monitoring": bool|None};
    None means the check itself was not possible. The two arguments are test hooks."""
    out = {"accessibility": None, "input_monitoring": None}
    try:
        if accessibility is None:
            from ApplicationServices import AXIsProcessTrusted as accessibility
        out["accessibility"] = bool(accessibility())
    except Exception:
        pass
    try:
        if listen is None:
            from Quartz import CGPreflightListenEventAccess as listen
        out["input_monitoring"] = bool(listen())
    except Exception:
        pass
    return out


PERMISSION_HELP = (
    "macOS has NOT granted this terminal permission to watch the keyboard, so key presses made while another app "
    "(your browser) is in front may not be logged. Open System Settings > Privacy & Security, add your terminal "
    "(or VS Code) under BOTH Accessibility and Input Monitoring, turn them on, then QUIT and reopen the terminal.")


def marker_schedule(n: int = 4, rng: random.Random | None = None, first: float = 1.5) -> list:
    """Times (seconds after start) for the calibration flashes. The gaps are irregular on purpose: evenly
    spaced flashes would match the video at several offsets and the alignment would be ambiguous."""
    rng = rng or random.Random()
    t, out = first, []
    for _ in range(n):
        out.append(round(t, 3))
        t += rng.uniform(0.7, 1.6)
    return out


def _marker_pos(text: str):
    """argparse type for --marker-pos: two whole numbers, checked BEFORE anything starts logging."""
    try:
        x, y = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be two whole numbers separated by a comma, like 20,300 (you passed {text!r}). "
            "Replace placeholders such as X,Y with real screen coordinates.")
    return x, y


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="keys.jsonl")
    ap.add_argument("--log-text", action="store_true", help="ALSO log typed characters (privacy risk; see above)")
    ap.add_argument("--no-marker", action="store_true", help="no calibration square; alignment then relies on key presses that change the screen")
    ap.add_argument("--marker-flashes", type=int, default=4)
    ap.add_argument("--marker-size", type=int, default=120, help="square size in screen points")
    ap.add_argument("--marker-pos", type=_marker_pos, default=(10, 60),
                    help="x,y of the square's top-left corner in screen points from the screen's top-left, e.g. 20,300. "
                         "Keep it inside the area you are screen-recording and on an empty part of the page")
    ap.add_argument("--stop-key", default="F10")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)

    try:
        from pynput import keyboard
    except ImportError:
        raise SystemExit("pip install pynput   (macOS will ask for Input Monitoring permission)")
    perms = check_permissions()
    if perms["accessibility"] is False or perms["input_monitoring"] is False:
        print(f"WARNING: {PERMISSION_HELP}\n  (accessibility={perms['accessibility']}, input monitoring={perms['input_monitoring']})")
    logger = KeyLogger(a.out, log_text=a.log_text, stop_key=a.stop_key)
    listener = keyboard.Listener(on_press=logger.on_press, on_release=logger.on_release)
    listener.start()
    print(f"Logging to {a.out}. Navigation keys only{' + TEXT (as requested)' if a.log_text else ''}. "
          f"Press {a.stop_key} (or Ctrl+C here) to stop. Make sure your screen recording is running and that the small calibration square will be INSIDE the recorded box.")

    try:
      with QuietTerminal():
        if a.no_marker:
            while not logger.stopped:
                time.sleep(0.1)
        else:
            try:
                import tkinter as tk
            except ImportError:
                logger.close()
                raise SystemExit("tkinter is missing (macOS Homebrew Python: brew install python-tk), or use --no-marker.")
            x, y = a.marker_pos
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.geometry(f"{a.marker_size}x{a.marker_size}+{x}+{y}")
            root.configure(bg="#202020")

            def flash():
                root.configure(bg="#d81b60")
                logger.log_marker()
                root.after(200, lambda: root.configure(bg="#202020"))

            for t in marker_schedule(a.marker_flashes):
                root.after(int(t * 1000), flash)

            def poll():
                root.destroy() if logger.stopped else root.after(100, poll)

            root.after(100, poll)
            root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        logger.close()
    print(f"\nDone. Wrote {logger.presses} key press(es) and {logger.markers} calibration flash(es) to {a.out}.")
    if logger.presses == 0:
        print("WARNING: no key presses were captured. " + PERMISSION_HELP)


if __name__ == "__main__":
    main()