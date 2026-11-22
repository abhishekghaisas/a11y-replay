"""Issue types and their WCAG 2.1 criteria. Dependency-free so the benchmark
generator and evaluation code can import it without pulling in the full pipeline."""
from enum import Enum


class IssueType(str, Enum):
    FOCUS_NOT_VISIBLE = "focus_not_visible"          # WCAG 2.4.7
    UNANNOUNCED_CHANGE = "unannounced_change"        # WCAG 4.1.3
    FLASHING = "flashing"                            # WCAG 2.3.1
    UNCONTROLLABLE_MOTION = "uncontrollable_motion"  # WCAG 2.2.2
    KEYBOARD_TRAP = "keyboard_trap"                  # WCAG 2.1.2 (stretch)


WCAG_CRITERION = {
    IssueType.FOCUS_NOT_VISIBLE: "2.4.7",
    IssueType.UNANNOUNCED_CHANGE: "4.1.3",
    IssueType.FLASHING: "2.3.1",
    IssueType.UNCONTROLLABLE_MOTION: "2.2.2",
    IssueType.KEYBOARD_TRAP: "2.1.2",
}