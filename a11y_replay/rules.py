"""Combine detector output into findings with severity, confidence, and WCAG mapping."""
from a11y_replay.schemas import WCAG_CRITERION, IssueType


def criterion_for(issue: IssueType) -> str:
    return WCAG_CRITERION[issue]
