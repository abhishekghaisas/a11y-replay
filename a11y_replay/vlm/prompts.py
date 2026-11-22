"""Narrow, closed-question prompts. Versioned so evaluation runs record which prompt produced them.

The prompt must not leak the benchmark: no style names, no hint of the expected answer. It describes
indicators generically (that is the task definition), which inevitably overlaps in kind with some benchmark
styles; say so when reporting results.
"""
PROMPT_VERSION = "focus-judge-v1"

FOCUS_JUDGE_INTRO = (
    "You are checking one moment of a screen recording for a web accessibility audit (WCAG 2.4.7, "
    "Focus Visible). A keyboard user just pressed the {key} key, which moved keyboard focus to a new "
    "element. Below are two crops of the same screen region: BEFORE the key press and AFTER it."
)

FOCUS_JUDGE_QUESTION = """Question: does the element that NEWLY RECEIVED keyboard focus now show a visible focus indicator?

Rules:
- Judge only the element that newly received focus. The element that lost focus may lose its own indicator; ignore that.
- Any clearly visible change counts as an indicator (for example an outline, or a color, background or text-decoration change), as long as a sighted keyboard user would reasonably notice it and it distinguishes the focused element.
- "no" means nothing perceptible changed on the newly focused element.
- "unsure" means you cannot tell, or a reasonable reviewer could disagree.

Answer with ONLY a JSON object, no other text:
{"indicator_visible": "yes" | "no" | "unsure", "indicator_type": "<few words, or empty>", "evidence": "<at most 25 words>"}"""

RETRY_REMINDER = "Reply with ONLY the JSON object described above. No other text."