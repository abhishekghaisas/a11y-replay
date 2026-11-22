"""WCAG 4.1.3 Status Messages: seen versus heard (classical layer: segmentation + OCR + transcript).

For every visual change that is not just a focus move or the calibration cue:
  1. take a frame before and after, crop the changed region, OCR both;
  2. the NEW words are the ones in `after` that were not in `before` (a banner that pushes other
     text around must not make that text look like the message);
  3. if there are enough new characters it is a status message that APPEARED; if only words
     disappeared it is a message being dismissed (ignored); if the text did not change at all it is a
     non-text change (an icon, a color) and is reported as needs_review for a model, never as a finding;
  4. compare the new text with the transcript: an utterance whose text is similar and whose start lies
     within [t - announce_pre_s, t + announce_window_s] means it was announced; otherwise it was not.

Everything is on the key-log clock. Video times are mapped with the sync offset; the transcript is
assumed to be on the log clock too (a real audio pipeline needs its own audio offset).

What is NOT excluded: changes caused by Enter/Space or any key outside the focus keys. WCAG 4.1.3
covers status messages whether or not the user triggered them; what it excludes is content announced
through a focus move, which the transcript would contain anyway.

Known limits: OCR can misread text; the appeared/disappeared test for non-text changes is the one used
for focus indicators and can be unclear after layout shifts (those stay needs_review); messages without text (icons, colors) are not judged here; one
transcript line can announce only one change; a message that stays on screen without changing is
never re-examined; thresholds are untuned; the benchmark transcript is simulated and error-free.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from a11y_replay.config import DEFAULT, Config
from a11y_replay.detectors.flash import relative_luminance
from a11y_replay.detectors.focus import _stands_out
from a11y_replay.events import Finding, KeyEvent, Utterance
from a11y_replay.ingest.video import read_frame_at
from a11y_replay.segment import detect_changes
from a11y_replay.taxonomy import WCAG_CRITERION, IssueType


@dataclass(frozen=True)
class AnnounceResult:
    t: float                      # onset, key-log clock
    bbox: tuple                   # changed region (x, y, w, h) in the video frame
    verdict: str                  # "announced" | "unannounced" | "needs_review"
    text: str                     # the new on-screen text (empty for needs_review)
    matched_text: str | None      # the utterance that announced it
    ratio: float                  # similarity of the best candidate utterance (0-100)
    nearest_text: str | None      # best candidate even when below the threshold, for the report
    reasoning: str


# ------------------------------------------------------------------ text helpers
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "x x twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell_number(n: int) -> str:
    """0-999 in English words, so on-screen digits compare equal to a spoken or transcribed number."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + spell_number(n % 100))


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, and spell out numbers up to 999 ("3 results" -> "three results")."""
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    return re.sub(r"\b\d{1,3}\b", lambda m: spell_number(int(m.group())), s)


_VALUES = {w: i for i, w in enumerate(_ONES)}
_VALUES.update({w: 10 * i for i, w in enumerate(_TENS) if w != "x"})
_NEGATION = re.compile(r"\b(no|not|never|cannot|without|none)\b|n't\b", re.I)


def numbers_in(text: str) -> list:
    """The numbers a text contains, as integers, whether written as digits or words ("3", "three", "twenty one",
    "a hundred"). Digits are spelled out by normalize(), then runs of number words are put back together."""
    out, cur = [], None
    for w in normalize(text).split():
        if w == "hundred":
            if cur is None:
                cur = 100
            elif cur < 10:
                cur *= 100
            else:
                out.append(cur)
                cur = 100
        elif w in _VALUES:
            v = _VALUES[w]
            joinable = cur is not None and ((cur >= 100 and cur % 100 == 0 and v > 0) or
                                            (20 <= cur % 100 <= 90 and cur % 10 == 0 and 0 < v < 10))
            if joinable:
                cur += v
            else:
                if cur is not None:
                    out.append(cur)
                cur = v
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def content_conflict(screen: str, spoken: str):
    """Why a similar-looking utterance still does not announce this message, or None.

    (1) every number on screen must be spoken: "13 results" or a missing count does not announce "3 results";
    (2) negation must agree: "No results found" does not announce "3 results found".
    Known costs: a recognizer that garbles a number ("tree results") is rejected like a wrong number, and a
    spoken prefix without these markers still counts as the whole message.
    """
    need, have = list(numbers_in(screen)), list(numbers_in(spoken))
    for n in need:
        if n in have:
            have.remove(n)
        else:
            return f"the number {n} on screen was not spoken"
    if bool(_NEGATION.search(screen)) != bool(_NEGATION.search(spoken)):
        return "negation differs between the screen and the speech"
    return None


def word_coverage(screen: str, spoken: str, word_match: float = 75.0) -> float:
    """Share of the on-screen message's words that were spoken (each spoken word used once), allowing small
    spelling differences per word ("changes" ~ "change", "too" ~ "to") but not a different word ("saved" vs
    "saving"). 1.0 means everything was said; extra spoken words (a role prefix) do not matter."""
    want, said = normalize(screen).split(), normalize(spoken).split()
    if not want:
        return 1.0
    free, hit = list(range(len(said))), 0
    for w in want:
        scored = [(100.0 * SequenceMatcher(None, w, said[i]).ratio(), i) for i in free]
        ok = [x for x in scored if x[0] >= word_match]
        if ok:
            free.remove(max(ok)[1])
            hit += 1
    return hit / len(want)


def similarity(a: str, b: str) -> float:
    """0-100. The shorter text is slid along the longer one, so a spoken "alert, <message>" still
    matches the on-screen message, and on-screen clutter around a message does not hurt."""
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    if len(a) == len(b):
        return 100.0 * SequenceMatcher(None, a, b).ratio()
    return 100.0 * max(SequenceMatcher(None, a, b[i:i + len(a)]).ratio() for i in range(len(b) - len(a) + 1))


def _tokens(s: str) -> list:
    return re.findall(r"[A-Za-z0-9]+", s or "")


def new_words(before_text: str, after_text: str) -> str:
    """Words present in `after_text` that are not accounted for in `before_text`, in order.

    Words are compared normalized (so "3" and "three" are the same word) but returned as written on
    screen, so a report says "3 results found" and not "three results found".
    """
    have = Counter(normalize(w) for w in _tokens(before_text))
    out = []
    for w in _tokens(after_text):
        key = normalize(w)
        if have[key] > 0:
            have[key] -= 1
        else:
            out.append(w)
    return " ".join(out)


def removed_words(before_text: str, after_text: str) -> str:
    return new_words(after_text, before_text)


def ocr_region(frame: np.ndarray, bbox: tuple, cfg: Config = DEFAULT, pad: int = 12) -> str:
    """OCR a padded crop of `bbox`. Needs the Tesseract program (brew install tesseract)."""
    try:
        import pytesseract
    except ImportError as e:
        raise RuntimeError("pip install pytesseract (and install Tesseract: brew install tesseract)") from e
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    crop = frame[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=cfg.announce_ocr_scale, fy=cfg.announce_ocr_scale, interpolation=cv2.INTER_CUBIC)
    if g.mean() < 110:                 # light text on a dark background: Tesseract prefers dark on light
        g = 255 - g
    try:
        return pytesseract.image_to_string(g, config="--psm 6").strip()
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError("Tesseract is not installed: brew install tesseract") from e


def _non_text_direction(before: np.ndarray, after: np.ndarray, bbox: tuple, cfg: Config) -> str:
    """Did a non-text change appear or disappear? Same idea as the focus detector: compare how much
    the changed pixels stand out from a band of surrounding pixels, before and after."""
    x, y, w, h = bbox
    H, W = before.shape[:2]
    g0 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    g1 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    changed = cv2.absdiff(g0, g1) > cfg.pixel_diff_threshold
    inside = np.zeros((H, W), bool); inside[y:y + h, x:x + w] = True
    region = changed & inside
    outer = np.zeros((H, W), bool)
    outer[max(0, y - 14):min(H, y + h + 14), max(0, x - 14):min(W, x + w + 14)] = True
    inner = np.zeros((H, W), bool)
    inner[max(0, y - 4):min(H, y + h + 4), max(0, x - 4):min(W, x + w + 4)] = True
    band = outer & ~inner
    if region.sum() == 0 or band.sum() < 20:
        return "unclear"
    l0, l1 = relative_luminance(before), relative_luminance(after)
    d_before = _stands_out(l0, region, float(np.median(l0[band])))
    d_after = _stands_out(l1, region, float(np.median(l1[band])))
    if d_after - d_before >= cfg.focus_direction_delta:
        return "appeared"
    if d_before - d_after >= cfg.focus_direction_delta:
        return "disappeared"
    return "unclear"


# ------------------------------------------------------------------ detector
def _explained_by_key(t: float, keys: Iterable[KeyEvent], cfg: Config) -> bool:
    """A change right after a focus key (or the calibration key) is a focus move, not a status message."""
    skip = set(cfg.focus_keys) | {cfg.calibration_key}
    return any(k.key in skip and k.t - cfg.effect_before_s <= t <= k.t + cfg.effect_after_s for k in keys)


@dataclass(frozen=True)
class Observation:
    """One candidate status message seen on screen, before any transcript is consulted."""
    t: float             # onset, key-log clock
    bbox: tuple
    kind: str            # "message" (new text appeared) or "non_text" (something changed, no new text)
    text: str            # the new on-screen words (empty for non_text)


def observe_changes(video_path: str | Path, keys: Iterable[KeyEvent], offset: float = 0.0,
                    cfg: Config = DEFAULT) -> list[Observation]:
    """Segmentation + OCR + word comparison. The expensive part; independent of the transcript."""
    keys = list(keys)
    out: list[Observation] = []
    for ch in detect_changes(video_path, cfg):
        t = ch.t - offset
        if t < cfg.announce_ignore_start_s or _explained_by_key(t, keys, cfg):
            continue
        before = read_frame_at(video_path, max(0.0, ch.t - 0.2))
        after = read_frame_at(video_path, ch.t_end + 0.3)
        if before is None or after is None or before.shape != after.shape:
            continue
        before_text, after_text = ocr_region(before, ch.bbox, cfg), ocr_region(after, ch.bbox, cfg)
        added = new_words(before_text, after_text)
        if len(added.replace(" ", "")) >= cfg.announce_min_chars:
            out.append(Observation(round(t, 3), ch.bbox, "message", added))
        elif len(removed_words(before_text, after_text).replace(" ", "")) >= cfg.announce_min_chars:
            continue  # a message disappearing (auto-dismiss), not a new one
        elif _non_text_direction(before, after, ch.bbox, cfg) == "disappeared":
            continue  # a non-text change going away is not a new status message either
        else:
            out.append(Observation(round(t, 3), ch.bbox, "non_text", ""))
    return out


def match_observations(observations: Iterable[Observation], utterances: Iterable[Utterance],
                       audio_offset: float = 0.0, cfg: Config = DEFAULT) -> list[AnnounceResult]:
    """Compare what was seen with what was heard. Cheap; rerun freely with different transcripts."""
    utts = sorted(utterances, key=lambda u: u.start)
    used: set = set()
    results: list[AnnounceResult] = []
    for ob in observations:
        if ob.kind == "non_text":
            results.append(AnnounceResult(
                t=ob.t, bbox=ob.bbox, verdict="needs_review", text="", matched_text=None, ratio=0.0,
                nearest_text=None,
                reasoning="Something changed on screen but no new text was read from it (icon, color or "
                          "layout change?). A semantic check is needed to say whether it is a status message."))
            continue
        t, added = ob.t, ob.text
        # candidate utterances: start within [t - pre, t + window] on the log clock
        cands = [u for u in utts if id(u) not in used
                 and t - cfg.announce_pre_s <= u.start - audio_offset <= t + cfg.announce_window_s]
        best, best_ratio, rejected = None, 0.0, None
        for u in cands:
            r = similarity(added, u.text)
            why = content_conflict(added, u.text) if cfg.announce_content_guards else None
            if why:
                if r >= cfg.text_match_ratio and (rejected is None or r > rejected[1]):
                    rejected = (u, r, why)       # would have matched on characters alone
                continue
            if r > best_ratio:
                best, best_ratio = u, r
        cov = word_coverage(added, best.text, cfg.announce_word_match) if best is not None else 1.0
        if best is not None and best_ratio >= cfg.text_match_ratio and cfg.announce_content_guards \
                and cov < cfg.announce_min_coverage:
            used.add(id(best))
            results.append(AnnounceResult(
                t=t, bbox=ob.bbox, verdict="needs_review", text=added, matched_text=best.text,
                ratio=round(best_ratio, 1), nearest_text=best.text,
                reasoning=f'Only part of the message was spoken: "{best.text}" covers {cov:.0%} of the words in '
                          f'"{added}" (needs {cfg.announce_min_coverage:.0%}). A person should decide whether '
                          f"that counts as announced."))
        elif best is not None and best_ratio >= cfg.text_match_ratio:
            used.add(id(best))
            results.append(AnnounceResult(
                t=t, bbox=ob.bbox, verdict="announced", text=added, matched_text=best.text,
                ratio=round(best_ratio, 1), nearest_text=best.text,
                reasoning=f'New text "{added}" was spoken as "{best.text}" (similarity {best_ratio:.0f}).'))
        else:
            if rejected is not None:
                near = (f' An utterance in the window, "{rejected[0].text}", looked similar ({rejected[1]:.0f}) but was '
                        f"rejected: {rejected[2]}.")
            elif best:
                near = (f' The closest utterance in the window was "{best.text}" (similarity {best_ratio:.0f}, '
                        f'needs {cfg.text_match_ratio:.0f}).')
            else:
                near = " Nothing was spoken in the window."
            results.append(AnnounceResult(
                t=t, bbox=ob.bbox, verdict="unannounced", text=added, matched_text=None,
                ratio=round(best_ratio, 1), nearest_text=(rejected[0].text if rejected else best.text if best else None),
                reasoning=f'New text "{added}" appeared on screen at {t:.1f}s and was not announced '
                          f"within {cfg.announce_window_s:.0f}s." + near))
    return results


def detect_announcements(
    video_path: str | Path, keys: Iterable[KeyEvent], utterances: Iterable[Utterance],
    offset: float = 0.0, audio_offset: float = 0.0, cfg: Config = DEFAULT,
) -> list[AnnounceResult]:
    """offset = video_time - log_time; audio_offset = audio_time - log_time (see sync.py)."""
    return match_observations(observe_changes(video_path, keys, offset, cfg), utterances, audio_offset, cfg)


def to_findings(results: Iterable[AnnounceResult], cfg: Config = DEFAULT) -> list[Finding]:
    return [
        Finding(
            issue_type=IssueType.UNANNOUNCED_CHANGE, criterion=WCAG_CRITERION[IssueType.UNANNOUNCED_CHANGE],
            start=r.t, end=round(r.t + cfg.announce_window_s, 3), confidence=None, method="classical",
            reasoning=r.reasoning, evidence_frames=(),
        )
        for r in results if r.verdict == "unannounced"
    ]