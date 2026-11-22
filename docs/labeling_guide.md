# Labeling guide

Each issue type is anchored to WCAG 2.1 (Understanding docs, Techniques, Failures).
Labels are written by `benchmark/generate.py` into `labels.json`; times are seconds on the
key-log clock (t0 = just after page creation).

| Issue | WCAG | Injected bug | Label timestamp | Implemented |
|---|---|---|---|---|
| Focus not visible | 2.4.7 | `*:focus { outline: none }` on every focusable element, no alternative indicator | `t` = time of each Tab press (one label per press) | yes |
| Flashing | 2.3.1 | banner toggles white/black 10 times per second for 2 s (5 flashes/s) | `t` = onset, `t_end` = end | yes |
| Unannounced change | 4.1.3 | toast without `aria-live` | change onset | planned |
| Uncontrollable motion | 2.2.2 | autoplay carousel, no pause control | motion onset | planned |

## focus_styles: the hard focus cases
`benchmark/testapps/focus_styles` has nine controls, each with one focus style, shuffled per
sample. Every Tab press has an expected verdict, stored in `focus_checks` in `labels.json`.
"ambiguous" means reasonable reviewers could disagree; those presses are not counted as failures,
and the detector is rewarded for escalating them instead of answering confidently.

| Style | What happens on focus | Expected | Reasoning |
|---|---|---|---|
| strong_ring | 4 px orange outline, offset | visible | Unmistakable ring. |
| box_shadow_ring | 4 px blue ring drawn with box-shadow | visible | Looks like a ring; a different CSS mechanism. |
| thin_dotted | 1 px dotted black outline | visible | Judgment call: a thin outline like older browser defaults. Faint at small sizes, so some reviewers might say ambiguous. |
| underline | 3 px underline appears, no outline | visible | A commonly accepted indicator, but not ring-shaped, so a ring test cannot confirm it. |
| bg_change | button fills dark | visible | A strong change, but not an outline. |
| faint_ring | 2 px light-gray ring on white (about 1.4:1) | ambiguous | A ring exists but is barely visible. 2.4.7 names no contrast number; my reading is that 1.4.11 Non-text Contrast sets 3:1 for identifying UI component states (verify before citing). |
| selection_only | text input, outline removed, browser highlights the text | ambiguous | The highlight does show where focus is, but it is not a designed indicator. |
| subtle_color | link color shifts from #1a5bdb to #1e60e0 | not_visible | The change is practically imperceptible. |
| none | outline removed, nothing in its place | not_visible | A clear failure. |

These expected verdicts are the author's judgments, made by the same person who wrote the detector.
Before reporting results, have independent reviewers label the same recordings and compare.

## status_messages: seen versus heard (WCAG 4.1.3)
`benchmark/testapps/status_messages` shows three status messages per sample, in random order: a
toast ("Changes saved"), an error banner ("Upload failed: file too large") and inline text
("3 results found"). Each is either

- announced: inserted into a live region; the sample's `utterances.json` contains a simulated
  screen-reader line starting 0.3-1.2 s later (the banner's line is prefixed "Alert, "), or
- silent: inserted into a plain element; nothing in the transcript. This is the labeled failure.

Each message is triggered either by a timer (no key press) or by pressing Enter on a button. Silent
messages are failures whatever triggered them: 4.1.3 does not exempt messages the user caused. What it
does not cover is content announced through a focus move, so only the focus keys and the calibration
key are excluded from the detector's candidates. Messages auto-dismiss after 3 s, and the banner pushes
the page content down (a layout shift that puts unrelated text inside the changed region).

The transcript is SIMULATED and ERROR-FREE and sits on the key-log clock. It tests the seen-versus-
heard logic and the OCR, not speech recognition or real screen-reader timing. Ground truth is "which
element did the page insert into a live region"; the real accessibility tree is not consulted.

### status_messages_noisy: wrong speech near a silent message
Same page and rules as `status_messages`, but each sample always has one silent message of a random kind, and every
silent message gets two extra transcript lines inside its 3 s announcement window: wrong speech of one category, then
an unrelated screen-reader line. The page never sent any of it, so the message is still a failure.

| Category | Spoken near a silent message | Why it is hard |
|---|---|---|
| near_miss | "Upload complete", "Changes discarded" | different words; should not match |
| wrong_number | "13 results found" for "3 results found" | 89% similar by characters, different information |
| negation | "No results found" for "3 results found" | 88% similar, says the opposite |
| truncated | "Upload failed" for "Upload failed: file too large" | a fragment scores 100% by characters |
| confusable | "Changes saving" for "Changes saved" | 85%: the same score as Whisper's real mishearing "Change is saved" |

The categories were chosen AFTER checking which plausible wrong lines the matcher accepted, so they are the matcher's known
weak spots, not a random sample of what a screen reader might say. Silent messages in this set are therefore far more
likely to be near wrong speech than in real use.

## Calibration cue
Every generated sample begins with four presses of the calibration key (F9), one more than
`Config.sync_min_pairs`, so a single unmatched press does not make the estimate untrusted. Each press shows
a small magenta square in the bottom-right corner for 0.2 s. They give `sync.py` key/effect
pairs even when the bug under test produces no visible effect. These key presses are not labeled
and are not part of the audited workflow: detectors and `check_sample.py` ignore them.
Without such a cue the clock offset cannot be estimated on "nothing visibly happens" bugs.

## Quality gate
The generator checks every recording with the pipeline's own clock-sync estimator. A sample is
kept only if its key log and video can be aligned (at least `sync_min_pairs` consistent pairs,
spread within `sync_max_spread_s`, and an absolute offset of at most 0.25 s). Otherwise it is
re-recorded (default up to 3 attempts) and, failing that, discarded. The outcome is stored under
`qa` in `labels.json`. Reason: in some recordings the video timing drifts against the key log
(seen once in the author's one-core test environment: video 0.8 s shorter than the log), which
would make the ground-truth timestamps wrong. Report the discard/retry rate with the results,
because dropping samples is a selection step. The real-world set cannot be filtered this way.

## Negative controls
- `focus_outline` clean: visible 4 px orange focus ring.
- `flashing_banner` clean: a slow 1 Hz pulse (2 toggles/s). This is a deliberate hard negative:
  it moves, but stays under the three-flashes-per-second threshold.

## Known properties of the generated data (observed, not guaranteed)
- Videos are recorded at 25 fps. In a first test run the video-to-log offset measured about
  0.05-0.08 s on the focus sample and about 0 on the flash sample. The offset is not written to
  labels; `sync.py` has to estimate it.
- An earlier version of the `focus_outline` bug variant still showed the browser's select-all
  highlight when the name input received focus (confirmed on two machines: a change of about
  100x20 px over the "Alex Morgan" text). That arguably is a visible focus cue, which made the
  label for that Tab press debatable. The bug variant now also hides the caret and the selection
  highlight (`caret-color` and `::selection` set to transparent), so no Tab press shows any
  focus cue. The select-all case may return later as a separately labeled "ambiguous" subset.
- The segmenter reports one bounding box per episode, which can span several elements when one
  focus ring disappears and another appears in the same transition. The focus detector should
  split changes into connected components before judging them.

TODO: expand with edge cases as more apps are added.