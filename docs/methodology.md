# Methodology

TODO: cascade design, thresholds and how they were tuned (dev split only), sync estimation,
metrics, and limitations. Keep in sync with README.

## Flash detector (WCAG 2.3.1)

Implemented in `a11y_replay/detectors/flash.py`, fully classical (no VLM).

- Per-pixel relative luminance on frames downscaled to 160 px wide.
- A transition is a valley-to-peak swing of at least 0.10 relative luminance where the darker side
  is below 0.80; a flash is two opposing transitions; more than three flashes in any 1 s window
  makes a pixel "flashing".
- A finding needs flashing pixels to cover at least 25% of a 341x256 block, expressed as a
  fraction of the 1024x768 reference screen (about 2.8%). The W3C material is ambiguous about
  whether the limit is that block or 25% of it; this is the conservative reading and is a config
  value (`flash_min_area_frac`).

Not implemented: red flash; the 50 Hz upper bound; contiguity of the flashing area. At 25 fps,
flashes faster than 12.5 Hz cannot be observed. Frames are averaged in sRGB when downscaled.
This is a screening aid, not a compliance test.

Boundary behavior: a flash is counted as two transitions (`transitions // 2`), so a 1 s window with
6-7 transitions counts as 3 flashes (passes) and 8-9 as 4 (flagged). At 25 fps, a toggle every 4
frames (about 3.1 flashes/s) therefore passes and every 3 frames (about 4.2 flashes/s) is flagged.
A stricter reading that flags any window with more than 6 transitions would flag the 4-frame case.

First check on generated samples (`python -m eval.eval_flash`): 3/3 labeled flashes found, 0 false
positives on 12 samples, onset 0.08-0.12 s after the label. The onset lag is inherent to the test
app (its first toggle fires one 100 ms interval after the label time). These samples are easy
(full-width black/white flashing) and synthetic, so this is a smoke test, not accuracy evidence.


## Focus detector (WCAG 2.4.7)

Implemented in `a11y_replay/detectors/focus.py`. For each Tab press it takes a frame 0.2 s before
and 0.3 s after (on the video clock, using the sync offset) and finds the changed regions.
Each region is described by shape (a ring-like outline with an unchanged interior, or solid) and
direction (appeared or disappeared, judged by comparing the region's most extreme pixels, the 10th and
90th percentile, with a ring of surrounding pixels in each frame; the median fails on thin or dotted
outlines, which are mostly light halo after blurring). Only regions that appeared can be an indicator on the newly focused element,
because the previous element's indicator disappearing also changes the screen.

Verdicts: `visible` (a ring appeared with at least 1.5:1 contrast, measured per pixel as the 90th percentile), `not_visible` (nothing
appeared), `ambiguous` (something appeared that is not a clear ring, or direction is unclear).
Ambiguous results are escalated, never reported as findings by the classical layer.

Known limits: touching old and new rings merge into one region of unclear direction (escalated);
indicators fainter than the 25-gray-level diff threshold read as not visible; all thresholds are
untuned. The classifier was changed repeatedly after seeing failures on the generated samples:
per-pixel contrast (rings moving between neighbors cancelled out), removing a morphological close
(it merged neighboring rings), the appear/disappear test (a vanishing indicator looked like a new
one), the ring-interior threshold (0.15 to 0.30), extreme-pixel direction and contrast (thin
dotted outlines are mostly light halo), and ring-ness judged by the LARGEST single interior hole rather
than the total (on macOS the text highlight keeps dark glyph strokes, whose many small holes summed to
30-35% and passed as a ring; found by a user's run, not by my own samples). Each fix has a regression test, but the samples were used
for development, so scores on them (80 Tab presses, 30% escalated, 0 confident errors) are a
development check, not an unbiased estimate. Fresh recordings with new random orders were used
after the last two changes, but they use the same nine styles.

## VLM escalation (cascade)

Model: Claude Sonnet 5.5 (`claude-sonnet-5-5`) is the default judge, chosen because the escalated
residue is small (about a quarter of focus presses), so at this scale accuracy matters more than the
per-call price. Haiku 5.5 (cheaper), Opus 5.5 (accuracy ceiling) and a local open model (Qwen3-VL via
Ollama) are comparison arms. The recommendation came from the model itself, which is a conflict of
interest, so it is treated as a hypothesis to be confirmed by `eval/eval_vlm.py`.

What is sent: before/after crops of the union of all changed regions (60 px padding, downscaled to at
most 900 px; Sonnet 5.5 bills images at a high-resolution tier) and one closed question: does the
element that newly received focus now show a visible indicator? Answer: yes / no / unsure as JSON. Only
`yes -> visible` and `no -> not_visible` resolve a press; `unsure` stays unresolved and is never a
finding. A reply that is not valid JSON is retried once, then treated as unsure.

Request settings for Sonnet 5.5 (per Anthropic's docs as read on 2026-11-10; the live API was never
called during development, only a fake SDK): thinking cannot be disabled, so `between_tools` turns off
up-front thinking; effort is `medium`; no temperature/top_p/top_k and no forced tool use, all of which
return 400 errors on this model. Effort and thinking mode are untuned starting values; the docs advise
re-running an effort sweep (`--effort`, `--thinking adaptive`).

Repeatability: these settings cannot make the model deterministic, so responses are cached by model
settings, prompt version and image content (`eval/results/vlm_cache`). Record the model ID, prompt
version and date with any reported result.

Decision rule, fixed before any real results: choose the cheapest model that makes no confident
mistake on presses expected to be failures and agrees with "visible" on at least 90% of the presses
expected to be visible; switch the default to Opus 5.5 if it beats the default by more than a press or
two. Known limits: the benchmark styles and expected verdicts were written by the detector's author;
the prompt describes indicators generically, which overlaps in kind with some styles; the labeled set
is small; findings decided by the model carry no calibrated confidence (`confidence=None`).

Privacy: crops of the screen recording leave the machine when a hosted model is used. Fine for the
synthetic benchmark; real recordings may need consent, which is a reason to keep a local model arm.


## Seen versus heard (WCAG 4.1.3)

Implemented in `a11y_replay/detectors/announce.py`, split in two so transcripts can be swapped without
redoing the expensive part. `observe_changes`: segment the video into changes, drop those within the
effect window of a focus key or the calibration key and anything in the first second, OCR a crop
before and after, and take the NEW words (words in `after` not accounted for in `before`, so text that
merely moved when a banner was inserted is ignored). New text means a message appeared; only removed
text means a dismissal (ignored); no text change means a non-text change, kept as `needs_review` for a
model and never a finding. `match_observations`: a message is announced if an unused utterance starts
within [t - 0.2 s, t + 3 s] and is at least 80% similar (the shorter text slid along the longer one,
using difflib; one utterance announces at most one message). Otherwise it is a finding.

Stress tests (`python -m eval.eval_announce ... --sweep-noise`, 3 samples, 9 messages, 8 seeds): with
an error-free transcript every message is judged correctly. Corrupting letters in the transcript, a crude
stand-in for recognition errors, leaves recall at 1.00 but precision falls to about 0.97 at 5% corrupted
letters, 0.79 at 15% and 0.51 at 30%, because a garbled announcement no longer matches and an announced
message is reported as silent. Delaying every utterance by 2.5 s or more drops precision to 0.33 with
the 3 s window. Recall cannot fall here by construction: a silent message has no utterance to match,
so the harder test (a silent message next to an unrelated or similar utterance) is not yet in the
benchmark. The 80% threshold, the 3 s window and the noise model are all untuned or invented; real
recognition errors are phonetic and word-level, not random letters.


### Real-audio tier for seen-versus-heard

`python -m benchmark.tts benchmark/data/*` (macOS only: `say` + ffmpeg) speaks each line of
`utterances.json` into one `audio.wav`, placed at its start time on the key-log clock.
`python -m eval.eval_announce benchmark/data/* --transcript asr` transcribes it with faster-whisper
(`a11y_replay/ingest/audio.py`, cached in `transcript_asr.json` per model size) and scores the same
detector, also reporting recognition quality against the true lines: word error rate, start-time error,
missed lines and invented lines (recognizers can hallucinate text on silence).

Status: the plumbing is tested here with stand-ins for `say` and Whisper (the development environment
has neither). A hand-made transcript with typical recognizer habits (punctuation, "three" for "3", a
200 ms start delay, one invented line) was scored correctly, which shows the route works and says
nothing about Whisper itself. Whether Whisper's own timestamps, number formatting and errors break the
3 s window and 80% match is exactly what the first real run will show.

Limits to state when reporting: one synthetic macOS voice reading clean text, no background noise, no
overlapping speech, audio offset 0 by construction. This is an optimistic case compared with real NVDA
or VoiceOver recordings. Numbers up to 999 are spelled out before comparing, so a screen reader saying
"three" and OCR reading "3" match; larger numbers are not normalized.

First real run (faster-whisper `base`, one macOS voice, 3 samples, 6 spoken lines, 3 silent messages).
Recall stayed 1.00 (a silent message has nothing to match, by construction) but one announced banner was
reported as silent: precision 0.75. Cause, read from `--show-asr`: Whisper joined two lines spoken
11 s apart into ONE segment stamped with the first line's start time (so the banner text existed but
was filed 11 s early), and split another line into two segments 2 s apart that the audio does not
contain. That also explains the word error rate above 1.0 (a 9-word segment scored against a 2-word
line) and the "invented" line (the second half of a split banner). Wording errors were small
("Change is saved" for "Changes saved", "File to large" for "file too large") and start times within
a line were within 0.22 s. The simulated letter-noise sweep above modeled the wrong failure: real
errors here were lines merged or split, not typos, so that sweep should not be cited as a prediction.

Fix: the audio is cut at real silences (>= 0.8 s) before recognition and each burst is transcribed on
its own, one burst = one utterance (`split_on_silence` in `a11y_replay/ingest/audio.py`). Tests cover the
11 s case, short pauses inside a line, clicks, faint hiss, and a track with no pauses (an earlier version
of the chunker returned nothing for that, caught by a unit test). Limits: energy-based, so it suits clean
audio; recordings with music or heavy background noise need a real voice-activity detector.

Re-run after the fix (same 3 samples, faster-whisper `base`): 6 lines found, 0 missed, 0 invented; start-time
error mean +0.01 s, worst +0.04 s; word error rate mean 0.42 (worst 1.00); all 9 messages judged correctly
(precision 1.00, recall 1.00). Wording errors persisted and were systematic: "Changes saved" came back as
"Change is saved" in all three samples (similarity 84.6 against the 80 threshold, a margin of under 5
points) and "file too large" as "file to large" (banner similarity 96.4). So the clean result rests on a
thin margin for short messages: one more wrong letter would have failed it. The 80% threshold has not been
tuned and cannot be from this data, because the benchmark has no negative pairs (a silent message near
unrelated speech) to set it against. Six lines from one synthetic voice: a best case, not an accuracy rate.


## Audit pipeline (`a11y_replay/audit.py`, `report.py`)

`run_audit` runs the checks on one recording folder and writes `report.json`, `report.html` and
`evidence/*.png`. It uses only the video, `keys.jsonl`, and a transcript (`utterances.json`, or recognized
`audio.wav`). It never reads `labels.json` or the generator's stored sync estimate; the clock offset is estimated
again from the recording, and a test with a deliberately false `labels.json` checks that nothing changes. Checks
that lack their inputs are skipped and the report says why; the two unbuilt criteria (2.2.2, 2.1.2) are listed in
every report. A flashing episode is also "a change with no readable text", so unlabeled-change review items that
coincide with a reported flash are dropped to avoid listing it twice. The HTML report is self-contained, uses
semantic headings, a table with header cells, alt text, words (not color alone) for status, and light/dark schemes.
The Gradio UI (`app.py`) is a thin wrapper over `audit_uploads` and has not been run in the development environment.


## Variable frame rate (found on the second real recording)

macOS screen recordings write a frame only when the screen changes, so a still page produces very few frames and the file
has a variable frame rate. Every module converts frame numbers to seconds as number / fps, which is only valid for a constant
rate. On the `focus_clean` recording the symptoms were: zero key/change pairs for clock sync, only 1 of 6 Tab presses analyzed
(the others fell "outside the video"), and the one analyzed press compared two frames that both already showed the focus ring,
reported as a false "focus not visible" on a page that should pass. The key log (6 Tabs, 4 calibration flashes, last event at
36.0 s) was fine; the video was reported as 34.8 s long. Reproduced with a synthetic variable-rate copy of a recording: 10 frames
in a 15 s recording, "reported fps 1.59", and the Tab press outside the video.

Fix: the audit detects variable frame rate (ffprobe's nominal versus average rate, or a gap scan of frame timestamps when ffprobe
is missing) and analyzes a constant 25 fps copy made with ffmpeg (frames repeated through still stretches, scaled to at most
1280 px wide). The copy's length comes from the decoder's real frame timestamps, NOT from the container's duration field, which
was wrong (6.3 s for a 14.7 s recording) in the synthetic case. The report's sync note says when this happened, or that
timing may be wrong if ffmpeg could not do it. It WAS the cause on the `focus_clean` recording: `ffprobe` showed a nominal 60 fps and an average of 17.3 fps, the real length
was 33.0 s (not the 34.8 s the old code computed), and the re-run analyzed all 6 presses and judged all 6 visible. That run also showed the video
starting 11 s after the logger (the flashes were never recorded), so its alignment came from the Tab presses themselves; the audit
now records how many calibration flashes were matched and says so when none were. Earlier real-recording results used the old
mapping and should be re-run. Separately, if the calibration flashes were never in the video the
alignment still fails, and the audit now reports that and does not issue confident timing-dependent findings.

## Black recordings and what counts as a calibration flash

A recording whose recorded area goes black (a browser window moved to another Space, minimized, or a locked screen) looks to the
detectors like a page where nothing changes. Found on the `focus_bug` recording by `python -m a11y_replay.diagnose`, which lists every
screen change in a time range and draws a labeled contact sheet: the window left the recorded area at about 2.2 s and was black
for most of the first 12 s. Consequences, all fixed or guarded: (1) a Tab press with a black frame before or after it is reported as
"unreadable" (needs review, never a model call), not as "no indicator appeared"; (2) `blank_stretches` finds black runs of at least 1 s
and the audit warns at the top of the report and in the console when more than 10% of the recording is black or any run is 3 s or more;
(3) a pair of a flash-time key and a screen change counts as a calibration flash only if the change covers under 6% of the frame, because
two window-sized changes (28% and 22%) had been counted as "2 calibration flashes". The console and `report.json` now list every
pair with how large its matched change was.

## Real sessions: alignment, privacy, labels

Alignment. In real sessions the screen recorder and the key logger are started by hand, so their clocks can differ by
seconds, far outside the local sync window (about half a second). `estimate_offset_search` histograms the time gap
between every key press and every visual change within +-120 s, takes the peak as a coarse offset, and refines it with the
local estimator. It needs irregularly spaced key presses (evenly spaced ones repeat at multiples of the spacing), which is why
the recorder's calibration flashes use random gaps. If even this cannot find a consistent offset the report says the sync is not trusted.
A transcript can be on the key-log clock (`--transcript-clock log`, benchmark default) or the video's own clock (`video`,
which is what audio taken from the recording is on).

Privacy. The key logger writes only navigation keys by default and drops typed characters at the point of capture; text
logging is an explicit opt-in. The recording guide tells people to close sensitive apps, and `recordings/` is git-ignored.

Labels. `eval_real` compares findings with labels written by a person on the video's timeline. It reports whether the labeler
declared themselves independent of the tool's author and whether they reviewed the whole recording (only then are extra findings
called false alarms). Nothing here has been run on real recordings yet; the key logger's pynput/tkinter parts have not been run at all.


### Matching guards and partial announcements (added after the first real runs)

Character similarity cannot see meaning. Measured before any recording: "13 results found" scores 89 against "3 results
found", "No results found" 88, and the fragment "Upload failed" 100 against "Upload failed: file too large". Three rules
were added (`announce_content_guards`, on by default): (1) every NUMBER on screen must be spoken, digits and words compared
as integers; (2) NEGATION must agree; (3) the announcement must COVER at least 60% of the message's words (each word may
differ by small spelling, "changes"~"change", "to"~"too", but not be a different word, "saved" vs "saving"). A fragment or a
different key word becomes needs_review, never a pass and never a finding. Every real faster-whisper output seen so far
("Change is saved.", "Alert, Upload, failed. File to large.", "Three results found.") has a regression test.

Result on 7 recorded `status_messages_noisy` samples (21 messages, 11 silent, 10 announced; simulated transcript):
guards off, 3 of 11 silent messages flagged (recall 0.27); guards on, 6 of 11 flagged as silent and the other 5 sent to
review, so none passes as announced; precision 1.00 in both. By category (flagged / sent to review): near_miss 3/0,
wrong_number 2/0, negation 1/0, truncated 0/3, confusable 0/2.

How far to trust it: the rules were written after seeing exactly these examples and the benchmark categories were then built
around them, so this shows the rules do what they were designed to do, not that the matcher is robust in general. Counts per
category are 1 to 3. The noisy set is adversarial by construction. Known costs and gaps: a recognizer that garbles a number is
rejected like a wrong one; synonyms, paraphrases and semantic opposites ("failed" vs "complete") are not handled by rules and
would need a model to judge equivalence (not built); and "Changes saving" versus a real mishearing like "Change is saved" can
only be told apart by the word-coverage rule, which rests on two examples.


## First real recording: what it showed, and the flash detector fix

Recording: a keyboard session on a Streamlit app, about 104 s, whole screen, key log from `a11y_replay.recorder`.

Clock sync worked on a real, hand-started recording: the local estimator failed (the two clocks differed by more than
half a second) and the wide search found an offset of 1.207 s from 15 key/change pairs with a spread of 0.077 s. Caveat:
the video also contained many large changes, which could in principle support a wrong alignment; a spread that small
argues against it.

The flash check reported 14 flashing episodes, "up to 9-11 flashes within one second over 18-43% of the frame". The
evidence frames showed the screen changing between unrelated windows (the Streamlit page, a long chat page, a dark
terminal), and a synthetic test then reproduced the claim: ordinary scrolling text at 120 px/s or faster was reported as
"up to 11 flashes within one second over 82% of the frame". Cause: flashing was judged per pixel, and a pixel under
scrolling text alternates between ink and paper. I had only ever tested full-width black-and-white banners.

Fix: a pixel counts as flashing only if the average brightness of its region (a cell of a `flash_grid` x `flash_grid`
division of the frame) flashes by the same rule. Choosing the grid from the maximum swing of a cell's average brightness
on synthetic dense scrolling text (17% ink, the bar is 0.10; slow and fast scrolls shown as 60 / 300 px/s):

| grid | cell | max swing at 60 px/s | at 300 px/s |
|---|---|---|---|
| 12 x 12 | 2.3% of frame | 0.174 (flagged) | 0.071 |
| 8 x 8 | 1.6% | 0.106 (flagged) | 0.056 |
| 6 x 6 | 2.8% (= the minimum area) | 0.085 | 0.058 |
| 4 x 4 | 6.3% | 0.039 | 0.039 |

The default is 4 x 4 for the margin; 6 x 6 would leave only a 15% margin. The area rule still applies per pixel on top of
this, so the cost of larger cells is sensitivity to very low-contrast small flashes, not to area. This was chosen from synthetic
text only; the real check is a re-run on the original recording, NOT yet done.

Limits, both covered by tests: a fine pattern that inverts (a checkerboard) has a constant regional average and is no
longer detected; and very large high-contrast bands scrolling fast (spacing of 1 to 2 cells, 600 px/s in the test) are
still reported. The benchmark flash recordings are unchanged (3 of 3 found, none false). An important scope note: the audit
analyses the whole recording, so anything else on screen (other windows, a video playing elsewhere) is analysed too.