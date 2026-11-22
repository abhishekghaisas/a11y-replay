# Results and open issues

One page, written to be read before anything else in this repository. Everything here comes from the author's own
machine. **Nobody else has labeled a recording or run the tool.**

## Evidence strength

| Level | Meaning |
|---|---|
| **A** | Synthetic recordings from this project's own test pages, labels written by the author |
| **B** | A real recording of a real browser session, judged by the author |
| **C** | Labels written by someone independent of the author | 

Nothing is at level C yet.

## What was tested

| Check | Result | Level |
|---|---|---|
| Focus visible (2.4.7) | 19 of 19 benchmark failures found, no confident mistakes, about a quarter of presses sent to a language model; Sonnet, Haiku and Opus agreed with the expected verdict on every clear case, so the benchmark could not tell them apart | A |
| Focus visible (2.4.7), real recording, clean page | all **6 of 6** Tab presses judged visible, no false alarm, after the variable-frame-rate fix. Caveat: the video started 11 s after the key logger, so the calibration flashes were not recorded and the clocks were aligned from the Tab presses themselves, which is partly circular (the audit now says so) **Re-run with the current code on 2026-11-21: still 6 of 6 visible, 0 false alarms** (sync trusted from 6 pairs, spread 0.01 s, 0 calibration flashes matched, so the circularity caveat applies) | B |
| Focus visible (2.4.7), real recording, clean page, second recording | **5 of 6** Tab presses judged visible with real rings (contrast 2.5:1 to 6.6:1); the sixth was escalated as "unclear" because the old ring on *Save changes* and the new one on the adjacent *Cancel* merged into one changed region (the documented touching-rings limit; the true answer is visible). **No false failure.** Clock sync trusted from 10 pairs, spread 0.02 s; the screen recording started 57 s before the logger and only 1 of 4 calibration flashes was matched, so the alignment rests mainly on the Tab presses and that one flash agreeing | B |
| Language-model escalation on one real unclear press | the ring moving between touching buttons from the lost second clean recording (saved as `docs/examples/touching_rings_*.png`): Sonnet 5.5 answered **yes**, "orange outline", naming the right element (Cancel gained the ring) in 740 tokens in, 77 out, about $0.002. **One case, not a rate.** The prompt was not changed after seeing it. The images are the report's evidence strips, smaller than the crops an escalation sends | B |
| Focus visible (2.4.7), real recording, buggy page (every press should fail), two recordings | **Run 2: 5 of 6 reported as findings, 1 left for review, 0 judged visible**; run 1: 4 of 6, 2 for review, 0 judged visible. Clock sync trusted both times (run 2: 3 pairs, 2 calibration flashes, spread 0.02 s; no black screen). Every press left for review coincided with something on screen that was NOT the page: run 1, the macOS menu bar and the *Stop Screen Recording* tooltip; run 2, a macOS "quit unexpectedly" crash dialog over the left edge of the page being dismissed (4.85% of the frame). The detector declined to call those presses failures because something had changed, and its evidence frames show why. **No false pass in either run** | B |
| Flashing (2.3.1) | 3 of 3 benchmark flashes found, none false; benchmark cases are easy | A |
| Flashing, real recordings | known-flashing page, re-run after the fix: sync trusted from 4 pairs (offset +26.3 s: the screen recording started 26 s before the logger), **1 flashing finding at 17.06 s** on the key-log clock. Whole-screen scrolling session (`streamlit1`): now **2 flashing findings** (77.9 s and 83.0 s) with sync NOT trusted; evidence frames **examined: both episodes are the screen switching between unrelated apps** (a chat page, a document, the Streamlit app, a dark code editor, other websites) inside the whole-screen recording; the detector described the recording accurately and the findings say nothing about the Streamlit app; the earlier "14, then 0" came from the old time mapping. The clean flash page has not been recorded | B |
| Flashing, a flash reported in the second clean-page recording | **not a page problem.** The three evidence frames show a document page, a portfolio site and a dark editor: the screen was switching between apps (inside the recorded box) while the person moved between windows. The detector described the recording accurately; the recording included more than the page under test | B |
| Status messages seen vs heard (4.1.3) | 9 of 9 judged correctly with simulated and with real speech recognition (one synthetic voice, 6 lines). On an adversarial set (11 silent messages with wrong speech nearby): without content rules 3 flagged; with them 6 flagged and 5 sent to review, none passed. The rules were written after seeing those cases | A |
| Clock sync of a hand-started recording | offset 1.2 s found from 15 key/change pairs, spread 0.08 s | B |
| Key logger | records navigation keys only; typed characters verified absent; captured keys while a browser was in front | B |

Not built: pause/stop/hide (2.2.2), keyboard traps (2.1.2). Not covered: anything about the page's code.

## What real contact with real recordings broke

Each of these passed every test I had written. They are the reason to distrust results at level A.

1. A second Mac rendered a text-selection highlight differently from mine, and the focus classifier called it a focus ring.
2. Two packages in the speech stack disagreed about a library version and crashed the first run.
3. Speech recognition merged two lines spoken 11 seconds apart into one, filing the second 11 seconds early.
4. Keys typed into the terminal while the logger ran were left for the shell to execute afterwards.
5. Scrolling text was reported as flashing across most of the screen (14 false findings on a real session).
6. A report with zero findings did not say that two of three checks had never run.
7. A placeholder pasted literally into the logger's command crashed it late with a bare error.
8. Some screen recordings have distorted timing; the generator now re-records samples whose clocks cannot be aligned.
9. Screen recordings have a variable frame rate (a frame only when the screen changes), so "frame number / fps" is not time.
   **Confirmed on the clean focus recording** (ffprobe: nominal 60 fps, average 17.3 fps; the code had computed 34.8 s for a 33.0 s video).
   It produced a false "focus not visible" and zero clock-sync pairs. Fixed by converting to a constant-rate copy; the same recording
   then gave 6 of 6 visible. Every real-recording result from before this fix is suspect.
10. A recording whose recorded area went **black** (the browser window left the area) was audited as if it were a page that
    shows nothing: identical black frames read as "nothing changed, so no focus indicator", and two window-sized changes were
    counted as calibration flashes. Found by writing a diagnostic that draws a contact sheet of the video. The audit now flags
    black stretches, calls presses on black frames "unreadable", and counts only marker-sized changes as flashes.

## Open issues

- **The buggy focus page: 5 of 6 found in the best run, 0 false passes.** The unresolved presses were each muddied by unrelated desktop activity inside the recorded box (a tooltip, a crash dialog). A recording can only be as clean as the machine it is made on; the audit now shows the evidence so a reviewer can tell. Not worth chasing a perfect 6 of 6.
- **The clean focus recording passed, but its clock alignment was partly circular** (no calibration flashes were captured). A cleaner control records the flashes.
- **The negative controls were not recorded.** Without the clean focus page and the clean flash page on a real recording, there is no real-recording evidence that the tool stays quiet when nothing is wrong, apart from the scrolling session.
- **Real-recording results are mixed after the variable-frame-rate fix:** the flash page passed with trusted sync; the buggy focus page is inconclusive; the scrolling session's 2 flashing findings turned out to be app-switching inside the recorded screen (frames examined).
- **Parameters chosen from my own data.** The flash grid (4 x 4), the 80% match threshold, the 60% word coverage and the focus contrast cutoff (1.5:1) were chosen by looking at the author's data and are untuned elsewhere.
- **The classifier and matcher were changed repeatedly after failures on the data used to score them.** `docs/methodology.md` lists every change. Scores on those samples are development checks, not accuracy estimates.
- **The Gradio demo has not been run against the real library by anyone.** Its logic is tested; its wiring was checked against a stand-in. `python app.py --selftest` is the first thing to run.
- **Flash detection** is the general-flash approximation only: no red flash, no 50 Hz bound, a checkerboard that inverts is missed, and very large high-contrast bands scrolling fast can still be reported.

## What would change this picture

Independent labels on 10 to 20 real recordings from at least three sites would move the main claims to level C, and would probably
break something not yet seen, as every earlier real run did.