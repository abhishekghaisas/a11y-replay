# Recording and labeling real sessions

Everything the project has been tested on so far was made by the project's own generator and labeled by its
author. This guide is how to collect something that is neither. Plan for **10 to 20 recordings** across **at least
three different sites or apps**, ideally on more than one machine or browser. One of this project's bugs was found
only because a second machine rendered a text highlight differently.

## 1. What you need (macOS)

- A screen recorder (QuickTime, or Cmd+Shift+5). Record **only the browser window with the app** ("Record Selected
  Portion"), not the whole desktop; see the first note in section 3.
- `ffmpeg`, `tesseract` (`brew install ffmpeg tesseract`), and the key logger: `pip install pynput`.
- Python with tkinter for the calibration square (`brew install python-tk` if `import tkinter` fails).
- **Optional, for the status-message check:** a transcript of what a screen reader said. See section 4.

## 2. Privacy and consent, read this first

- The key logger sees **every key you press in every app**. By default it writes only navigation keys (Tab, Enter,
  Space, Escape, arrows, Page/Home/End, F-keys) and never the characters you type. Do not use `--log-text` unless
  nothing sensitive can be typed during the session.
- Close password managers, chats, email and notifications before recording. The screen recording captures everything
  visible.
- Record only sites you own, or have permission to test. Do not record other people's accounts or personal data.
- `recordings/` is in `.gitignore`. Keep it that way unless you have checked every recording for private content.

## 3. Recording one session

1. `mkdir -p recordings/checkout-flow`
2. Pick a **keyboard-only task of 1 to 3 minutes**. Good tasks, to cover the three checks:
   - *Focus:* press Tab through every control on a page or form, then Shift+Tab back.
   - *Status messages:* do something that produces a message with no page change: save, add to cart, submit a
     form with an error, run a search that reports a result count.
   - *Flashing:* load a page with an animated banner, carousel or loading effect and leave it on screen for 10 s.
3. Start the screen recording with Cmd+Shift+5 and choose **Record Selected Portion**: drag the box around the browser
   window that shows the app, nothing else. **Why:** the audit analyses everything in the recording. The first real
   recording was of the whole screen, and the tool reported 14 "flashing" episodes that were really other windows and
   scrolling text on the desktop. **Also: read these steps before you start recording.** Switching to another window
   to read instructions puts that window in the video.
   **Do not use "Record Selected Window"**, even though it feels more private. It captures only that one window's own pixels, and the
   calibration square is a separate window, so it can never appear in the video and the key log cannot be lined up (this happened on two
   recordings). Window capture also turns black whenever the window is not on the Desktop currently shown, for example if your terminal is
   a full-screen app on another Space. To keep a portion recording private: put the browser window alone inside the box (close or minimize
   everything else that could overlap it), turn on a Focus / Do Not Disturb mode so notifications cannot slide in, keep the browser and the
   terminal on the same Desktop (neither full-screen), and look at the recording before sharing it: it contains everything inside the box.
4. **Wait about two seconds after the recording starts**, then, in a terminal: `python -m a11y_replay.recorder --out recordings/checkout-flow/keys.jsonl`
   A small square appears at the top-left and flashes four times over about six seconds. **Do not start
   the session until it has finished.** It is how the key log and the video are lined up. If you must move it,
   use `--marker-pos x,y` (screen points, from the top-left of the screen) to put it **inside the box you selected**; if it is
   outside the recorded area the audit cannot line the key log up with the video.
5. Do the task. Use the keyboard for navigation, and pause about a second between actions so effects are visible.
6. Press **F10** to stop the logger, then stop the screen recording. (While the logger runs, the terminal does not echo
   keystrokes and discards anything typed into it, so nothing you press during a session can be run by the shell afterwards.)
7. **Keep the browser on the same Desktop as the terminal, and do not make it a full-screen app.** A full-screen browser lives in its own
   Space, and switching Spaces turned the recorded area black in one recording. Do not switch windows, Spaces or Mission Control during the
   recording.
8. **Before anything else, check the video really contains the calibration.** Open it in QuickTime and scrub the first 8 seconds:
   you must see the small square flash four times. If you do not, the key log cannot be lined up with the video and this
   recording does not count: record again. (Also check, in Cmd+Shift+5 > Options, that the **Timer is None**; a 5 or 10 second timer
   swallows the flashes.) Run `python -m a11y_replay.diagnose recordings/<name> --from 0 --to 12` and open the PNG it writes: the page must be
   visible in every tile, with the small pink square flashing four times, and no black frames. Then save the video as `recordings/checkout-flow/video.mov` (or `.mp4`). If the audit cannot open it, convert and
   shrink it (a full retina recording is huge and slow to analyze):
   `ffmpeg -i video.mov -vf scale=1280:-2 -c:v libx264 -pix_fmt yuv420p video.mp4`
9. Try the audit: `python -m a11y_replay.audit recordings/checkout-flow` and open the report. Look at the **Clock
   sync** line first. If it says the sync is not trusted, the calibration square was probably outside the recorded
   area; record again.

Do **not** read the audit's findings before the labeler has labeled the recording (section 5).

**Check your key log before auditing.** The focus check only runs if the log contains Tab presses. This tells you what was captured:
`grep -o '"key": "[^"]*"' recordings/<name>/keys.jsonl | sort | uniq -c`

## 4. Getting a transcript for the status-message check (optional)

Capturing a screen reader's audio into a recording is not trivial on macOS: QuickTime does not record system audio
without a loopback audio tool. Options, from simplest:

1. **Skip it.** The focus and flashing checks still run; the status-message check is reported as skipped.
2. **Write it down.** If you can hear the screen reader but cannot capture it, listen to the recording and write
   `recordings/<name>/utterances.json`: `[{"start": 31.4, "end": 32.9, "text": "Order placed"}]`, with times in
   seconds on the **video's** timeline. Audit with `--transcript-clock video`. This makes the transcript
   human-made and error-free, so say so when you report results.
3. **Capture audio and recognize it.** If the video has an audio track with the screen reader in it, audit with
   `--transcribe --transcript-clock video` (the first run downloads a speech model). This is the only option that
   exercises speech recognition on real screen-reader audio. Expect more errors than the synthetic benchmark showed.

## 5. Labeling (done by someone else)

For the labels to count as evidence they must come from a person who is **not the tool's author** and who has **not seen
the audit output**.

1. Watch the whole video in a player that shows the time. Do not open `audit_report/`.
2. Note each problem, with its time in **seconds on the video's timeline**, and which of these it is:
   - **2.4.7 Focus visible:** after a keyboard focus move, you cannot see which element has focus.
   - **4.1.3 Status messages:** a message appears on screen (a toast, a banner, a result count) and the screen reader
     does not announce it. You need the transcript or audio to know this, so skip it if there is none.
   - **2.3.1 Flashing:** something flashes more than three times in one second over a large part of the screen.
3. Copy `docs/human_labels.example.json` to `recordings/<name>/human_labels.json` and fill it in:
   - `independent`: `true` only if you are not the person who built the tool.
   - `complete`: `true` only if you watched the **entire** recording and listed everything you saw. Only then can an extra
     finding from the tool be called a false alarm; otherwise it is "unconfirmed".
4. If you are unsure whether something counts, **say so in `note` and decide with a second labeler.** Disagreement is a
   result worth reporting, not noise to hide. Ideally two people label separately (`human_labels_A.json`,
   `human_labels_B.json`) and agree on a final `human_labels.json`.

## 6. Scoring

```
python -m eval.eval_real recordings/*
python -m eval.eval_real recordings/* --model claude-sonnet-5-5      # let the model judge unclear focus cases
```

Per recording and in total, it reports found, missed (and "flagged for review only"), and false alarms per criterion,
and it prints warnings if labels are not independent or not complete, or the clock sync was not trusted.

## 7. What you can honestly say afterwards

You can report how the tool did on **these** recordings against **these** labelers, with the counts. You cannot say
it is accurate in general, you should not tune thresholds on these recordings and then report results on them (keep
some recordings unseen until the end), and any recording where the clock sync was not trusted should be reported
separately.

## Troubleshooting

- **"This process is not trusted! Input event monitoring will not be possible…" or no keys logged:** macOS has not
  allowed your terminal to watch the keyboard. In System Settings > Privacy & Security add the terminal (or VS Code,
  whichever runs the command) under **both** Accessibility **and** Input Monitoring, switch them on, then **quit and
  reopen** the terminal. The recorder now says this itself at startup when it can tell.
  **A quick test that mimics a real session:** start the recorder, click into your browser (not the terminal), press
  Tab five times, then F10. `cat` the file: the five Tab lines must be there. Keys pressed while the terminal itself
  is in front can be captured even without permission, so testing in the terminal alone proves nothing.
- **tkinter missing:** `brew install python-tk`, or run with `--no-marker` and rely on key presses that change the screen
  (Tab rings, for example). Alignment is then less reliable on pages whose problem is "nothing visibly happens".
- **Sync not trusted:** record again with the square visible, make sure to wait for all four flashes, and avoid moving
  windows over the square.
  To see what is actually in the recording, run `python -m a11y_replay.diagnose recordings/<name> --from 0 --to 12`: it
  lists every screen change and writes a contact sheet of frames (`audit_report/diagnose_0-12s.png`). If the pink square
  is missing from the first frames, the recording does not contain the calibration.