# Example recordings for the hosted demo

Five short **synthetic** recordings of this project's own test pages (`benchmark/testapps/`), made with
`python -m benchmark.generate --seed 11` on Linux in headless Chromium. They contain no real person's screen. Each is under
110 KB. They are committed to the repository (the rest of `recordings/` and `benchmark/data/` is git-ignored) because the hosted
demo cannot run a browser to make its own.

The pages have a switchable bug, so the right answer is known. "Observed" is what the audit actually produced when each example was
run through the demo path before shipping (not a claim about other recordings):

| Example | What it is | Expected | Observed |
|---|---|---|---|
| `focus_page_with_bug` | a settings form whose bug removes every focus indicator; six Tab presses | 6 focus findings | 6 findings |
| `focus_page_clean` | the same form with visible focus rings | no findings | 5 presses judged fine, **1 sent to review** (see below) |
| `flashing_banner_with_bug` | a banner that flashes for 2 s | 1 flashing finding | 1 finding |
| `flashing_banner_clean` | the same banner pulsing slowly | no findings | no findings |
| `status_messages_two_silent` | three status messages; two are never announced | 2 findings (4.1.3) | 2 findings, and the announced message judged fine |

**About the one that goes to review.** On the clean form, the last Tab moves the focus ring from *Save changes* to the adjacent
*Cancel* button. The old and new rings touch, so the classical detector cannot tell which appeared and sends it to review, as designed.
With a language model enabled (local use only) it would be resolved; the hosted demo has the model turned off. The same limit showed
up on a real recording (see `docs/RESULTS.md`).

**About the transcript.** `status_messages_two_silent/utterances.json` is **simulated**: it contains only the one message that was
supposed to be announced, with its timing. It is not the output of a real screen reader or a speech recognizer.

To check that every folder behaves as its name says, run `python -m a11y_replay.verify_examples` from the repository root. That works
whoever made the files; `CHECKSUMS.sha256` only applies if you use exactly the files shipped here.

Files per example: `video.mp4`, `keys.jsonl` (the key log), and `utterances.json` where there is a transcript.