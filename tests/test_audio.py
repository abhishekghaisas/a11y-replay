"""Audio plumbing tests. `say` and Whisper are replaced by stand-ins; neither real tool is exercised."""
import json
import types

import numpy as np

from a11y_replay.events import Utterance
from a11y_replay.ingest.audio import load_or_transcribe, transcribe
from benchmark.tts import SR, add_audio_to_sample, have_say, mix_utterances, read_wav, write_wav
from eval.common import word_error_rate
from eval.eval_announce import asr_quality


def fake_synth(text):
    """0.1 s of tone per character, so every utterance has a known length."""
    n = int(0.1 * SR * len(text))
    return (0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)).astype(np.float32)


def test_wav_round_trip(tmp_path):
    a = (0.3 * np.sin(np.linspace(0, 50, 4000))).astype(np.float32)
    write_wav(tmp_path / "x.wav", a)
    b = read_wav(tmp_path / "x.wav")
    assert len(b) == len(a) and np.max(np.abs(a - b)) < 1e-3


def test_utterances_are_placed_at_their_start_times():
    utts = [{"start": 1.0, "end": 9.9, "text": "ab"}, {"start": 3.0, "end": 9.9, "text": "abc"}]
    audio, placed = mix_utterances(utts, 5.0, fake_synth)
    loud = np.flatnonzero(np.abs(audio) > 0.05)
    assert loud[0] / SR == 1.0 or abs(loud[0] / SR - 1.0) < 0.01
    assert audio[int(0.5 * SR)] == 0 and np.abs(audio[int(2.5 * SR)]) == 0       # silence between lines
    assert [p["end"] for p in placed] == [1.2, 3.3]                              # real spoken lengths
    assert placed[0]["text"] == "ab"


def test_track_grows_if_speech_runs_past_the_end():
    audio, _ = mix_utterances([{"start": 1.0, "end": 2.0, "text": "x" * 30}], 2.0, fake_synth)
    assert len(audio) >= int(4.0 * SR)


def test_add_audio_to_sample_writes_audio_and_meta(tmp_path):
    (tmp_path / "labels.json").write_text(json.dumps({"duration_s": 8.0}))
    (tmp_path / "utterances.json").write_text(json.dumps([{"start": 2.0, "end": 3.0, "text": "hello"}]))
    out = add_audio_to_sample(tmp_path, fake_synth, voice="Test")
    assert out.exists() and len(read_wav(out)) >= 8 * SR
    meta = json.loads((tmp_path / "audio_meta.json").read_text())
    assert meta["voice"] == "Test" and meta["utterances"][0]["end"] == 2.5
    assert json.loads((tmp_path / "utterances.json").read_text())[0]["end"] == 3.0   # ground truth untouched


def test_sample_without_a_transcript_is_skipped(tmp_path):
    (tmp_path / "labels.json").write_text("{}")
    assert add_audio_to_sample(tmp_path, fake_synth) is None


def test_have_say_returns_a_bool():
    assert isinstance(have_say(), bool)


class FakeWhisper:
    """Records each call; returns the next scripted list of (start, end, text) segments."""

    def __init__(self, script=None):
        self.calls = 0
        self.sources, self.kwargs = [], []
        self.script = script or [[(1.0, 2.0, "  Changes saved "), (3.0, 3.1, "   ")]]

    def transcribe(self, source, **kw):
        self.sources.append(source)
        self.kwargs.append(kw)
        segs = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return iter([types.SimpleNamespace(start=a, end=b, text=t) for a, b, t in segs]), types.SimpleNamespace()


def _wav(tmp_path, sr=SR):
    path = tmp_path / "audio.wav"
    write_wav(path, 0.2 * np.sin(np.linspace(0, 300, sr * 2)).astype(np.float32), sr)
    return path


def test_transcribe_strips_text_and_drops_empty_segments(tmp_path):
    out = transcribe(_wav(tmp_path), model=FakeWhisper())
    assert out == [Utterance(1.0, 2.0, "Changes saved")]


def test_chunk_mode_asks_for_language_and_does_not_double_filter(tmp_path):
    m = FakeWhisper()
    transcribe(_wav(tmp_path), model=m)
    assert m.kwargs[0]["language"] == "en" and m.kwargs[0]["vad_filter"] is False   # already cut at silences


def test_whole_file_mode_keeps_the_recognizers_own_filter(tmp_path):
    m = FakeWhisper()
    transcribe(_wav(tmp_path), model=m, chunk=False)
    assert m.kwargs[0]["vad_filter"] is True


def test_samples_not_a_path_are_given_to_whisper_so_its_pyav_decoder_is_never_used(tmp_path):
    """Regression: faster-whisper's own decoder crashed on a PyAV version mismatch."""
    m = FakeWhisper()
    transcribe(_wav(tmp_path), model=m)
    assert isinstance(m.sources[0], np.ndarray) and m.sources[0].dtype == np.float32
    assert len(m.sources[0]) == SR * 2


def test_other_wav_formats_fall_back_to_a_path(tmp_path):
    m = FakeWhisper()
    transcribe(_wav(tmp_path, sr=8000), model=m)       # 8 kHz is not what Whisper expects: let it decode
    assert isinstance(m.sources[0], str)


def test_transcripts_are_cached_per_model_size(tmp_path):
    _wav(tmp_path)
    m = FakeWhisper()
    load_or_transcribe(tmp_path, "base", model=m)
    load_or_transcribe(tmp_path, "base", model=m)
    assert m.calls == 1                                   # second call read the cache
    load_or_transcribe(tmp_path, "small", model=m)
    assert m.calls == 2                                   # different model size -> new transcription
    load_or_transcribe(tmp_path, "small", model=m, force=True)
    assert m.calls == 3


def test_word_error_rate():
    assert word_error_rate("changes saved", "Changes saved!") == 0.0
    assert word_error_rate("changes saved", "changes saved now") == 0.5
    assert word_error_rate("three results found", "tree results found") == 1 / 3
    assert word_error_rate("3 results found", "Three results found.") == 0.0   # numbers are not errors
    assert word_error_rate("", "") == 0.0 and word_error_rate("", "x") == 1.0


def test_asr_quality_counts_missed_invented_and_errors():
    truth = [Utterance(2.0, 3.0, "Changes saved"), Utterance(8.0, 9.0, "Upload failed")]
    asr = [Utterance(2.3, 3.2, "Changes safe"), Utterance(20.0, 21.0, "thank you")]   # 2nd line missed, one invented
    q = asr_quality([{"utts": truth, "asr": asr}])
    assert q["n"] == 2 and q["missed"] == 1 and q["invented"] == 1
    assert abs(q["wer"][0] - 0.5) < 1e-9 and abs(q["terr"][0] - 0.3) < 1e-9


def _track(bursts, seconds=24.0, hiss=0.0):
    """A 16 kHz track with a tone wherever there is a burst, silence (or faint hiss) elsewhere."""
    x = np.zeros(int(seconds * SR), np.float32)
    for a, b in bursts:
        i, j = int(a * SR), int(b * SR)
        x[i:j] = 0.3 * np.sin(2 * np.pi * 300 * np.arange(j - i) / SR)
    if hiss:
        x += (np.random.default_rng(0).normal(0, hiss, len(x))).astype(np.float32)
    return x


def test_split_on_silence_finds_each_burst():
    from a11y_replay.ingest.audio import split_on_silence
    cuts = split_on_silence(_track([(7.8, 9.0), (18.6, 20.4)]))
    assert len(cuts) == 2
    assert abs(cuts[0][0] - 7.65) < 0.1 and abs(cuts[0][1] - 9.15) < 0.1      # 0.15 s padding each side
    assert abs(cuts[1][0] - 18.45) < 0.1 and abs(cuts[1][1] - 20.55) < 0.1


def test_a_short_pause_inside_one_line_does_not_split_it():
    from a11y_replay.ingest.audio import split_on_silence
    assert len(split_on_silence(_track([(5.0, 6.0), (6.5, 8.0)]))) == 1        # 0.5 s pause (a comma)
    assert len(split_on_silence(_track([(5.0, 6.0), (7.5, 8.0)]))) == 2        # 1.5 s pause is a new line


def test_clicks_and_silence_are_not_speech():
    from a11y_replay.ingest.audio import split_on_silence
    assert split_on_silence(_track([(5.0, 5.05)])) == []                        # a 50 ms click
    assert split_on_silence(np.zeros(SR * 3, np.float32)) == []
    assert split_on_silence(np.zeros(10, np.float32)) == []


def test_faint_background_hiss_does_not_hide_the_silences():
    from a11y_replay.ingest.audio import split_on_silence
    assert len(split_on_silence(_track([(7.8, 9.0), (18.6, 20.4)], hiss=0.004))) == 2


def test_lines_far_apart_are_never_given_to_the_recognizer_together(tmp_path):
    """Regression from the first real run: Whisper returned ONE segment, stamped with the first
    line's start time, for two lines spoken 11 s apart. Each line must be recognized separately."""
    write_wav(tmp_path / "audio.wav", _track([(7.8, 9.0), (18.6, 20.4)]))
    m = FakeWhisper(script=[[(0.2, 1.1, "Changes saved")], [(0.2, 1.9, "Alert, upload failed, file too large.")]])
    out = transcribe(tmp_path / "audio.wav", model=m)
    assert all(len(src) < 4 * SR for src in m.sources)                          # nothing near 24 s was sent
    assert [u.text for u in out] == ["Changes saved", "Alert, upload failed, file too large."]
    assert abs(out[0].start - 7.85) < 0.15 and abs(out[1].start - 18.65) < 0.15   # each at ITS OWN time


def test_segments_within_one_burst_are_joined_into_one_utterance(tmp_path):
    write_wav(tmp_path / "audio.wav", _track([(5.0, 8.0)], seconds=10))
    m = FakeWhisper(script=[[(0.2, 1.2, "Alert Upload failed."), (2.0, 2.9, "File to large.")]])
    out = transcribe(tmp_path / "audio.wav", model=m)
    assert len(out) == 1 and out[0].text == "Alert Upload failed. File to large."
    assert m.calls == 1


def test_old_cached_transcripts_from_another_method_are_redone(tmp_path):
    _wav(tmp_path)
    (tmp_path / "transcript_asr.json").write_text(json.dumps(
        {"engine": "faster-whisper", "model_size": "base", "utterances": [{"start": 0, "end": 1, "text": "stale"}]}))
    m = FakeWhisper()
    out = load_or_transcribe(tmp_path, "base", model=m)
    assert m.calls == 1 and out[0].text == "Changes saved"


def test_a_track_with_no_pauses_is_one_burst_not_nothing():
    """Regression: an adaptive threshold above every frame returned no speech at all, which would
    have produced an empty transcript for a recording that is speech from start to finish."""
    from a11y_replay.ingest.audio import split_on_silence
    cuts = split_on_silence(_track([(0.0, 6.0)], seconds=6.0))
    assert cuts == [(0.0, 6.0)]