"""Speech recognition for the screen-reader track (faster-whisper), returning Utterances.

Lines are separated by silence BEFORE recognition. On the first real run, Whisper returned one
segment for two lines spoken 11 s apart (text joined, timestamp of the first line), and split one
line into two segments 2 s apart that the audio does not contain. Segment boundaries from the
recognizer cannot be trusted to mean "one spoken line", so the audio is cut at real pauses and each
burst of speech is transcribed on its own; one burst becomes one Utterance.

Times are seconds on the AUDIO clock. For benchmark samples the audio is built on the key-log clock, so
the audio offset is 0; for real recordings it must be estimated separately.

NOT run in the development environment (no faster-whisper, no network); the model download on first
use (a few hundred MB depending on size) happens on the user's machine. The wrapper is thin on
purpose: pass `model=` to inject any object with a faster-whisper-style `.transcribe`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from a11y_replay.events import Utterance


def extract_audio(video_path: str | Path, out_wav: str | Path) -> bool:
    """Pull the video's audio track into a 16 kHz mono WAV with ffmpeg. Returns False if there is no
    audio track (or ffmpeg is missing). The audio is on the VIDEO clock, not the key-log clock."""
    if not shutil.which("ffmpeg"):
        return False
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1",
                        "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)], capture_output=True)
    return r.returncode == 0 and Path(out_wav).exists() and Path(out_wav).stat().st_size > 1000


def _load_model(model_size: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("pip install faster-whisper") from e
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def read_wav_16k_mono(path: str | Path):
    """Return float32 samples if the file is 16-bit PCM at 16 kHz (any channel count), else None.

    Passing samples to faster-whisper instead of a path skips its built-in decoder, which depends on
    PyAV. A mismatch between the two packages raised "open() got an unexpected keyword argument
    'metadata_errors'" on the first real run; benchmark audio is already in this format.
    """
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2 or w.getframerate() != 16000:
                return None
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() > 1:
                data = data.reshape(-1, w.getnchannels()).mean(axis=1)
            return data
    except (wave.Error, EOFError):
        return None


def split_on_silence(samples: np.ndarray, sr: int = 16000, frame_s: float = 0.02, min_silence_s: float = 0.8,
                     pad_s: float = 0.15, min_speech_s: float = 0.15) -> list:
    """Find bursts of speech as (start_s, end_s) pairs using frame energy.

    Pauses shorter than `min_silence_s` stay inside one burst (commas, breaths); bursts shorter than
    `min_speech_s` are dropped as clicks. The threshold adapts to the track: above the quiet-frame floor
    and a fraction of the loud-frame level. Reliable for clean audio such as the synthetic benchmark
    track; recordings with music or heavy background noise need a real voice-activity detector.
    """
    n = int(frame_s * sr)
    if len(samples) < n:
        return []
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    floor, loud = float(np.percentile(rms, 20)), float(np.percentile(rms, 95))
    if loud <= 0.005:
        return []                                  # silence (or a click too short to register)
    if floor >= 0.5 * loud:
        return [(0.0, round(len(samples) / sr, 3))]  # no pauses to cut at: the whole track is one burst
    thr = max(0.005, 3.0 * floor, 0.1 * loud)
    speech = rms > thr
    runs, start = [], None
    for i, s in enumerate(speech):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(speech)])
    gap = int(round(min_silence_s / frame_s))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    total = len(samples) / sr
    out = []
    for a, b in merged:
        if (b - a) * frame_s < min_speech_s:
            continue
        out.append((round(max(0.0, a * frame_s - pad_s), 3), round(min(total, b * frame_s + pad_s), 3)))
    return out


METHOD = "silence-chunked-v1"


def transcribe(audio_path: str | Path, model_size: str = "base", model=None, chunk: bool = True) -> list[Utterance]:
    """One Utterance per burst of speech. With chunk=False (or for audio that is not 16 kHz 16-bit WAV)
    falls back to the recognizer's own segments, which can merge or split lines."""
    model = model or _load_model(model_size)
    samples = read_wav_16k_mono(audio_path)
    if samples is None:
        segments, _ = model.transcribe(str(audio_path), language="en", vad_filter=True, beam_size=5,
                                       condition_on_previous_text=False)
        return [Utterance(round(float(s.start), 3), round(float(s.end), 3), s.text.strip())
                for s in segments if s.text.strip()]
    if not chunk:
        segments, _ = model.transcribe(samples, language="en", vad_filter=True, beam_size=5,
                                       condition_on_previous_text=False)
        return [Utterance(round(float(s.start), 3), round(float(s.end), 3), s.text.strip())
                for s in segments if s.text.strip()]
    out = []
    for a, b in split_on_silence(samples):
        piece = samples[int(a * 16000):int(b * 16000)]
        segs = [s for s in model.transcribe(piece, language="en", vad_filter=False, beam_size=5,
                                            condition_on_previous_text=False)[0] if s.text.strip()]
        if segs:
            out.append(Utterance(round(a + float(segs[0].start), 3), round(min(b, a + float(segs[-1].end)), 3),
                                 " ".join(s.text.strip() for s in segs)))
    return out


def load_or_transcribe(sample_dir: str | Path, model_size: str = "base", model=None,
                       force: bool = False) -> list[Utterance]:
    """Transcribe <sample>/audio.wav, caching the result in transcript_asr.json (keyed by model size)."""
    d = Path(sample_dir)
    cache = d / "transcript_asr.json"
    if cache.exists() and not force:
        c = json.loads(cache.read_text())
        if c.get("model_size") == model_size and c.get("method") == METHOD:
            return [Utterance(**u) for u in c["utterances"]]
    utts = transcribe(d / "audio.wav", model_size, model)
    cache.write_text(json.dumps({"engine": "faster-whisper", "model_size": model_size,
                                 "method": METHOD, "utterances": [u.__dict__ for u in utts]}, indent=2))
    return utts