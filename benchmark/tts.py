"""Simulated screen-reader AUDIO for benchmark samples, using macOS `say` (needs `say` and ffmpeg).

Each status_messages sample has `utterances.json`: the simulated transcript (text + start on the
key-log clock). This module speaks every line with `say` and places it in a single `audio.wav` at its
start time, so the audio and the key log share one clock (audio offset 0 by construction; real
recordings will not). The ground-truth transcript stays untouched; `audio_meta.json` records the voice
and the actual spoken end times.

Honest limits: this is one synthetic voice reading clean text, not NVDA or VoiceOver, with no
background noise, and utterances never overlap. It measures what speech recognition does to clean
synthetic speech, which is an optimistic case.

Usage (from the repo root):
    python -m benchmark.tts benchmark/data/*
    python -m benchmark.tts benchmark/data/* --voice Samantha --rate 180
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 16000  # sample rate Whisper expects


def have_say() -> bool:
    return bool(shutil.which("say") and shutil.which("ffmpeg"))


def read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM wav")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data


def write_wav(path, audio: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def say_synth(voice: str | None = None, rate: int | None = None):
    """Return synth(text) -> float32 mono audio at 16 kHz, using macOS `say` then ffmpeg.
    NOT run in the development environment (it has no `say`)."""
    def synth(text: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as d:
            aiff, wav = Path(d) / "u.aiff", Path(d) / "u.wav"
            cmd = ["say"] + (["-v", voice] if voice else []) + (["-r", str(rate)] if rate else []) + ["-o", str(aiff), text]
            subprocess.run(cmd, check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(SR), "-ac", "1", str(wav)],
                           check=True, capture_output=True)
            return read_wav(wav)
    return synth


def mix_utterances(utterances: list, seconds: float, synth, sr: int = SR):
    """Place each utterance's speech at its start time in one track. Returns (audio, placed) where
    `placed` has the real spoken end times. Overlapping utterances would simply add."""
    audio = np.zeros(int(seconds * sr) + 1, np.float32)
    placed = []
    for u in utterances:
        clip = np.asarray(synth(u["text"]), np.float32)
        a = int(round(u["start"] * sr))
        b = a + len(clip)
        if b > len(audio):
            audio = np.pad(audio, (0, b - len(audio)))
        audio[a:b] += clip
        placed.append({**u, "end": round(b / sr, 3)})
    return np.clip(audio, -1.0, 1.0), placed


def add_audio_to_sample(sample_dir, synth, voice: str | None = None, rate: int | None = None) -> Path | None:
    sample_dir = Path(sample_dir)
    utt_file, labels = sample_dir / "utterances.json", sample_dir / "labels.json"
    if not utt_file.exists() or not labels.exists():
        return None
    meta = json.loads(labels.read_text())
    utts = json.loads(utt_file.read_text())
    audio, placed = mix_utterances(utts, float(meta.get("duration_s", 0)) + 1.0, synth)
    out = sample_dir / "audio.wav"
    write_wav(out, audio)
    (sample_dir / "audio_meta.json").write_text(json.dumps(
        {"synthesizer": "macOS say", "voice": voice or "system default", "rate": rate, "sample_rate": SR,
         "clock": "key log (audio offset 0 by construction)", "utterances": placed}, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--rate", type=int, default=None, help="words per minute")
    ap.add_argument("--force", action="store_true", help="regenerate audio that already exists")
    a = ap.parse_args(argv)
    if not have_say():
        raise SystemExit("This needs macOS `say` and ffmpeg (brew install ffmpeg).")
    synth = say_synth(a.voice, a.rate)
    for s in map(Path, a.samples):
        if not (s / "utterances.json").exists():
            continue
        if (s / "audio.wav").exists() and not a.force:
            print(f"skip {s.name} (audio.wav exists; --force to redo)")
            continue
        print(f"wrote {add_audio_to_sample(s, synth, a.voice, a.rate)}")


if __name__ == "__main__":
    main()