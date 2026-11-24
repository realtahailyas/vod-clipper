"""faster-whisper wrapper producing word-timed transcripts in SOURCE time."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import wave

import numpy as np

from .config import TranscriptionConfig
from .models import Segment, Transcript, Word

log = logging.getLogger(__name__)


def load_wav(path: Path) -> np.ndarray:
    """Read the 16 kHz mono s16 WAV made by media.extract_audio as float32 in [-1, 1].

    Passing an array to faster-whisper skips its PyAV decoder, which breaks across PyAV releases.
    """
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != 16000:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _cuda_count() -> int:
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - missing lib / driver means no CUDA
        return 0


def resolve_device(cfg: TranscriptionConfig) -> tuple[str, str]:
    """Return (device, compute_type) after resolving 'auto'."""
    device = cfg.device
    if device == "auto":
        device = "cuda" if _cuda_count() > 0 else "cpu"
    compute = cfg.compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def resolve_cpu_threads(requested: int) -> int:
    """Explicit thread count if > 0, else half the cores clamped to 1..6 (keeps the laptop responsive)."""
    if requested > 0:
        return requested
    return max(1, min(6, (os.cpu_count() or 4) // 2))


class Transcriber:
    """Lazy faster-whisper model; reuses the first detected language for later clips."""

    def __init__(self, cfg: TranscriptionConfig) -> None:
        self.cfg = cfg
        self._device, self._compute = resolve_device(cfg)
        self._threads = resolve_cpu_threads(cfg.cpu_threads)
        self._model = None
        self._language = cfg.language

    @property
    def device(self) -> str:
        return self._device

    def _load(self) -> None:
        from faster_whisper import WhisperModel  # lazy: heavy import
        try:
            log.info("Loading whisper %s on %s (%s, cpu_threads=%d)",
                     self.cfg.model, self._device, self._compute, self._threads)
            self._model = WhisperModel(self.cfg.model, device=self._device,
                                       compute_type=self._compute,
                                       cpu_threads=self._threads, num_workers=1)
        except Exception as exc:  # noqa: BLE001
            if self._device != "cuda":
                raise
            log.warning("CUDA model load failed (%s); falling back to CPU int8", exc)
            self._device, self._compute = "cpu", "int8"
            self._model = WhisperModel(self.cfg.model, device="cpu", compute_type="int8",
                                       cpu_threads=self._threads, num_workers=1)

    def _run(self, wav: Path):
        segments, info = self._model.transcribe(
            load_wav(wav), word_timestamps=True, vad_filter=self.cfg.vad_filter,
            beam_size=self.cfg.beam_size, language=self._language,
            # Short clips gain nothing from cross-segment context, and it causes repeat loops
            # ("I'm in blood! I'm in blood!") over loud game audio.
            condition_on_previous_text=False)
        return list(segments), info  # generator: must be consumed

    def transcribe(self, wav: Path, offset: float = 0.0) -> Transcript:
        if self._model is None:
            self._load()
        try:
            segs, info = self._run(wav)
        except Exception as exc:  # noqa: BLE001
            if self._device != "cuda":
                raise
            log.warning("CUDA transcription failed (%s); falling back to CPU int8", exc)
            self._device, self._compute = "cpu", "int8"
            self._model = None
            self._load()
            segs, info = self._run(wav)

        if self._language is None and getattr(info, "language", None):
            self._language = info.language

        out: list[Segment] = []
        for s in segs:
            words = []
            for w in (s.words or []):
                text = (w.word or "").strip()
                if not text:
                    continue
                words.append(Word(text=text, start=float(w.start) + offset,
                                  end=float(w.end) + offset,
                                  probability=float(getattr(w, "probability", 1.0))))
            out.append(Segment(text=(s.text or "").strip(), start=float(s.start) + offset,
                               end=float(s.end) + offset, words=words))
        return Transcript(segments=out, language=getattr(info, "language", None) or self._language)
