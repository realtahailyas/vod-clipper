"""Streaming audio feature extraction (loudness + speech-band energy per hop), with an on-disk cache."""
from __future__ import annotations

import collections
import hashlib
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .. import media

SAMPLE_RATE = 16000
FRAME = 800                      # 50 ms
CHUNK_SECONDS = 30
DB_FLOOR = -90.0
_EPS = 10 ** (DB_FLOOR / 10)
_BAND = (300.0, 3400.0)


@dataclass
class AudioFeatures:
    hop: float                 # seconds per frame
    duration: float            # seconds actually analysed
    loudness_db: np.ndarray    # float32, per hop, RMS level in dBFS (floor -90)
    vocal_db: np.ndarray       # float32, per hop, 300-3400 Hz band energy in dB (floor -90)

    def times(self) -> np.ndarray:
        """Centre time of each frame."""
        return (np.arange(len(self.loudness_db)) + 0.5) * self.hop

    def index(self, t: float) -> int:
        """Frame index containing time t, clamped to the valid range."""
        n = len(self.loudness_db)
        return int(min(max(int(t // self.hop), 0), max(n - 1, 0)))


def _to_db(power: np.ndarray) -> np.ndarray:
    return np.maximum(10.0 * np.log10(np.maximum(power, _EPS)), DB_FLOOR).astype(np.float32)


def _band_bins() -> slice:
    df = SAMPLE_RATE / FRAME
    return slice(int(np.ceil(_BAND[0] / df)), int(np.floor(_BAND[1] / df)) + 1)


def _process_blocks(x: np.ndarray, hop_samples: int, frames_per_hop: int,
                    band: slice) -> tuple[np.ndarray, np.ndarray]:
    """x: float32 samples, len = nblocks*hop_samples. Returns (loudness_db, vocal_db) per block."""
    nb = len(x) // hop_samples
    blocks = x[: nb * hop_samples].reshape(nb, hop_samples)
    # time-domain mean square (exact RMS level)
    ms = np.mean(blocks.astype(np.float64) ** 2, axis=1)
    frames = blocks[:, : frames_per_hop * FRAME].reshape(nb * frames_per_hop, FRAME)
    spec = np.fft.rfft(frames, axis=1)
    p = (spec.real ** 2 + spec.imag ** 2)[:, band].sum(axis=1)
    band_ms = 2.0 * p / (FRAME * FRAME)               # Parseval: band share of mean-square
    vocal = band_ms.reshape(nb, frames_per_hop).mean(axis=1)
    return _to_db(ms), _to_db(vocal)


def _cache_path(video: Path, hop: float, track: int, cache_dir: Path) -> Path:
    st = video.stat()
    key = f"{video.resolve()}|{st.st_size}|{st.st_mtime_ns}|{hop}|{track}"
    return cache_dir / f"audio_{hashlib.sha1(key.encode('utf-8')).hexdigest()}.npz"


def compute_features(video: Path, hop: float = 0.5, audio_track: int = 0,
                     cache_dir: Path | None = None,
                     progress: Callable[[float], None] | None = None) -> AudioFeatures:
    """Decode the audio in ~30 s chunks and reduce it to per-hop loudness / vocal-band dB curves."""
    video = Path(video)
    cache_file = _cache_path(video, hop, audio_track, Path(cache_dir)) if cache_dir else None
    if cache_file is not None and cache_file.exists():
        try:
            with np.load(cache_file) as z:
                feats = AudioFeatures(hop=float(z["hop"]), duration=float(z["duration"]),
                                      loudness_db=z["loudness_db"].astype(np.float32),
                                      vocal_db=z["vocal_db"].astype(np.float32))
            if progress:
                progress(1.0)
            return feats
        except Exception:
            pass  # corrupt cache -> recompute

    info = media.probe(video)
    if info.audio_streams == 0:
        raise ValueError(f"{video.name} has no audio stream; cannot detect highlights from audio.")
    if audio_track >= info.audio_streams:
        raise ValueError(f"{video.name} has only {info.audio_streams} audio stream(s); "
                         f"audio_track={audio_track} does not exist.")

    hop_samples = max(FRAME, int(round(hop * SAMPLE_RATE)))
    frames_per_hop = hop_samples // FRAME
    band = _band_bins()
    chunk_bytes = CHUNK_SECONDS * SAMPLE_RATE * 2
    total_bytes = max(1.0, info.duration * SAMPLE_RATE * 2)

    cmd = [media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", str(video), "-map", f"0:a:{audio_track}", "-vn", "-ac", "1",
           "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err_tail: collections.deque[bytes] = collections.deque(maxlen=30)

    def _drain() -> None:
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                err_tail.append(line)
        except Exception:
            pass

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    loud_parts: list[np.ndarray] = []
    vocal_parts: list[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    leftover_bytes = b""
    total_samples = 0
    read_bytes = 0
    try:
        while True:
            raw = proc.stdout.read(chunk_bytes)  # type: ignore[union-attr]
            if not raw:
                break
            read_bytes += len(raw)
            raw = leftover_bytes + raw
            usable = len(raw) - (len(raw) % 2)
            leftover_bytes = raw[usable:]
            pcm = np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0
            total_samples += len(pcm)
            buf = np.concatenate([carry, pcm]) if len(carry) else pcm
            nb = len(buf) // hop_samples
            if nb:
                ld, vd = _process_blocks(buf[: nb * hop_samples], hop_samples, frames_per_hop, band)
                loud_parts.append(ld)
                vocal_parts.append(vd)
            carry = buf[nb * hop_samples:].copy()
            if progress:
                progress(min(0.999, read_bytes / total_bytes))
        # final partial block: use whatever whole 50 ms frames remain
        nf = len(carry) // FRAME
        if nf >= 1:
            part = carry[: nf * FRAME]
            ms = float(np.mean(part.astype(np.float64) ** 2))
            spec = np.fft.rfft(part.reshape(nf, FRAME), axis=1)
            p = (spec.real ** 2 + spec.imag ** 2)[:, band].sum(axis=1)
            vb = float(np.mean(2.0 * p / (FRAME * FRAME)))
            loud_parts.append(_to_db(np.array([ms])))
            vocal_parts.append(_to_db(np.array([vb])))
        proc.stdout.close()  # type: ignore[union-attr]
        rc = proc.wait()
        drainer.join(timeout=2)
        if rc != 0 and total_samples == 0:
            tail = b"".join(err_tail).decode("utf-8", "replace").strip()
            raise ValueError(f"Could not decode audio from {video.name}: {tail or f'ffmpeg exit {rc}'}")
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    if total_samples == 0 or not loud_parts:
        raise ValueError(f"No audio samples could be decoded from {video.name}.")

    feats = AudioFeatures(hop=float(hop), duration=total_samples / SAMPLE_RATE,
                          loudness_db=np.concatenate(loud_parts),
                          vocal_db=np.concatenate(vocal_parts))
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_name(cache_file.name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez(f, hop=feats.hop, duration=feats.duration,
                         loudness_db=feats.loudness_db, vocal_db=feats.vocal_db)
            os.replace(tmp, cache_file)
        except OSError:
            pass
    if progress:
        progress(1.0)
    return feats
