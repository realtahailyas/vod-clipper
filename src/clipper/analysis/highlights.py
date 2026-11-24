"""Highlight scoring, peak picking, clip windows and transcript-based boundary refinement."""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..config import HighlightsConfig
from ..models import Highlight, Transcript
from . import audio, chat as chat_mod
from .audio import AudioFeatures

log = logging.getLogger(__name__)

BASELINE_SECONDS = 300.0
BLOCK_SECONDS = 15.0
SILENCE_DB = -60.0
Z_MIN, Z_MAX = -3.0, 6.0


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over n samples (edge-padded, same length)."""
    n = int(min(max(n, 1), max(len(x), 1)))
    if n <= 1 or len(x) == 0:
        return x.astype(np.float64)
    padded = np.pad(x.astype(np.float64), (n // 2, n - 1 - n // 2), mode="edge")
    c = np.concatenate([[0.0], np.cumsum(padded)])
    return (c[n:] - c[:-n]) / n


def _block_median(x: np.ndarray, bsz: int) -> np.ndarray:
    nb = int(math.ceil(len(x) / bsz))
    padded = np.full(nb * bsz, np.nan)
    padded[: len(x)] = x
    return np.nanmedian(padded.reshape(nb, bsz), axis=1)


def _rolling_median(v: np.ndarray, w: int) -> np.ndarray:
    w = min(max(w, 1), len(v))
    if w <= 1:
        return v
    left = w // 2
    padded = np.pad(v, (left, w - 1 - left), mode="edge")
    return np.median(sliding_window_view(padded, w), axis=1)


def robust_z(x: np.ndarray, hop: float, abs_floor: float = 1.0) -> np.ndarray:
    """Z-score vs. a rolling ~5 min median / MAD (block medians on ~15 s blocks, interpolated), clipped."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0:
        return x
    bsz = max(1, int(round(BLOCK_SECONDS / hop)))
    win = max(1, int(round(BASELINE_SECONDS / BLOCK_SECONDS)))
    centres = (np.arange(int(math.ceil(n / bsz))) + 0.5) * bsz
    pos = np.arange(n) + 0.5
    base_b = _rolling_median(_block_median(x, bsz), win)
    base = np.interp(pos, centres, base_b)
    dev_b = _rolling_median(_block_median(np.abs(x - base), bsz), win)
    mad = np.interp(pos, centres, dev_b) * 1.4826
    scale = np.maximum(mad, max(abs_floor, 0.1 * float(np.std(x))))
    return np.clip((x - base) / scale, Z_MIN, Z_MAX)


def score_curve(features: AudioFeatures, cfg: HighlightsConfig,
                chat: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Combine baseline-normalised loudness / vocal / excitement (+ chat) into one score per hop."""
    hop = features.hop
    loud = features.loudness_db.astype(np.float64)
    n = len(loud)
    z_loud = robust_z(loud, hop, 1.5)
    z_vocal = robust_z(features.vocal_db, hop, 1.5)
    sm = _smooth(loud, max(1, int(round(2.0 / hop))))
    rise = np.maximum(np.gradient(sm) if n > 1 else np.zeros(n), 0.0)
    z_exc = robust_z(rise, hop, 0.25)
    if chat is not None and len(chat):
        c = np.zeros(n)
        m = min(n, len(chat))
        c[:m] = chat[:m]
        z_chat = robust_z(c, hop, 0.05)
    else:
        chat = None
        z_chat = np.zeros(n)

    w = cfg.weights
    combined = w.loudness * z_loud + w.vocal * z_vocal + w.excitement * z_exc
    if chat is not None:
        combined = combined + cfg.chat.weight * z_chat
    combined = _smooth(combined, max(1, int(round(3.0 / hop))))
    combined = np.where(loud < SILENCE_DB, np.minimum(combined, 0.0), combined)
    comps = {"loudness": z_loud, "vocal": z_vocal, "excitement": z_exc, "chat": z_chat}
    return combined.astype(np.float32), {k: v.astype(np.float32) for k, v in comps.items()}


def _build_window(peak_i: int, score: np.ndarray, loud: np.ndarray, hop: float, cfg: HighlightsConfig,
                  blo: float, bhi: float) -> tuple[float, float] | None:
    """Window around a peak, constrained to the free interval [blo, bhi]. None if it cannot fit."""
    n = len(score)
    if bhi - blo < cfg.min_duration - 1e-9:
        return None
    peak = (peak_i + 0.5) * hop
    start, end = peak - cfg.pre_roll, peak + cfg.post_roll
    # keep extending while the reaction continues
    thr = 0.6 * float(score[peak_i])
    limit = min(start + cfg.max_duration, bhi)
    while end + hop <= limit and score[min(int(end // hop), n - 1)] > thr:
        end += hop
    if end - start > cfg.max_duration:
        end = start + cfg.max_duration
    start, end = max(start, blo), min(end, bhi)
    need = cfg.min_duration - (end - start)
    if need > 0:
        for side in sorted(("l", "r"), key=lambda s: -(start - blo if s == "l" else bhi - end)):
            room = (start - blo) if side == "l" else (bhi - end)
            g = min(need, room)
            if side == "l":
                start -= g
            else:
                end += g
            need -= g
    if end - start < cfg.min_duration - 1e-6 or end - start > cfg.max_duration + 1e-6:
        return None
    # snap cuts to the quietest hop within +-snap_window (only where length limits allow)
    def quiet(t: float, lo: float, hi: float) -> float:
        a, b = int(max(lo, t - cfg.snap_window) // hop), int(min(hi, t + cfg.snap_window) // hop)
        a, b = max(a, 0), min(b, n - 1)
        if b < a:
            return t
        return (int(np.argmin(loud[a:b + 1])) + a) * hop
    s2 = quiet(start, blo, min(peak - hop, bhi))
    if cfg.min_duration - 1e-9 <= end - s2 <= cfg.max_duration + 1e-9 and blo <= s2 < peak:
        start = s2
    e2 = quiet(end, max(peak + hop, blo), bhi)
    if cfg.min_duration - 1e-9 <= e2 - start <= cfg.max_duration + 1e-9 and peak < e2 <= bhi:
        end = e2
    return start, end


def select_highlights(features: AudioFeatures, score: np.ndarray, components: dict[str, np.ndarray],
                      cfg: HighlightsConfig, duration: float) -> list[Highlight]:
    """Greedy peak picking with suppression; windows never overlap."""
    hop = features.hop
    n = len(score)
    duration = float(duration)
    if n == 0 or duration <= 0:
        return []
    loud = features.loudness_db
    if duration < cfg.min_duration:
        if duration < 5.0 or float(np.max(loud)) < SILENCE_DB:
            return []
        pi = int(np.argmax(score))
        return [Highlight(0.0, duration, (pi + 0.5) * hop, float(score[pi]), 1,
                          {k: float(v[pi]) for k, v in components.items() if len(v) > pi})]

    lo, hi = cfg.skip_start, duration - cfg.skip_end
    if hi - lo < cfg.min_duration:
        lo, hi = 0.0, duration
    times = (np.arange(n) + 0.5) * hop
    s = np.where((times >= lo) & (times <= hi), score.astype(np.float64), -np.inf)
    s[loud < SILENCE_DB] = -np.inf

    accepted: list[Highlight] = []
    wins: list[tuple[float, float]] = []

    def suppress(a: float, b: float) -> None:
        s[max(int(a // hop), 0): min(int(b // hop) + 1, n)] = -np.inf

    while len(accepted) < cfg.num_clips:
        i = int(np.argmax(s))
        if not np.isfinite(s[i]) or s[i] <= 0:
            break
        peak = (i + 0.5) * hop
        blo = max([0.0] + [e for _, e in wins if e <= peak])
        bhi = min([duration] + [b for b, _ in wins if b > peak])
        win = _build_window(i, score, loud, hop, cfg, blo, bhi)
        if win is None:
            suppress(peak - cfg.min_gap, peak + cfg.min_gap)
            continue
        start, end = win
        wins.append((start, end))
        sep = max(cfg.min_gap, end - start)
        suppress(peak - sep, peak + sep)
        suppress(start, end)
        accepted.append(Highlight(start, end, peak, float(score[i]), 0,
                                  {k: float(v[i]) for k, v in components.items() if len(v) > i}))
    accepted.sort(key=lambda h: (-h.score, h.start))
    for r, h in enumerate(accepted, 1):
        h.rank = r
    return accepted


def find_highlights(video: Path, duration: float, cfg: HighlightsConfig, cache_dir: Path | None,
                    chat_path: Path | None = None,
                    progress: Callable[[float], None] | None = None
                    ) -> tuple[list[Highlight], AudioFeatures]:
    """Full analysis of one video: audio scan (cached) + optional chat -> ranked highlights."""
    feats = audio.compute_features(video, cfg.hop, cfg.audio_track, cache_dir, progress)
    dur = duration if duration and duration > 0 else feats.duration
    chat = None
    if chat_path is not None and cfg.chat.enabled:
        try:
            ts = chat_mod.load_chat_timestamps(chat_path)
            chat = chat_mod.chat_activity(ts, max(dur, feats.duration), cfg.hop, cfg.chat.window, cfg.chat.delay)
        except ValueError as e:
            log.warning("Ignoring chat file %s: %s", chat_path, e)
    score, comps = score_curve(feats, cfg, chat)
    return select_highlights(feats, score, comps, cfg, dur), feats


def refine_with_transcript(h: Highlight, transcript: Transcript, cfg: HighlightsConfig,
                           duration: float) -> Highlight:
    """Move the cut points to sentence edges (within snap_window), with a little breathing room."""
    segs = [s for s in transcript.segments if s.end > s.start]
    if not cfg.snap_to_sentences or not segs:
        return h

    def nearest(t: float):
        return min(segs, key=lambda s: 0.0 if s.start <= t <= s.end else min(abs(s.start - t), abs(s.end - t)))

    ns, ne = h.start, h.end
    seg = nearest(h.start)
    if abs(seg.start - h.start) <= cfg.snap_window:
        ns = seg.start - 0.15
    seg = nearest(h.end)
    if abs(seg.end - h.end) <= cfg.snap_window:
        ne = seg.end + 0.35
    ns, ne = max(0.0, ns), min(float(duration), ne)
    floor = min(cfg.min_duration, h.duration)

    def ok(s: float, e: float) -> bool:
        return e - s <= cfg.max_duration + 1e-9 and e - s >= floor - 1e-9 and s < h.peak < e

    for s, e in ((ns, ne), (ns, h.end), (h.start, ne)):
        if ok(s, e):
            return dataclasses.replace(h, start=s, end=e)
    return h
