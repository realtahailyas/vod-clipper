"""Chat-log discovery and parsing -> message-activity curve."""
from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..models import parse_time

_JSON_KEYS = ("content_offset_seconds", "time_in_seconds", "offset_seconds", "offset", "seconds",
              "videoOffsetTimeMsec", "t", "time_text", "time", "timestamp")
_LIST_KEYS = ("comments", "messages", "chat", "events", "data", "items")
_CSV_COL = re.compile(r"time|offset|second|^t$", re.I)


def find_chat_file(video: Path) -> Path | None:
    """Look next to the video for a chat log (first match wins)."""
    video = Path(video)
    stem, d = video.stem, video.parent
    for name in (f"{stem}.chat.json", f"{stem}_chat.json", f"{stem}.json",
                 f"{stem}.chat.csv", f"{stem}_chat.csv", f"{stem}.csv"):
        p = d / name
        if p.is_file():
            return p
    return None


def _to_seconds(v: Any, key: str) -> tuple[float, bool] | None:
    """Convert one raw value to (seconds, is_absolute_epoch). None if unusable."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        neg = s.startswith("-")
        try:
            val = parse_time(s.lstrip("-"))
        except ValueError:
            return None
        v = -val if neg else val
    try:
        val = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    if key == "videoOffsetTimeMsec":
        return val / 1000.0, False
    if key in ("time", "timestamp"):
        if val > 1e17:      # nanoseconds since epoch
            return val / 1e9, True
        if val > 1e14:      # microseconds since epoch
            return val / 1e6, True
        if val > 1e11:      # milliseconds since epoch
            return val / 1e3, True
        if val > 1e9:       # seconds since epoch
            return val, True
    return val, False


def _find_key(obj: Any, key: str, depth: int = 0) -> Any:
    if depth > 4 or not isinstance(obj, dict):
        return None
    if key in obj:
        return obj[key]
    for v in obj.values():
        if isinstance(v, dict):
            r = _find_key(v, key, depth + 1)
            if r is not None:
                return r
    return None


def _from_record(rec: Any) -> tuple[float, bool] | None:
    if isinstance(rec, (int, float, str)) and not isinstance(rec, bool):
        return _to_seconds(rec, "t")
    if not isinstance(rec, dict):
        return None
    for key in _JSON_KEYS:
        v = rec.get(key)
        if v is None and key == "videoOffsetTimeMsec":
            v = _find_key(rec, key)
        if v is None:
            continue
        r = _to_seconds(v, key)
        if r is not None:
            return r
    return None


def _records_from_json(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in _LIST_KEYS:
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []


def _finish(pairs: list[tuple[float, bool]], path: Path) -> np.ndarray:
    abs_vals = [v for v, a in pairs if a]
    base = min(abs_vals) if abs_vals else 0.0
    vals = [(v - base) if a else v for v, a in pairs]
    arr = np.asarray([v for v in vals if v >= 0.0], dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"No chat timestamps could be parsed from {path.name}.")
    arr.sort()
    return arr


def _parse_csv(text: str, path: Path) -> np.ndarray:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError(f"{path.name} is empty.") from None
    col = next((i for i, h in enumerate(header) if _CSV_COL.search(h.strip())), None)
    if col is None:
        raise ValueError(f"{path.name}: no time/timestamp/offset/seconds column in header {header}.")
    key = header[col].strip().lower()
    pairs = []
    for row in reader:
        if col < len(row):
            r = _to_seconds(row[col], "timestamp" if key == "timestamp" else "t")
            if r is not None:
                pairs.append(r)
    return _finish(pairs, path)


def load_chat_timestamps(path: Path) -> np.ndarray:
    """Parse a chat log (TwitchDownloader / chat-downloader / generic JSON, JSON-lines or CSV)
    into sorted float64 seconds from VOD start. Raises ValueError if nothing parseable."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        raise ValueError(f"Cannot read chat file {path}: {e}") from e
    if path.suffix.lower() == ".csv":
        return _parse_csv(text, path)
    records: list[Any]
    try:
        records = _records_from_json(json.loads(text))
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.extend(_records_from_json(json.loads(line)) if line[0] == "[" else [json.loads(line)])
            except json.JSONDecodeError:
                continue
    pairs = [r for r in (_from_record(rec) for rec in records) if r is not None]
    return _finish(pairs, path)


def chat_activity(timestamps: np.ndarray, duration: float, hop: float,
                  window: float, delay: float) -> np.ndarray:
    """Messages per hop, moving-averaged over `window` s, shifted EARLIER by `delay` s."""
    n = int(math.ceil(duration / hop)) if duration > 0 else 0
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.float64) - float(delay)
    idx = np.floor(ts / hop).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    counts = np.bincount(idx, minlength=n).astype(np.float64)
    w = min(max(1, int(round(window / hop))), n)
    if w > 1:
        counts = np.convolve(counts, np.full(w, 1.0 / w), mode="same")
    return counts.astype(np.float32)
