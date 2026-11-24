"""Plain data types shared across modules. No heavy imports here."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Highlight:
    """A clip window in SOURCE time (seconds from the start of the VOD)."""
    start: float
    end: float
    peak: float                     # moment of maximum score inside the window
    score: float                    # combined score at the peak (higher = better)
    rank: int = 0                   # 1 = best
    reasons: dict[str, float] = field(default_factory=dict)  # per-feature contribution at the peak

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Word:
    text: str
    start: float                    # SOURCE time, seconds
    end: float
    probability: float = 1.0


@dataclass
class Segment:
    """A sentence-ish unit from the speech recogniser, SOURCE time."""
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Layout:
    """Pixel geometry for one source resolution. Output canvas = out_w x out_h.

    Output bands top->bottom: header [0, header_h), facecam [header_h, header_h+facecam_h),
    gameplay [gameplay_y, out_h). In gameplay_only mode facecam_h == 0 and facecam_crop is None.
    """
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    mode: str                       # split | gameplay_only
    header_h: int
    facecam_h: int
    gameplay_y: int
    gameplay_h: int
    facecam_crop: Rect | None       # source-pixel crop for the facecam
    gameplay_crop: Rect             # source-pixel crop for the gameplay
    gameplay_fit: str               # crop | blur
    caption_y: int                  # output-pixel y of the caption centre line
    background_color: str = "black" # fill for the header band / letterbox areas


# ---------------------------------------------------------------------------
# time helpers

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def parse_time(value: str | float | int) -> float:
    """'1:02:03.5' / '02:03' / '123.4' / 123 -> seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = _TIME_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognised time: {value!r}")
    h, mnt, sec = m.groups()
    return int(h or 0) * 3600 + int(mnt) * 60 + float(sec)


def fmt_time(seconds: float, sep: str = ":") -> str:
    """Seconds -> 'H:MM:SS' (sep='-' gives a filename-safe form)."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}{sep}{m:02d}{sep}{s:02d}"
