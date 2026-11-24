"""ASS subtitle generation: word-by-word animated captions plus an optional title."""
from __future__ import annotations

import re

from .config import CaptionsConfig, TitleConfig, WatermarkConfig
from .models import Layout, Transcript, Word

SENTENCE_END = (".", "?", "!")
MAX_GAP = 0.6          # seconds of silence that forces a new chunk
LINGER = 0.6           # max seconds a chunk stays after its last word


def ass_color(hex_color: str) -> str:
    """'#RRGGBB' -> '&H00BBGGRR' (style colour)."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise ValueError(f"Bad colour {hex_color!r}, expected #RRGGBB")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _tag_color(hex_color: str) -> str:
    """'#RRGGBB' -> '&HBBGGRR&' (override-tag colour)."""
    return "&H" + ass_color(hex_color)[4:] + "&"


def ass_time(t: float) -> str:
    """Seconds -> H:MM:SS.cc"""
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


# On-screen swearing (above all in the first seconds) is a common reason YouTube age-restricts a
# Short, which kills its reach. Slurs are masked completely; strong swears keep the first letter.
_SLUR = re.compile(r"\bn[i1!]gg(?:a|as|az|ah|uh|er|ers)?\b", re.IGNORECASE)
_SWEAR = re.compile(
    r"\b(?:(?:mother|muther)?f+u+c*k+\w*|f+u+c+\w*|fk|fck\w*|wtf|bullsh[i1]t\w*|sh[i1]t+\w*|b[i1]tch\w*|"
    r"cunt\w*|puss(?:y|ies)|d[i1]ck(?:s|head\w*)?|wh[o0]re\w*|sluts?|fagg?ots?|f[a@]gs?|"
    r"assholes?|bastards?)\b", re.IGNORECASE)


def censor_text(text: str) -> str:
    """Mask slurs ('N****') and strong swears ('F***', 'Sh*t') in caption text; other words unchanged."""
    text = _SLUR.sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), text)
    def mask(m: re.Match) -> str:
        w = m.group(0)
        if w.lower() == "wtf":
            return w[0] + "*" + w[2]
        keep = 2 if w[:2].lower() == "sh" else 1
        return w[:keep] + "*" * (len(w) - keep)
    return _SWEAR.sub(mask, text)


def escape_text(text: str) -> str:
    """Escape ASS special characters; drop line breaks."""
    text = text.replace("\r", " ").replace("\n", " ")
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def chunk_words(words: list[Word], max_words: int, max_chars: int) -> list[list[Word]]:
    """Group words into on-screen chunks."""
    max_words = max(1, max_words)
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    chars = 0
    for w in words:
        if cur:
            prev = cur[-1]
            new_chars = chars + 1 + len(w.text)
            if (len(cur) >= max_words or new_chars > max_chars
                    or w.start - prev.end > MAX_GAP or prev.text.endswith(SENTENCE_END)):
                chunks.append(cur)
                cur, chars = [], 0
        chars = chars + 1 + len(w.text) if cur else len(w.text)
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def _style_line(name: str, font: str, size: int, primary: str, outline_c: str, outline: float,
                shadow: float, bold: int = -1) -> str:
    # Name,Fontname,Fontsize,Primary,Secondary,Outline,Back,Bold,Italic,Underline,StrikeOut,
    # ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
    return (f"Style: {name},{font},{size},{ass_color(primary)},{ass_color(primary)},"
            f"{ass_color(outline_c)},&H64000000,{bold},0,0,0,100,100,0,0,1,{outline:g},{shadow:g},"
            f"5,20,20,20,1")


def _wrap_title(text: str, font_size: int, out_w: int, header_h: int) -> tuple[str, int]:
    """Return (text with optional \\N, font size) so the title fits ~90% width, max 2 lines."""
    max_w = 0.9 * out_w
    est = lambda n, fs: 0.6 * fs * n  # noqa: E731
    lines = [text]
    if est(len(text), font_size) > max_w and " " in text:
        spaces = [i for i, c in enumerate(text) if c == " "]
        mid = len(text) / 2.0
        cut = min(spaces, key=lambda i: abs(i - mid))
        lines = [text[:cut].strip(), text[cut + 1:].strip()]
    longest = max(len(line) for line in lines)
    fs = font_size
    if est(longest, fs) > max_w:
        fs = max(10, int(max_w / (0.6 * longest)))
    if header_h > 0:
        fs = min(fs, max(10, int(header_h / (1.2 * len(lines)))))
    return "\\N".join(lines), fs


def ass_alpha(opacity: float) -> str:
    """Opacity 0..1 -> ASS alpha byte as '&HAA' (00 = opaque, FF = transparent)."""
    o = min(1.0, max(0.0, opacity))
    return f"&H{round((1 - o) * 255):02X}"


def watermark_position(layout: Layout, wm: WatermarkConfig,
                       captions: CaptionsConfig) -> tuple[int, int, int]:
    """Return (x, y, alignment) for the watermark. Alignment 3 = bottom-right anchor,
    6 = right-middle (header_right)."""
    out_w, out_h, m = layout.out_w, layout.out_h, wm.margin
    x = out_w - m
    pos = wm.position
    if pos == "custom":
        return int(wm.x), int(wm.y), 3
    if pos == "facecam_bottom_right" and layout.facecam_h <= 0:
        pos = "bottom_right"
    if pos == "header_right" and layout.header_h <= 0:
        pos = "facecam_top_right"
    if pos == "facecam_bottom_right":
        seam = layout.header_h + layout.facecam_h
        y = seam - m
        if abs(layout.caption_y - seam) < captions.font_size * 1.5:
            y = min(y, int(round(layout.caption_y - captions.font_size * 0.75 - 10)))
        return x, int(round(y)), 3
    if pos == "facecam_top_right":
        return x, int(round(layout.header_h + m + wm.font_size)), 3
    if pos == "header_right":
        return x, int(round(layout.header_h / 2)), 6
    # bottom_right: stays above the platform bottom UI
    return x, int(round(out_h - 0.20 * out_h)), 3


def build_ass(transcript: Transcript | None, clip_start: float, clip_end: float, layout: Layout,
              captions: CaptionsConfig, title: TitleConfig | None = None,
              title_text: str = "", watermark: WatermarkConfig | None = None) -> str:
    """Build the full .ass file text. Times in the file are relative to clip_start."""
    out_w, out_h = layout.out_w, layout.out_h
    dur = max(0.0, clip_end - clip_start)
    tcfg = title or TitleConfig()

    wm_on = bool(watermark and watermark.enabled and watermark.text.strip())

    lines = [
        "[Script Info]",
        "; Generated by clipper",
        "ScriptType: v4.00+",
        f"PlayResX: {out_w}",
        f"PlayResY: {out_h}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: None",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        _style_line("Caption", captions.font, captions.font_size, captions.primary_color,
                    captions.outline_color, captions.outline, captions.shadow),
        _style_line("Title", tcfg.font, tcfg.font_size, tcfg.color, tcfg.outline_color,
                    tcfg.outline, 0),
    ]
    if wm_on:
        a = ass_alpha(watermark.opacity)

        def c(hx: str) -> str:
            return "&H" + a[2:] + ass_color(hx)[4:]

        lines.append(f"Style: Watermark,{watermark.font},{watermark.font_size},{c(watermark.color)},"
                     f"{c(watermark.color)},{c(watermark.outline_color)},{c('#000000')},-1,0,0,0,100,100,0,0,1,"
                     f"{watermark.outline:g},{watermark.shadow:g},3,20,20,20,1")
    lines += [
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def dialogue(layer: int, start: float, end: float, style: str, text: str) -> str | None:
        if int(round(end * 100)) <= int(round(start * 100)):
            return None
        return f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"

    events: list[str] = []

    # --- title -----------------------------------------------------------
    if title_text.strip():
        txt = title_text.strip()
        if tcfg.uppercase:
            txt = txt.upper()
        txt = escape_text(txt)
        wrapped, fs = _wrap_title(txt, tcfg.font_size, out_w, layout.header_h)
        ty = layout.header_h / 2.0 if layout.header_h > 0 else 0.06 * out_h
        tag = f"{{\\an5\\pos({out_w // 2},{int(round(ty))})\\fs{fs}}}"
        d = dialogue(0, 0.0, dur, "Title", tag + wrapped)
        if d:
            events.append(d)

    # --- captions --------------------------------------------------------
    if captions.enabled and transcript is not None:
        rel: list[Word] = []
        for w in transcript.words:
            text = w.text.strip()
            if not text:
                continue
            s, e = w.start - clip_start, w.end - clip_start
            if e <= 0 or s >= dur:
                continue
            s = max(0.0, s)
            e = min(dur, e)
            if e <= s:
                e = min(dur, s + 0.02)
            rel.append(Word(text, s, e, w.probability))

        chunks = chunk_words(rel, captions.max_words, captions.max_chars)
        pos = f"{{\\an5\\pos({out_w // 2},{layout.caption_y})}}"
        hl = _tag_color(captions.highlight_color)
        if captions.pop:
            active_tag = (f"{{\\c{hl}\\fscx100\\fscy100\\t(0,90,\\fscx118\\fscy118)"
                          f"\\t(90,180,\\fscx108\\fscy108)}}")
        else:
            active_tag = f"{{\\c{hl}}}"

        for k, chunk in enumerate(chunks):
            c_start = chunk[0].start
            limit = chunk[-1].end + LINGER
            if k + 1 < len(chunks):
                limit = min(limit, chunks[k + 1][0].start)
            c_end = min(limit, dur)
            if c_end <= c_start:
                continue
            starts = [c_start]
            for w in chunk[1:]:
                starts.append(min(max(w.start, starts[-1]), c_end))
            ends = starts[1:] + [c_end]
            texts = []
            for w in chunk:
                t = censor_text(w.text) if captions.censor else w.text
                t = t.upper() if captions.uppercase else t
                texts.append(escape_text(t))
            for j in range(len(chunk)):
                parts = [(active_tag + texts[i] + "{\\r}") if i == j else texts[i]
                         for i in range(len(chunk))]
                d = dialogue(1, starts[j], ends[j], "Caption", pos + " ".join(parts))
                if d:
                    events.append(d)

    # --- watermark (layer 5: above captions) -------------------------------
    if wm_on:
        wx, wy, an = watermark_position(layout, watermark, captions)
        d = dialogue(5, 0.0, dur, "Watermark",
                     f"{{\\an{an}\\pos({wx},{wy})}}" + escape_text(watermark.text.strip()))
        if d:
            events.append(d)

    lines.extend(events)
    return "\n".join(lines) + "\n"
