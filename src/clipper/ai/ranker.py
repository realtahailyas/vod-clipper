"""AI judge for candidate clips: contact sheet + transcript -> score, hook, category.

The OpenAI SDK is imported lazily and the API key is never logged or printed.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper import media
from clipper.config import AIConfig
from clipper.models import Highlight

log = logging.getLogger(__name__)

CATEGORIES = ("funny", "scare", "rage", "fail", "clutch", "chaos", "wholesome",
              "story", "game_audio", "music", "boring", "other")
TRANSCRIPT_LIMIT = 1500

SYSTEM_PROMPT = (
    "You are a short-form video editor for gaming streams. You score 30-60 second candidate clips "
    "for TikTok, YouTube Shorts and Instagram Reels.\n"
    "What goes viral: the streamer's own strong reactions (screaming, laughing, raging, roasting, "
    "panicking), chaos, fails, clutch plays, funny lines, surprising moments.\n"
    "What does not: game cutscenes or NPC dialogue, music, intro/BRB screens, menus or loading screens, "
    "reading donations, quiet talking, AFK.\n"
    "The image is a contact sheet of frames from the clip in time order, left to right, top to bottom.\n"
    "Score 0-10 for short-form virality (0 = unusable, 5 = okay, 8+ = post immediately). "
    "Set keep=false for anything you would not post.\n"
    "Write a hook: at most 7 words, curiosity-driven, matching what actually happens in the clip, "
    "in English, with no hashtags, emojis or quotes.\n"
    "Give the category, whether the streamer's own voice/reaction is the main thing, and a reason of "
    "at most 20 words.\n"
    "Also report facecam_slot: the message lists numbered positions where the streamer's webcam overlay "
    "(a live camera of a real person) may be. Answer the number of the position where it is visible in "
    "most frames, or 0 if no webcam overlay is visible at any listed position (only game footage, a menu, "
    "or the cam is somewhere else).\n"
    "Also report adult_content: true if ANY frame shows nudity or partial nudity (bare breasts, "
    "bare buttocks, genitals), anyone (man or woman) wearing only underwear, a bikini or a thong, "
    "people in swimwear sunbathing or lying down, a nightclub, club dance floor or party with "
    "revealing outfits, a strip club, stripper, pole dance, lap dance or twerking, sexual acts or "
    "simulated sex, graphic gore (a close-up of a bloody wound, a mutilated body or a bloody corpse), "
    "or if the transcript describes "
    "these happening on screen. The clip will be posted to YouTube, so when in doubt answer true. "
    "Ordinary violence, swearing and normal clothing are not adult_content."
)

STORY_PARAGRAPH = (
    "This is a newly released story-driven game: major story cutscenes, dramatic dialogue, plot reveals, "
    "character introductions and spectacular set pieces ARE valuable clips because viewers want to see them. "
    "Score them on drama, spectacle and curiosity, and do not mark them game_audio or boring (use the "
    "story category). Still prefer moments where the streamer reacts."
)


def build_system_prompt(cfg: AIConfig) -> str:
    """The system prompt, with the story-scenes paragraph appended when ai.story_scenes is on."""
    return SYSTEM_PROMPT + ("\n" + STORY_PARAGRAPH if cfg.story_scenes else "")


VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "0-10 short-form virality"},
        "keep": {"type": "boolean"},
        "hook": {"type": "string", "description": "at most 7 words, no hashtags/quotes/emojis"},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "streamer_reaction": {"type": "boolean"},
        "reason": {"type": "string", "description": "at most 20 words"},
        "facecam_slot": {"type": "integer",
                         "description": "listed webcam position visible in most frames; 0 = none"},
        "adult_content": {"type": "boolean",
                          "description": "nudity, strip club, pole/lap dance or sexual content anywhere"},
    },
    "required": ["score", "keep", "hook", "category", "streamer_reaction", "reason", "facecam_slot",
                 "adult_content"],
    "additionalProperties": False,
}


@dataclass
class AIVerdict:
    score: float            # 0-10 short-form virality
    keep: bool
    hook: str               # <= 7 words, punchy, no hashtags/quotes/emojis
    category: str
    streamer_reaction: bool
    reason: str             # <= 20 words
    facecam_slot: int = 1   # 1..k = which listed cam position is visible, 0 = none (gameplay-only)
    adult_content: bool = False  # nudity / strip club / sexual content -> never posted

    @property
    def facecam_visible(self) -> bool:
        return self.facecam_slot > 0

    def to_dict(self) -> dict:
        return {"score": round(self.score, 2), "keep": self.keep, "hook": self.hook,
                "category": self.category, "streamer_reaction": self.streamer_reaction,
                "reason": self.reason, "facecam_slot": self.facecam_slot,
                "adult_content": self.adult_content}


# Words in a clip's transcript that mean adult content is (probably) on screen. Whole words only,
# case-insensitive. Deliberately strict: a false positive costs one clip, a miss can cost the channel.
ADULT_TERMS = re.compile(
    r"\b(strip ?clubs?|strippers?|stripping|lap ?dances?|pole ?danc\w*|naked|nude|nudes|nudity|"
    r"topless|boobs?|boobies|tits|titties|nipples?|porn\w*|sex|having sex|hookers?|prostitutes?|"
    r"thongs?|onlyfans|twerk\w*|booty ?shak\w*|night ?clubs?|clubbing|bikinis?|underwear|lingerie)\b",
    re.IGNORECASE)


def transcript_is_adult(text: str) -> bool:
    """True when the transcript mentions nudity / strip clubs / sex (see ADULT_TERMS)."""
    return bool(text and ADULT_TERMS.search(text))


def is_blocked(verdict: "AIVerdict | None", transcript: str) -> bool:
    """Adult-content gate: the AI flagged the frames, or the transcript gives it away."""
    return bool(verdict is not None and verdict.adult_content) or transcript_is_adult(transcript)


# ---------------------------------------------------------------------------
# API key / environment


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser: ignores blanks and # comments, strips quotes and `export `."""
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        else:
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
        if key:
            out[key] = val
    return out


def find_api_key(cfg: AIConfig, base_dir: Path) -> str | None:
    """Environment variable first, then a `.env` file in base_dir. Never logged."""
    val = os.environ.get(cfg.api_key_env, "").strip()
    if val:
        return val
    val = parse_env_file(Path(base_dir) / ".env").get(cfg.api_key_env, "").strip()
    return val or None


def openai_installed() -> bool:
    try:
        return importlib.util.find_spec("openai") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# pure helpers


def candidate_pool(num_clips: int, cfg: AIConfig) -> int:
    """How many candidates the AI judges: ceil(n * multiplier), capped by max_candidates."""
    return min(cfg.max_candidates, math.ceil(num_clips * cfg.candidate_multiplier))


def final_scores(audio: list[float], verdicts: list[AIVerdict | None], weight: float) -> list[float]:
    """weight*ai + (1-weight)*audio, both 0-10. Audio is normalised by the pool maximum.
    A candidate without a verdict gets its audio-only score."""
    top = max(audio, default=0.0)
    out: list[float] = []
    for a, v in zip(audio, verdicts):
        a10 = 10.0 * a / top if top > 0 else 0.0
        out.append(a10 if v is None else weight * v.score + (1.0 - weight) * a10)
    return out


def select_final(candidates: list[Highlight], verdicts: list[AIVerdict | None], n: int,
                 cfg: AIConfig, blocked: set[int] | None = None) -> list[tuple[int, bool]]:
    """(index, weak) pairs, best final score first (at most n).

    Candidates at/above min_score (or without a verdict) come first with weak=False. With cfg.fill
    the remaining slots are topped up from the rest by final score, flagged weak=True.
    Indices in `blocked` (adult content) are never selected, not even to fill."""
    blocked = blocked or set()
    finals = final_scores([c.score for c in candidates], verdicts, cfg.weight)
    good = [i for i, v in enumerate(verdicts)
            if i not in blocked and (v is None or v.score >= cfg.min_score)]
    good.sort(key=lambda i: finals[i], reverse=True)
    out = [(i, False) for i in good[:n]]
    if cfg.fill and len(out) < n:
        taken = set(good) | blocked
        rest = sorted((i for i in range(len(verdicts)) if i not in taken),
                      key=lambda i: finals[i], reverse=True)
        out += [(i, True) for i in rest[:n - len(out)]]
    return out


def build_contact_sheet(video: Path, start: float, end: float, out_jpg: Path,
                        frames: int = 4, width: int = 640) -> Path:
    """One ffmpeg call: `frames` evenly spaced stills from [start, end] tiled into a grid JPEG."""
    n = max(1, int(frames))
    dur = max(end - start, 0.1)
    lo, hi = (start + 1.0, end - 1.0) if dur > 4.0 else (start, end - 0.1)
    times = [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]
    cols = math.ceil(math.sqrt(n))
    cmd = [media.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"]
    for t in times:
        cmd += ["-ss", f"{max(t, 0.0):.3f}", "-i", str(video)]
    parts = [f"[{i}:v]scale={width}:-2,setsar=1[v{i}]" for i in range(n)]
    if n == 1:
        parts = [f"[0:v]scale={width}:-2,setsar=1[o]"]
    else:
        layout = "|".join(
            f"{'+'.join(['w0'] * (i % cols)) or '0'}_{'+'.join(['h0'] * (i // cols)) or '0'}"
            for i in range(n))
        ins = "".join(f"[v{i}]" for i in range(n))
        parts.append(f"{ins}xstack=inputs={n}:layout={layout}:fill=black[o]")
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["-filter_complex", ";".join(parts), "-map", "[o]", "-frames:v", "1", "-q:v", "4", str(out_jpg)]
    media.run(cmd, desc="contact sheet")
    return out_jpg


# ---------------------------------------------------------------------------
# parsing


HOOK_MAX_WORDS = 10  # prompt asks for <= 7; this only guards against runaway output


def _clean_hook(text: str) -> str:
    text = re.sub(r"#\w+", "", text)
    text = re.sub("[‘’`]", "'", text)                    # keep apostrophes: HE'S, COULDN'T
    text = re.sub("[\"“”]", "", text)                    # drop surrounding quotes
    text = "".join(ch for ch in text if ord(ch) < 0x2190)  # drop emoji / pictographs
    text = text.strip(" '")
    words = text.split()
    if len(words) > HOOK_MAX_WORDS:                     # rare: cut at the limit rather than mid-clause
        words = words[:HOOK_MAX_WORDS]
    return " ".join(words)


def _slot(d: dict) -> int:
    """facecam_slot from the model; legacy answers fall back to facecam_visible; default slot 1."""
    s = d.get("facecam_slot")
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return max(0, int(s))
    if "facecam_visible" in d:
        return 1 if d["facecam_visible"] else 0
    return 1


def parse_verdict(text: str) -> AIVerdict | None:
    """Parse the model's JSON; None when it is not valid or fields are missing."""
    try:
        d = json.loads(text)
        if not isinstance(d, dict):
            return None
        score = d["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        cat = str(d["category"]).strip().lower()
        return AIVerdict(
            score=max(0.0, min(10.0, float(score))),
            keep=bool(d["keep"]),
            hook=_clean_hook(str(d["hook"])),
            category=cat if cat in CATEGORIES else "other",
            streamer_reaction=bool(d["streamer_reaction"]),
            reason=" ".join(str(d["reason"]).split()[:20]),
            facecam_slot=_slot(d),
            adult_content=d.get("adult_content") is True,
        )
    except (ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# ranker


class AIRanker:
    def __init__(self, cfg: AIConfig, base_dir: Path, client: Any = None) -> None:
        self.cfg = cfg
        self.base_dir = Path(base_dir)
        self.available = False
        self.unavailable_reason = ""
        self.spent_usd = 0.0
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()
        self._client = client
        if client is not None:          # injected (tests)
            self.available = True
            return
        if not openai_installed():
            self.unavailable_reason = "the 'openai' package is not installed (pip install -e .[ai])"
            return
        key = find_api_key(cfg, self.base_dir)
        if not key:
            self.unavailable_reason = f"no API key found (set {cfg.api_key_env} or add it to .env)"
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key, max_retries=5)  # flaky home networks: retry with backoff
        except Exception as e:  # noqa: BLE001
            self.unavailable_reason = f"could not create the OpenAI client ({type(e).__name__})"
            return
        self.available = True

    # -- the only place that talks to the SDK -------------------------------
    def _call(self, sheet: Path | None, user_text: str) -> tuple[str, int, int]:
        content: list[dict] = [{"type": "input_text", "text": user_text}]
        if sheet is not None:
            b64 = base64.b64encode(Path(sheet).read_bytes()).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}",
                            "detail": self.cfg.image_detail})
        resp = self._client.responses.create(
            model=self.cfg.model,
            reasoning={"effort": self.cfg.reasoning_effort},
            input=[{"role": "system", "content": build_system_prompt(self.cfg)},
                   {"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "clip_verdict",
                             "strict": True, "schema": VERDICT_SCHEMA}},
            timeout=self.cfg.timeout,
        )
        usage = getattr(resp, "usage", None)
        return (resp.output_text,
                int(getattr(usage, "input_tokens", 0) or 0),
                int(getattr(usage, "output_tokens", 0) or 0))

    def _user_text(self, transcript: str, meta: dict) -> str:
        t = " ".join((transcript or "").split())
        if len(t) > TRANSCRIPT_LIMIT:
            t = t[:TRANSCRIPT_LIMIT] + "..."
        lines = []
        if self.cfg.context:
            lines.append(f"Context: {self.cfg.context}")
        lines.append(f"Video: {meta.get('video', '?')}")
        lines.append(f"Timestamp: {meta.get('timestamp', '?')}  Duration: {meta.get('duration', '?')} s")
        if meta.get("facecam_slots"):
            lines.append("Possible webcam positions: " + "; ".join(
                f"{i}) {s}" for i, s in enumerate(meta["facecam_slots"], 1)))
        if meta.get("signals"):
            lines.append(f"Audio signal: {meta['signals']}")
        lines.append(f"Transcript: {t or '(none / unintelligible)'}")
        return "\n".join(lines)

    def judge(self, sheet: Path | None, transcript: str, meta: dict) -> AIVerdict | None:
        """Thread-safe. None on any error or once the budget is reached."""
        if not self.available:
            return None
        with self._lock:
            if self.spent_usd >= self.cfg.max_cost_usd:
                return None
            self.calls += 1
        try:
            text, tin, tout = self._call(sheet, self._user_text(transcript, meta))
        except Exception as e:  # noqa: BLE001
            msg = (str(e).strip().splitlines() or [""])[0][:200]
            log.warning("AI call failed (%s): %s", type(e).__name__, msg)
            with self._lock:
                self.failures += 1
            return None
        cost = (tin * self.cfg.price_input_per_m + tout * self.cfg.price_output_per_m) / 1e6
        verdict = parse_verdict(text)
        with self._lock:
            self.spent_usd += cost
            if verdict is None:
                self.failures += 1
        return verdict
