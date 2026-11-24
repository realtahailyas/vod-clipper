"""Package rendered clips into a ready-to-post folder tree (YouTube Shorts) plus a manifest.

Layout: <root>/<creator>/<NNN - hook>/{video.mp4, title.txt,
description.txt, tags.txt, captions.srt, metadata.json}. Runs append (running number per creator
folder) and re-exporting the same clip never duplicates it (id = sha1(source video + start)).

The OpenAI SDK is used lazily, only in MetadataWriter._call; the API key is never logged.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from clipper import __version__, media
from clipper.ai.ranker import find_api_key, openai_installed
from clipper.captions import censor_text, chunk_words
from clipper.config import AIConfig, Config, PublishConfig
from clipper.models import Word

log = logging.getLogger(__name__)

TITLE_MAX = 70
TAG_MAX_LEN = 30
TAGS_MAX_TOTAL = 450
TAGS_MAX_COUNT = 15
FOLDER_MAX = 80
TRANSCRIPT_LIMIT = 1200
SRT_MAX_WORDS = 8
SRT_MAX_CHARS = 42

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

CSV_COLUMNS = ["id", "status", "folder", "platform", "game", "creator", "title", "hook", "category",
               "ai_score", "duration", "source_video", "source_start", "source_end", "video_file",
               "captions_file", "created_at", "posted_at", "url"]


# ---------------------------------------------------------------------------
# naming helpers


def safe_name(text: str, max_len: int = FOLDER_MAX, default: str = "untitled") -> str:
    """Windows-safe file/folder name: illegal chars removed, no trailing dots/spaces, length capped."""
    t = _ILLEGAL.sub("", text or "")
    t = " ".join(t.split())
    t = t[:max_len].rstrip(" .")
    if not t:
        return default
    if t.split(".")[0].upper() in _RESERVED:
        t = "_" + t
    return t


def clip_id(source_video: str, start: float) -> str:
    return hashlib.sha1(f"{source_video}|{start:.3f}".encode("utf-8")).hexdigest()[:12]


def _hashtag(name: str) -> str:
    core = re.sub(r"[^0-9A-Za-z]", "", name or "")
    return f"#{core}" if core else ""


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for i in items:
        k = i.casefold()
        if i and k not in seen:
            seen.add(k)
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# captions


def _srt_time(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcript_to_srt(words: list[Word], clip_start: float, censor: bool = False) -> str:
    """Standard SRT from word-timed text; cue times are relative to clip_start."""
    if censor:
        words = [dataclasses.replace(w, text=censor_text(w.text)) for w in words]
    chunks = chunk_words(list(words), SRT_MAX_WORDS, SRT_MAX_CHARS)
    out: list[str] = []
    for i, ch in enumerate(chunks):
        start = max(0.0, ch[0].start - clip_start)
        end = max(0.0, ch[-1].end - clip_start)
        if i + 1 < len(chunks):                      # never overlap the next cue
            end = min(end, max(0.0, chunks[i + 1][0].start - clip_start))
        end = max(end, start + 0.3) if i + 1 == len(chunks) else max(end, start + 0.05)
        out.append(f"{i + 1}\n{_srt_time(start)} --> {_srt_time(end)}\n{' '.join(w.text for w in ch)}\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# metadata


METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "at most 70 characters, no hashtags"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "10-15 search phrases"},
    },
    "required": ["title", "description", "tags"],
    "additionalProperties": False,
}

METADATA_SYSTEM_PROMPT = (
    "You are a YouTube Shorts SEO writer for gaming clips.\n"
    "Title: at most 70 characters, a curiosity hook that matches what actually happens in the clip, "
    "include the creator name and the game, no hashtags in the title, no ALL CAPS sentences, no false claims.\n"
    "Description: 2 short lines about the moment (the first line repeats the hook naturally), then a blank line, "
    "then '🎥 Credit: <creator> — <creator_url>' (omit the url part if there is none), then "
    "'Subscribe for daily <game> highlights: <channel>', then one line of 3-5 hashtags that must include "
    "#Shorts, the game as a hashtag (like #GTA6) and the creator (like #xQc).\n"
    "Tags: 10-15 search phrases (creator, game, 'gta 6 funny moments' style), each at most 30 characters, "
    "total at most 450 characters."
)


def _user_text(info: dict) -> str:
    t = " ".join((info.get("transcript") or "").split())
    if len(t) > TRANSCRIPT_LIMIT:
        t = t[:TRANSCRIPT_LIMIT] + "..."
    return "\n".join([
        f"Creator: {info.get('creator', '')}",
        f"Creator URL: {info.get('creator_url', '') or '(none)'}",
        f"Game: {info.get('game', '') or '(unknown)'}",
        f"Channel: {info.get('channel', '')}",
        f"Hook: {info.get('hook', '') or '(none)'}",
        f"Category: {info.get('category', '') or '(none)'}",
        f"Why it is good: {info.get('reason', '') or '(none)'}",
        f"Transcript: {t or '(none / unintelligible)'}",
    ])


class MetadataWriter:
    """Writes title/description/tags with the same OpenAI setup as AIRanker; one budget per run."""

    def __init__(self, ai_cfg: AIConfig, base_dir: Path, client: Any = None) -> None:
        self.cfg = ai_cfg
        self.available = False
        self.unavailable_reason = ""
        self.spent_usd = 0.0
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()
        self._client = client
        if client is not None:
            self.available = True
            return
        if ai_cfg.enabled == "false":
            self.unavailable_reason = "ai.enabled is false"
            return
        if not openai_installed():
            self.unavailable_reason = "the 'openai' package is not installed"
            return
        key = find_api_key(ai_cfg, Path(base_dir))
        if not key:
            self.unavailable_reason = f"no API key found (set {ai_cfg.api_key_env} or add it to .env)"
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key, max_retries=5)
        except Exception as e:  # noqa: BLE001
            self.unavailable_reason = f"could not create the OpenAI client ({type(e).__name__})"
            return
        self.available = True

    def _call(self, user_text: str) -> tuple[str, int, int]:      # the only SDK call
        resp = self._client.responses.create(
            model=self.cfg.model,
            reasoning={"effort": self.cfg.reasoning_effort},
            input=[{"role": "system", "content": METADATA_SYSTEM_PROMPT},
                   {"role": "user", "content": [{"type": "input_text", "text": user_text}]}],
            text={"format": {"type": "json_schema", "name": "shorts_metadata",
                             "strict": True, "schema": METADATA_SCHEMA}},
            timeout=self.cfg.timeout,
        )
        usage = getattr(resp, "usage", None)
        return (resp.output_text,
                int(getattr(usage, "input_tokens", 0) or 0),
                int(getattr(usage, "output_tokens", 0) or 0))

    def write(self, info: dict) -> dict | None:
        """{title, description, tags} or None (error, bad JSON, budget reached)."""
        if not self.available:
            return None
        with self._lock:
            if self.spent_usd >= self.cfg.max_cost_usd:
                return None
            self.calls += 1
        try:
            text, tin, tout = self._call(_user_text(info))
        except Exception as e:  # noqa: BLE001
            log.warning("AI metadata call failed (%s): %s", type(e).__name__,
                        (str(e).strip().splitlines() or [""])[0][:200])
            with self._lock:
                self.failures += 1
            return None
        cost = (tin * self.cfg.price_input_per_m + tout * self.cfg.price_output_per_m) / 1e6
        try:
            d = json.loads(text)
            ok = (isinstance(d, dict) and isinstance(d["title"], str) and isinstance(d["description"], str)
                  and isinstance(d["tags"], list))
        except (ValueError, KeyError, TypeError):
            ok = False
        with self._lock:
            self.spent_usd += cost
            if not ok:
                self.failures += 1
        return d if ok else None


def _clean_title(title: str) -> str:
    t = re.sub(r"#\w+", "", title or "")
    t = re.sub("[‘’`]", "'", t)
    t = re.sub("[\"“”]", "", t)
    t = " ".join(t.split()).strip(" '|-–—:,")
    letters = [c for c in t if c.isalpha()]
    if len(letters) > 3 and all(c.isupper() for c in letters):     # no ALL CAPS sentences
        t = " ".join(w.capitalize() for w in t.split())
    if len(t) > TITLE_MAX:
        cut = t[:TITLE_MAX]
        if " " in cut and t[TITLE_MAX] != " ":
            cut = cut.rsplit(" ", 1)[0]
        t = cut.rstrip(" '|-–—:,&")
    return t


def _hashtags_for(cfg: PublishConfig) -> list[str]:
    req = list(cfg.hashtags) + ["#Shorts", _hashtag(cfg.game), _hashtag(cfg.creator)]
    return _dedupe([h if h.startswith("#") else "#" + h for h in req if h and h.strip("#")])


def _clean_tags(tags: list[Any]) -> list[str]:
    out: list[str] = []
    total = 0
    for raw in _dedupe([" ".join(str(t).replace("#", " ").replace(",", " ").split()) for t in tags]):
        if len(raw) > TAG_MAX_LEN:
            continue
        add = len(raw) + (1 if out else 0)
        if len(out) >= TAGS_MAX_COUNT or total + add > TAGS_MAX_TOTAL:
            break
        out.append(raw)
        total += add
    return out


def _build_description(body: list[str], cfg: PublishConfig, extra_tags: list[str]) -> str:
    credit = f"🎥 Credit: {cfg.creator}" + (f" — {cfg.creator_url}" if cfg.creator_url else "")
    sub = f"Subscribe for daily {cfg.game + ' ' if cfg.game else ''}highlights: {cfg.channel}"
    req = _hashtags_for(cfg)
    tags = _dedupe(req + [t for t in extra_tags if t not in req])[:max(5, len(req))]
    cta = [cfg.cta.strip()] if cfg.cta.strip() else []
    return "\n".join([*body[:2], "", credit, *cta, sub, " ".join(tags)])


def _template_tags(info: dict, cfg: PublishConfig) -> list[str]:
    c, g = cfg.creator, cfg.game
    cat = (info.get("category") or "").replace("_", " ")
    cand = [c, g, f"{c} {g}".strip(), f"{c} highlights", f"{c} clips", "gaming clips", "gaming shorts",
            "funny moments", "stream highlights"]
    if g:
        cand += [f"{g} funny moments", f"{g} highlights", f"{g} clips", f"{g} gameplay", f"{g} reaction",
                 f"{g} shorts"]
    if cat and cat not in ("other", "boring", "game audio", "music"):
        cand += [f"{c} {cat}", f"{g} {cat}".strip()]
    return cand


def generate_metadata(info: dict, cfg: PublishConfig, ai_cfg: AIConfig, base_dir: Path,
                      writer: MetadataWriter | None = None) -> dict:
    """{title, description, tags}. AI when available (cfg.ai_metadata), else a template; always post-processed."""
    ai: dict | None = None
    if cfg.ai_metadata:
        if writer is None:
            writer = MetadataWriter(ai_cfg, base_dir)
        ai = writer.write({**info, "creator": cfg.creator, "creator_url": cfg.creator_url,
                           "game": cfg.game, "channel": cfg.channel})
    hook = _clean_title(info.get("hook") or "")
    suffix = " ".join(x for x in (cfg.creator, cfg.game) if x)
    if ai is not None:
        title = _clean_title(ai["title"])
        body = [ln.strip() for ln in str(ai["description"]).splitlines()
                if ln.strip() and "credit:" not in ln.lower() and not ln.lower().startswith("subscribe")
                and not all(tok.startswith("#") for tok in ln.split())]
        tags = _clean_tags([*ai["tags"], *_template_tags(info, cfg)])
        extra_hash = [t for t in re.findall(r"#\w+", str(ai["description"]))]
    else:
        title = ""
        body = []
        tags = _clean_tags(_template_tags(info, cfg))
        extra_hash = []
    if not title:
        title = _clean_title(f"{hook} | {suffix}" if hook and suffix else hook or f"{suffix} moment".strip())
    if not body:
        reason = " ".join((info.get("reason") or "").split())
        body = [hook or title] + ([reason] if reason else [f"{suffix} highlight." if suffix else "Stream highlight."])
    return {"title": title or "Stream highlight", "description": _build_description(body, cfg, extra_hash),
            "tags": tags}


# ---------------------------------------------------------------------------
# manifest / handover


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_manifest(auto_dir: Path) -> list[dict]:
    try:
        data = json.loads((auto_dir / "manifest.json").read_text(encoding="utf-8"))
        return [d for d in data if isinstance(d, dict) and "id" in d] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def write_manifest(auto_dir: Path, packages: list[dict]) -> list[dict]:
    """Merge into manifest.json/.csv by id (new fields win, but status/posted_at/url of an existing
    entry are kept unless the new package carries them), sorted by created_at."""
    merged = {d["id"]: d for d in load_manifest(auto_dir)}
    for p in packages:
        merged[p["id"]] = {**merged.get(p["id"], {}), **p}
    rows = sorted(merged.values(), key=lambda d: (d.get("created_at", ""), d.get("folder", "")))
    _atomic_write(auto_dir / "manifest.json", json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    _atomic_write(auto_dir / "manifest.csv", buf.getvalue())
    return rows


HANDOVER = """# Publishing handover

@@DRIVE@@Packages here are for the YouTube channel @@CHANNEL@@ only. All paths below (and every `folder` value in
the manifest) are relative to this folder. It holds finished vertical clips (YouTube Shorts) ready to post.
Written by `clipper export`.

## Structure

    publish/ (this folder)
      manifest.json  manifest.csv      one entry per package (the index: start here)
      HANDOVER.md                      this file
      <creator>/<NNN - hook>/
        <title>.mp4  title.txt  description.txt  tags.txt  captions.srt  metadata.json

`NNN` is a running number per creator folder. New runs append; nothing is overwritten. Each clip has a
stable `id` (sha1 of source video + start time), so exporting the same clip twice never duplicates it.

## manifest.json

A list of objects: id, status, folder (relative to this folder, forward slashes, e.g. "xQc/001 - ..."),
platform, game, creator, creator_url, channel, title, description, tags, hashtags, video_file,
captions_file, source_video, source_start, source_end, duration, ai_score, category, hook, weak,
created_at, tool_version. `<folder>/metadata.json` holds the same data for one package.

## Status values

- `ready`: not posted yet. Pick these.
- `posted`: uploaded. When you post a package, set in BOTH `<folder>/metadata.json` and manifest.json:
  `status: "posted"`, `posted_at` (ISO timestamp) and `url` (the video link). manifest.csv is regenerated
  from manifest.json by the next export; update it too if you read it.

## Posting checklist (YouTube Shorts)

1. Video: the `.mp4` in the package (vertical, 60 seconds or less; check `duration`).
2. Title: the text in `title.txt`.
3. Description: the text in `description.txt` (keep the credit line and the hashtags as they are).
4. Tags: `tags.txt` (comma separated).
5. Subtitles: upload `captions.srt` as English subtitles.
6. Audience: made for kids = No.
7. Keep the credit line to the original creator. Do not remove it.
8. After upload, set status/posted_at/url as described above.
"""


def handover_text(pc) -> str:
    """HANDOVER.md for one package root; names its own Drive mirror and channel, never another flow's."""
    drive = (f'This folder is mirrored on Google Drive at "My Drive/{pc.drive_folder.strip("/")}" '
             "(same creator/video sub-folders). ") if pc.drive_folder.strip() else ""
    return HANDOVER.replace("@@DRIVE@@", drive).replace("@@CHANNEL@@", pc.channel or "(not set)")


# ---------------------------------------------------------------------------
# packaging


def _next_number(creator_dir: Path) -> int:
    best = 0
    if creator_dir.is_dir():
        for d in creator_dir.iterdir():
            m = re.match(r"^(\d+) - ", d.name)
            if m and d.is_dir():
                best = max(best, int(m.group(1)))
    return best + 1


def _clip_srt(mp4: Path, cfg: Config, console: Console, transcriber: Any, holder: dict) -> str:
    """captions.srt text: a .srt next to the mp4 if present, else transcribe the rendered clip."""
    existing = mp4.with_suffix(".srt")
    if existing.is_file():
        return existing.read_text(encoding="utf-8")
    try:
        if transcriber is None:
            transcriber = holder.get("t")
            if transcriber is None:
                from clipper.transcribe import Transcriber
                transcriber = holder["t"] = Transcriber(cfg.transcription)
        with tempfile.TemporaryDirectory(prefix="clipper_srt_") as d:
            wav = media.extract_audio(mp4, 0.0, media.probe(mp4).duration, Path(d) / "a.wav",
                                      track=cfg.render.audio_track)
            tr = transcriber.transcribe(wav, offset=0.0)
        return transcript_to_srt(tr.words, 0.0, cfg.captions.censor)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [yellow]![/yellow] captions for {mp4.name} failed: {(str(e).splitlines() or [''])[0][:120]}")
        return ""


def _package_clip(clip: dict, mp4: Path, source_video: str, cfg: Config, console: Console,
                  auto_dir: Path, writer: MetadataWriter | None, transcriber: Any,
                  holder: dict) -> dict:
    pc = cfg.publish
    info = {"hook": clip.get("hook") or "", "category": clip.get("category") or "",
            "reason": clip.get("reason") or "", "transcript": clip.get("transcript") or ""}
    meta = generate_metadata(info, pc, cfg.ai, cfg.base_dir, writer)
    creator_dir = auto_dir / safe_name(pc.creator)
    n = _next_number(creator_dir)
    folder = creator_dir / f"{n:03d} - {safe_name(clip.get('hook') or meta['title'], 60)}"
    folder.mkdir(parents=True, exist_ok=True)
    video_name = safe_name(meta["title"], FOLDER_MAX) + ".mp4"
    dest = folder / video_name
    if not (dest.is_file() and dest.stat().st_size == mp4.stat().st_size):
        shutil.copy2(mp4, dest)
    (folder / "title.txt").write_text(meta["title"] + "\n", encoding="utf-8")
    (folder / "description.txt").write_text(meta["description"] + "\n", encoding="utf-8")
    (folder / "tags.txt").write_text(", ".join(meta["tags"]) + "\n", encoding="utf-8")
    (folder / "captions.srt").write_text(_clip_srt(mp4, cfg, console, transcriber, holder), encoding="utf-8")
    start = float(clip.get("start") or 0.0)
    end = float(clip.get("end") or start)
    pkg = {
        "id": clip_id(source_video, start), "platform": pc.platform, "game": pc.game, "creator": pc.creator,
        "creator_url": pc.creator_url, "channel": pc.channel, "title": meta["title"],
        "description": meta["description"], "tags": meta["tags"],
        "hashtags": re.findall(r"#\w+", meta["description"].splitlines()[-1]),
        "video_file": video_name, "captions_file": "captions.srt", "source_video": source_video,
        "source_start": round(start, 3), "source_end": round(end, 3), "duration": round(end - start, 2),
        "ai_score": clip.get("ai_score"), "category": clip.get("category"), "hook": clip.get("hook"),
        "weak": bool(clip.get("weak")), "status": "ready",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool_version": __version__,
    }
    (folder / "metadata.json").write_text(json.dumps(pkg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**pkg, "folder": folder.relative_to(auto_dir).as_posix()}


def resolve_clips_json(path: Path) -> Path:
    path = Path(path)
    return path / "clips.json" if path.is_dir() else path


def export_clips(clips_json: Path, cfg: Config, console: Console, transcriber: Any = None,
                 writer: MetadataWriter | None = None) -> list[dict]:
    """Package every good clip of a clips.json; returns the package dicts (new and already-exported)."""
    pc = cfg.publish
    if not pc.creator.strip():
        raise ValueError("publish.creator is not set (use a creator preset or --creator NAME)")
    clips_json = resolve_clips_json(clips_json)
    data = json.loads(clips_json.read_text(encoding="utf-8"))
    source_video = data.get("video", clips_json.parent.name)
    root = cfg.resolve(pc.root) or Path(pc.root).resolve()
    auto_dir = root
    existing = {d["id"]: d for d in load_manifest(auto_dir)}
    if pc.ai_metadata and writer is None:
        writer = MetadataWriter(cfg.ai, cfg.base_dir)
        if not writer.available:
            console.print(f"[dim]AI metadata off ({writer.unavailable_reason}); using templates.[/dim]")
    holder: dict = {}
    out: list[dict] = []
    new: list[dict] = []
    skipped = 0
    for clip in sorted(data.get("clips", []), key=lambda c: c.get("start", 0.0)):
        if clip.get("error") or not clip.get("file"):
            skipped += 1
            continue
        if pc.skip_weak and clip.get("weak"):
            skipped += 1
            continue
        mp4 = clips_json.parent / clip["file"]
        if not mp4.is_file():
            console.print(f"  [yellow]![/yellow] missing file {mp4}, skipped")
            skipped += 1
            continue
        cid = clip_id(source_video, float(clip.get("start") or 0.0))
        prev = existing.get(cid)
        if prev and (auto_dir / prev.get("folder", "")).is_dir():
            out.append(prev)
            continue
        try:
            pkg = _package_clip(clip, mp4, source_video, cfg, console, auto_dir, writer, transcriber, holder)
        except Exception as e:  # noqa: BLE001
            console.print(f"  [red]x[/red] packaging {mp4.name} failed: {(str(e).splitlines() or [''])[0][:160]}")
            skipped += 1
            continue
        new.append(pkg)
        out.append(pkg)
    if new:
        write_manifest(auto_dir, new)
    auto_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(auto_dir / "HANDOVER.md", handover_text(pc))
    table = Table(title=f"Publish package: {pc.creator} / {pc.game or '-'}", header_style="bold cyan")
    table.add_column("NN", justify="right")
    table.add_column("Title")
    table.add_column("Folder", overflow="fold")
    for p in out:
        m = re.search(r"(\d+) - ", p["folder"].rsplit("/", 1)[-1])
        table.add_row(m.group(1) if m else "?", p["title"], p["folder"] + ("" if p in new else " (existing)"))
    console.print(table)
    cost = f", AI ${writer.spent_usd:.4f} ({writer.calls} calls, {writer.failures} failed)" if writer else ""
    console.print(f"Packaged {len(new)} new, {len(out) - len(new)} already exported, {skipped} skipped{cost}. "
                  f"Manifest: {auto_dir / 'manifest.json'}")
    return out
