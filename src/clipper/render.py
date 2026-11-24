"""ffmpeg filter graph, clip rendering and preview stills."""
from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

from . import media
from .config import RenderConfig
from .models import Layout, Rect

log = logging.getLogger(__name__)

_SOFT_ENCODERS = ("libx264", "libx265")


def _escape_filter_value(value: str) -> str:
    """Escape a path/name for use as a filter option value (wrapped in single quotes)."""
    v = value.replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")
    return f"'{v}'"


def _crop(r: Rect) -> str:
    return f"crop={r.w}:{r.h}:{r.x}:{r.y}"


def _gameplay_chain(src_label: str, layout: Layout, out_label: str) -> str:
    """Filter chain that turns the source frame into the gameplay band [out_label]."""
    W, GH = layout.out_w, layout.gameplay_h
    c = _crop(layout.gameplay_crop)
    if layout.gameplay_fit == "blur":
        return (f"{src_label}{c},split[g1][g2];"
                f"[g1]scale={W}:{GH}:force_original_aspect_ratio=increase,crop={W}:{GH},boxblur=20:2[bg];"
                f"[g2]scale={W}:-2[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1{out_label}")
    return f"{src_label}{c},scale={W}:{GH}:flags=lanczos,setsar=1{out_label}"


def build_filtergraph(layout: Layout, fps: float | None, ass_name: str | None,
                      fonts_dir: str | None, bg_color: str = "black") -> str:
    """Single-pass -filter_complex graph; the final video label is [vout]."""
    W, H = layout.out_w, layout.out_h
    parts: list[str] = []
    if layout.facecam_crop is not None and layout.facecam_h > 0:
        parts.append("[0:v]split=2[fc][gp]")
        parts.append(f"[fc]{_crop(layout.facecam_crop)},"
                     f"scale={W}:{layout.facecam_h}:flags=lanczos,setsar=1[fcs]")
        parts.append(_gameplay_chain("[gp]", layout, "[gps]"))
        parts.append("[fcs][gps]vstack=inputs=2[st]")
    else:
        parts.append(_gameplay_chain("[0:v]", layout, "[st]"))
    parts.append(f"[st]pad={W}:{H}:0:{layout.header_h}:color={bg_color}[base]")

    tail = "[base]"
    if ass_name:
        sub = f"subtitles=filename={_escape_filter_value(ass_name)}"
        if fonts_dir:
            sub += f":fontsdir={_escape_filter_value(fonts_dir)}"
        parts.append(f"{tail}{sub}[sub]")
        tail = "[sub]"
    final = []
    if fps:
        final.append(f"fps={fps:g}")
    final.append("format=yuv420p")
    parts.append(f"{tail}{','.join(final)}[vout]")
    return ";".join(parts)


def _fonts_arg(fonts_dir: Path | None) -> str | None:
    return str(Path(fonts_dir).resolve()) if fonts_dir else None


def _stage_ass(ass_path: Path | None, workdir: Path, tag: str) -> str | None:
    """Make sure the .ass sits in workdir under a filter-safe bare name; return that name."""
    if ass_path is None:
        return None
    ass_path = Path(ass_path)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ass_path.name)
    if ass_path.parent.resolve() == workdir.resolve() and safe == ass_path.name:
        return ass_path.name
    name = f"{re.sub(r'[^A-Za-z0-9_-]', '_', tag)}_{safe}"
    shutil.copyfile(ass_path, workdir / name)
    return name


def _resolve_fps(video: Path, cfg: RenderConfig) -> float | None:
    if cfg.fps:
        return cfg.fps
    try:
        src = media.probe(video).fps
    except Exception:  # noqa: BLE001 - fps is a nicety, never fatal
        return None
    return min(src, 60.0) if src > 0 else None


def render_clip(video: Path, start: float, duration: float, layout: Layout, out_path: Path,
                cfg: RenderConfig, encoder: str, workdir: Path, ass_path: Path | None = None,
                fonts_dir: Path | None = None) -> Path:
    """Render one vertical clip. Retries without hwaccel, then with libx264, before giving up."""
    video = Path(video).resolve()
    out_path = Path(out_path).resolve()
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ass_name = _stage_ass(ass_path, workdir, out_path.stem)
    graph = build_filtergraph(layout, _resolve_fps(video, cfg), ass_name, _fonts_arg(fonts_dir),
                              layout.background_color)
    part = out_path.with_suffix(".part.mp4")

    attempts: list[tuple[str, str]] = [(cfg.hwaccel, encoder)]
    if cfg.hwaccel != "none":
        attempts.append(("none", encoder))
    if encoder not in _SOFT_ENCODERS:
        attempts.append(("none", "libx264"))

    last_err: Exception | None = None
    for i, (hw, enc) in enumerate(attempts):
        cmd = [media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{max(0.0, start):.3f}"]
        if hw != "none":
            cmd += ["-hwaccel", hw]
        cmd += ["-i", str(video), "-t", f"{duration:.3f}",
                "-filter_complex", graph,
                "-map", "[vout]", "-map", f"0:a:{cfg.audio_track}?"]
        af = "aresample=48000"
        if cfg.normalize_audio:
            af = "loudnorm=I=-14:TP=-1.5:LRA=11," + af
        cmd += ["-af", af, "-c:a", "aac", "-b:a", cfg.audio_bitrate, "-ar", "48000"]
        cmd += media.encoder_args(enc, cfg.quality)
        if enc in _SOFT_ENCODERS:
            cmd += ["-pix_fmt", "yuv420p"]
        cmd += ["-movflags", "+faststart", "-f", "mp4", str(part)]
        try:
            media.run(cmd, cwd=workdir, desc=f"render {out_path.name}")
            os.replace(part, out_path)
            return out_path
        except media.FFmpegError as exc:
            last_err = exc
            part.unlink(missing_ok=True)
            if i + 1 < len(attempts):
                log.warning("render attempt %d (hwaccel=%s, encoder=%s) failed, retrying: %s",
                            i + 1, hw, enc, str(exc).splitlines()[0])
    assert last_err is not None
    raise last_err


def _require_frame(out_png: Path, at: float) -> None:
    """ffmpeg exits 0 but writes nothing when `at` is past the end (e.g. a recording still in progress)."""
    if not Path(out_png).exists() or Path(out_png).stat().st_size == 0:
        raise media.FFmpegError(f"No frame at {at:.1f}s - is the time past the end of the video?")


def render_preview(video: Path, at: float, layout: Layout, out_png: Path, workdir: Path,
                   ass_path: Path | None = None, fonts_dir: Path | None = None) -> Path:
    """One frame of the final 9:16 composition as PNG."""
    video = Path(video).resolve()
    out_png = Path(out_png).resolve()
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ass_name = _stage_ass(ass_path, workdir, out_png.stem)
    graph = build_filtergraph(layout, None, ass_name, _fonts_arg(fonts_dir), layout.background_color)
    cmd = [media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{max(0.0, at):.3f}", "-i", str(video),
           "-filter_complex", graph, "-map", "[vout]", "-an",
           "-frames:v", "1", "-update", "1", str(out_png)]
    media.run(cmd, cwd=workdir, desc="preview")
    _require_frame(out_png, at)
    return out_png


def render_source_preview(video: Path, at: float, layout: Layout, out_png: Path) -> Path:
    """Raw source frame with the facecam crop (red) and gameplay crop (green) outlined."""
    video = Path(video).resolve()
    out_png = Path(out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    boxes = []
    if layout.facecam_crop is not None:
        r = layout.facecam_crop
        boxes.append(f"drawbox=x={r.x}:y={r.y}:w={r.w}:h={r.h}:color=red:t=6")
    g = layout.gameplay_crop
    boxes.append(f"drawbox=x={g.x}:y={g.y}:w={g.w}:h={g.h}:color=green:t=6")
    cmd = [media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{max(0.0, at):.3f}", "-i", str(video),
           "-vf", ",".join(boxes), "-frames:v", "1", "-update", "1", str(out_png)]
    media.run(cmd, desc="source preview")
    _require_frame(out_png, at)
    return out_png
