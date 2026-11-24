"""Command line interface: `clipper <command>`."""
from __future__ import annotations

import argparse
import copy
import logging
import os
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path

from rich.console import Console
from rich.table import Table

from clipper import media
from clipper.config import FACECAM_PRESETS, Config, ConfigError, load_config, validate

FACECAM_CHOICES = [p for p in FACECAM_PRESETS if p != "custom"]
DOCTOR_ENCODERS = ("h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_amf", "libx264")


# ---------------------------------------------------------------------------
# argument parsing


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", metavar="CONFIG", action="append",
                        help="path to config.yaml (default: ./config.yaml); repeat to layer files, later wins")
    common.add_argument("-v", "--verbose", action="store_true", help="show INFO logging")

    p = argparse.ArgumentParser(prog="clipper", description="Turn long gaming VODs into vertical highlight clips.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", parents=[common], help="analyse, caption and render clips")
    r.add_argument("inputs", nargs="*", metavar="INPUTS", help="video files or folders (default: paths.input_dir)")
    r.add_argument("-n", "--clips", type=int, help="number of clips per video")
    r.add_argument("--min", type=float, dest="min", metavar="SECONDS", help="minimum clip duration")
    r.add_argument("--max", type=float, dest="max", metavar="SECONDS", help="maximum clip duration")
    r.add_argument("--facecam", choices=FACECAM_CHOICES, help="facecam corner preset")
    r.add_argument("--layout", choices=["split", "gameplay_only"], help="output layout")
    r.add_argument("--title", help="title text ({n} = clip number, {stem} = video name, {hook} = AI hook)")
    r.add_argument("--no-captions", action="store_true", help="disable burned-in captions")
    wm_r = r.add_mutually_exclusive_group()
    wm_r.add_argument("--watermark", metavar="TEXT", help="watermark text (overrides watermark.text)")
    wm_r.add_argument("--no-watermark", action="store_true", help="disable the watermark")
    r.add_argument("--model", help="whisper model name (e.g. small, large-v3)")
    r.add_argument("--encoder", help="video encoder (auto, h264_nvenc, libx264, ...)")
    r.add_argument("--chat", metavar="FILE", help="chat log file")
    ai_grp = r.add_mutually_exclusive_group()
    ai_grp.add_argument("--ai", action="store_true", help="force AI ranking on (ai.enabled: true)")
    ai_grp.add_argument("--no-ai", action="store_true", help="disable AI ranking (ai.enabled: false)")
    fill_grp = r.add_mutually_exclusive_group()
    fill_grp.add_argument("--fill", action="store_true", help="always deliver num_clips: fill with weak clips (ai.fill: true)")
    fill_grp.add_argument("--strict", action="store_true", help="drop clips below ai.min_score, even if fewer than num_clips (ai.fill: false)")
    r.add_argument("--no-publish", action="store_true", help="skip packaging clips into the publish folder")
    r.add_argument("--keep-temp", action="store_true", help="keep temporary files")
    r.add_argument("--dry-run", action="store_true", help="only analyse and write clips.json")

    ex = sub.add_parser("export", parents=[common], help="package rendered clips into the publish folder")
    ex.add_argument("path", metavar="PATH", help="an output/<video> folder or its clips.json")
    ex.add_argument("--creator", metavar="NAME", help="creator display name (publish.creator)")
    ex.add_argument("--creator-url", metavar="URL", help="credit link (publish.creator_url)")
    ex.add_argument("--game", metavar="NAME", help="game name (publish.game)")
    ex.add_argument("--root", metavar="DIR", help="publish root folder (publish.root)")
    ex.add_argument("--include-weak", action="store_true", help="also package clips the AI marked weak")

    a = sub.add_parser("analyze", parents=[common], help="show detected highlights only")
    a.add_argument("video", metavar="VIDEO")
    a.add_argument("-n", "--clips", type=int, help="number of highlights")
    a.add_argument("--chat", metavar="FILE", help="chat log file")

    pv = sub.add_parser("preview", parents=[common], help="write a preview still of the layout")
    pv.add_argument("video", metavar="VIDEO")
    pv.add_argument("--at", metavar="TIME", help="timestamp, e.g. 1:23:45 or 5025 (default: auto)")
    pv.add_argument("-o", "--out-dir", metavar="DIR", help="output folder (default: <output_dir>/_preview)")
    pv.add_argument("--facecam", choices=FACECAM_CHOICES, help="facecam corner preset")
    wm_p = pv.add_mutually_exclusive_group()
    wm_p.add_argument("--watermark", metavar="TEXT", help="watermark text (overrides watermark.text)")
    wm_p.add_argument("--no-watermark", action="store_true", help="disable the watermark")

    sub.add_parser("doctor", parents=[common], help="check ffmpeg, encoders, CUDA and whisper")

    ic = sub.add_parser("init-config", parents=[common], help="write a default config.yaml")
    ic.add_argument("path", nargs="?", default="config.yaml", metavar="PATH")
    ic.add_argument("--force", action="store_true", help="overwrite an existing file")
    return p


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI flags on top of the loaded config (returns a new, validated Config)."""
    cfg = copy.deepcopy(cfg)

    def get(name: str):
        return getattr(args, name, None)

    if get("clips") is not None:
        cfg.highlights.num_clips = get("clips")
    if get("min") is not None:
        cfg.highlights.min_duration = get("min")
    if get("max") is not None:
        cfg.highlights.max_duration = get("max")
    if get("facecam") is not None:
        cfg.layout.facecam.preset = get("facecam")
    if get("layout") is not None:
        cfg.layout.mode = get("layout")
    if get("title") is not None:
        cfg.title.text = get("title")
    if get("no_captions"):
        cfg.captions.enabled = False
    if get("watermark") is not None:
        cfg.watermark.text = get("watermark")
        cfg.watermark.enabled = True
    if get("no_watermark"):
        cfg.watermark.enabled = False
    if get("model") is not None:
        cfg.transcription.model = get("model")
    if get("encoder") is not None:
        cfg.render.encoder = get("encoder")
    if get("chat") is not None:
        cfg.highlights.chat.file = str(Path(get("chat")).resolve())
        cfg.highlights.chat.enabled = True
    if get("ai"):
        cfg.ai.enabled = "true"
    if get("no_ai"):
        cfg.ai.enabled = "false"
    if get("fill"):
        cfg.ai.fill = True
    if get("strict"):
        cfg.ai.fill = False
    if get("no_publish"):
        cfg.publish.enabled = False
    if get("creator") is not None:
        cfg.publish.creator = get("creator")
    if get("creator_url") is not None:
        cfg.publish.creator_url = get("creator_url")
    if get("game") is not None:
        cfg.publish.game = get("game")
    if get("root") is not None:
        cfg.publish.root = str(Path(get("root")).resolve())
    if get("include_weak"):
        cfg.publish.skip_weak = False
    if get("keep_temp"):
        cfg.keep_temp = True
    return validate(cfg)


# ---------------------------------------------------------------------------
# commands


def cmd_run(args: argparse.Namespace, cfg: Config, console: Console) -> int:
    from clipper import pipeline

    sources = args.inputs or [cfg.resolve(cfg.paths.input_dir)]
    videos = media.list_videos(sources)
    if not videos:
        console.print(f"[red]No video files found in: {', '.join(map(str, sources))}[/red]")
        return 2
    results = pipeline.run_batch(videos, cfg, console, dry_run=args.dry_run)
    return 1 if any(r.error for r in results) else 0


def cmd_export(args: argparse.Namespace, cfg: Config, console: Console) -> int:
    from clipper.publish import export_clips, resolve_clips_json

    src = resolve_clips_json(Path(args.path))
    if not src.is_file():
        console.print(f"[red]clips.json not found: {src}[/red]")
        return 2
    try:
        export_clips(src, cfg, console)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return 2
    return 0


def cmd_analyze(args: argparse.Namespace, cfg: Config, console: Console) -> int:
    from clipper import pipeline

    video = Path(args.video)
    if not video.is_file():
        console.print(f"[red]File not found: {video}[/red]")
        return 2
    pipeline.analyze_video(video, cfg, console)
    return 0


def cmd_preview(args: argparse.Namespace, cfg: Config, console: Console) -> int:
    from clipper.captions import build_ass
    from clipper.layout import compute_layout
    from clipper.models import Segment, Transcript, Word, parse_time
    from clipper.pipeline import format_title
    from clipper.render import render_preview, render_source_preview

    video = Path(args.video)
    if not video.is_file():
        console.print(f"[red]File not found: {video}[/red]")
        return 2
    info = media.probe(video)
    at = parse_time(args.at) if args.at else min(120.0, info.duration * 0.3)
    at = max(0.0, min(at, max(info.duration - 1.0, 0.0)))
    layout = compute_layout(info.width, info.height, cfg.layout, cfg.captions)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.resolve(cfg.paths.output_dir) / "_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    words = [Word(t, at + i * 0.4, at + i * 0.4 + 0.4) for i, t in enumerate(["this", "is", "insane"])]
    transcript = Transcript([Segment("this is insane", at, at + 1.2, words)])
    title_text = format_title(cfg.title.text, 1, video.stem) or "Sample title text"
    ass = build_ass(transcript, at, at + 3.0, layout, cfg.captions, cfg.title, title_text, cfg.watermark)
    fonts_dir = cfg.resolve(cfg.captions.fonts_dir)

    with tempfile.TemporaryDirectory(prefix="clipper_") as d:
        work = Path(d)
        ass_path = work / "preview.ass"
        ass_path.write_text(ass, encoding="utf-8")
        final_png = render_preview(video, at, layout, out_dir / f"{video.stem}_preview.png", work,
                                   ass_path=ass_path, fonts_dir=fonts_dir)
        source_png = render_source_preview(video, at, layout, out_dir / f"{video.stem}_source.png")
    console.print(f"Final composition: {final_png}")
    console.print(f"Source with facecam box: {source_png}")
    return 0


def _tool_version(exe: str) -> str:
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=15).stdout
        return out.splitlines()[0] if out else "?"
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def cmd_doctor(args: argparse.Namespace, cfg: Config, console: Console) -> int:
    table = Table(title="clipper doctor", header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Result")
    ok = True

    def yn(v: bool) -> str:
        return "[green]yes[/green]" if v else "[red]no[/red]"

    ffmpeg_ok = True
    for label, getter in (("ffmpeg", media.ffmpeg_exe), ("ffprobe", media.ffprobe_exe)):
        try:
            exe = getter()
            table.add_row(label, f"{exe}\n{_tool_version(exe)}")
        except media.FFmpegError as e:
            table.add_row(label, f"[red]missing[/red] ({e})")
            ffmpeg_ok = False
    ok &= ffmpeg_ok

    if ffmpeg_ok:
        for enc in DOCTOR_ENCODERS:
            table.add_row(f"encoder {enc}", yn(media.encoder_works(enc)))
        table.add_row("encoder picked by 'auto'", media.detect_encoder("auto"))
    else:
        table.add_row("encoders", "[yellow]skipped (no ffmpeg)[/yellow]")

    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        table.add_row("CUDA devices", str(n))
    except Exception as e:  # noqa: BLE001
        table.add_row("CUDA devices", f"0 ({type(e).__name__})")
    try:
        import faster_whisper  # noqa: F401
        table.add_row("faster-whisper", yn(True))
    except Exception:  # noqa: BLE001
        table.add_row("faster-whisper", yn(False))
        ok = False
    from clipper.ai.ranker import find_api_key, openai_installed
    table.add_row("openai package", yn(openai_installed()))
    table.add_row(f"API key ({cfg.ai.api_key_env})", yn(find_api_key(cfg.ai, cfg.base_dir) is not None))
    table.add_row("AI model", f"{cfg.ai.model} (ai.enabled: {cfg.ai.enabled})")
    console.print(table)
    return 0 if ok else 1


def cmd_init_config(args: argparse.Namespace, console: Console) -> int:
    dest = Path(args.path)
    if dest.exists() and not args.force:
        console.print(f"[red]{dest} already exists (use --force to overwrite).[/red]")
        return 1
    text = resources.files("clipper").joinpath("default_config.yaml").read_text(encoding="utf-8")
    if dest.parent != Path(""):
        dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    console.print(f"Wrote {dest}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    # Quiet Hugging Face model-download chatter (symlink/HF_TOKEN notices) unless -v.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if not args.verbose:
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    console = Console()
    try:
        if args.command == "init-config":
            return cmd_init_config(args, console)
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, args)
        if args.command == "doctor":
            return cmd_doctor(args, cfg, console)
        handler = {"run": cmd_run, "export": cmd_export, "analyze": cmd_analyze, "preview": cmd_preview}[args.command]
        return handler(args, cfg, console)
    except ConfigError as e:
        console.print(f"[red]{e}[/red]", highlight=False)
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    except media.FFmpegError as e:
        console.print(f"[red]{e}[/red]", highlight=False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
