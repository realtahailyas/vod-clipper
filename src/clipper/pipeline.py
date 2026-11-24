"""Per-video orchestration and batch processing."""
from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from clipper import media
from clipper.ai.ranker import (AIRanker, AIVerdict, build_contact_sheet, candidate_pool, final_scores,
                               is_blocked, select_final, transcript_is_adult)
from clipper.analysis.chat import find_chat_file
from clipper.analysis.highlights import find_highlights, refine_with_transcript
from clipper.captions import build_ass
from clipper.config import Config, HighlightsConfig
from clipper.layout import compute_layout
from clipper.models import Highlight, Transcript, fmt_time
from clipper.publish import export_clips, transcript_to_srt
from clipper.render import render_clip
from clipper.transcribe import Transcriber

log = logging.getLogger(__name__)


@dataclass
class ClipResult:
    video: Path
    index: int
    highlight: Highlight
    output: Path | None
    error: str | None
    transcript_text: str
    extra: dict = field(default_factory=dict)   # AI fields for clips.json


# ---------------------------------------------------------------------------
# helpers


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_title(template: str, n: int, stem: str, hook: str = "") -> str:
    """Fill {n} / {stem} / {hook} in the title template; fall back to the raw text on bad braces."""
    if not template:
        return ""
    try:
        return template.format_map(_SafeDict(n=n, stem=stem, hook=hook))
    except (ValueError, IndexError, KeyError, AttributeError):
        return template


def needs_transcription(cfg: Config) -> bool:
    return cfg.captions.enabled or cfg.highlights.snap_to_sentences


def _first_line(e: BaseException) -> str:
    s = str(e).strip()
    return s.splitlines()[0] if s else type(e).__name__


@contextmanager
def _workdir(cfg: Config) -> Iterator[Path]:
    temp_root = cfg.resolve(cfg.paths.temp_dir)
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)
    if cfg.keep_temp:
        yield Path(tempfile.mkdtemp(prefix="clipper_", dir=temp_root))
    else:
        with tempfile.TemporaryDirectory(prefix="clipper_", dir=temp_root) as d:
            yield Path(d)


def _top_reason(h: Highlight) -> str:
    if not h.reasons:
        return "-"
    k, v = max(h.reasons.items(), key=lambda kv: kv[1])
    return f"{k} {v:+.1f}"


def print_highlights_table(console: Console, highlights: list[Highlight], title: str = "Highlights") -> None:
    table = Table(title=title, header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("Start - End")
    table.add_column("Dur", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Top reason")
    for h in sorted(highlights, key=lambda x: x.rank):
        table.add_row(str(h.rank), f"{fmt_time(h.start)} - {fmt_time(h.end)}",
                      f"{h.duration:.0f}s", f"{h.score:.2f}", _top_reason(h))
    console.print(table)


def analyze_video(video: Path, cfg: Config, console: Console,
                  hcfg: HighlightsConfig | None = None, show_table: bool = True) -> list[Highlight]:
    """Probe + highlight detection with a progress bar; prints the table.
    `hcfg` overrides cfg.highlights (the AI stage asks for a larger candidate pool)."""
    hcfg = hcfg or cfg.highlights
    info = media.probe(video)
    if info.audio_streams == 0:
        raise ValueError(f"{video.name} has no audio track, cannot detect highlights.")
    console.rule(f"[bold]{video.name}[/bold]  {fmt_time(info.duration)}  "
                 f"{info.width}x{info.height}@{info.fps:.0f}")
    chat_path: Path | None = None
    chat = cfg.highlights.chat
    if chat.enabled:
        chat_path = cfg.resolve(chat.file) if chat.file else find_chat_file(video)
        if chat_path is not None:
            console.print(f"Chat log: {chat_path}")
    cache_dir = cfg.resolve(cfg.paths.cache_dir)
    with Progress(TextColumn("Analysing audio"), BarColumn(), TaskProgressColumn(),
                  TimeElapsedColumn(), console=console, transient=True) as prog:
        task = prog.add_task("scan", total=100)

        def on_progress(frac: float) -> None:
            prog.update(task, completed=max(0.0, min(1.0, frac)) * 100)

        highlights, _ = find_highlights(video, info.duration, hcfg, cache_dir,
                                        chat_path=chat_path, progress=on_progress)
    if highlights:
        if show_table:
            print_highlights_table(console, highlights)
    else:
        console.print("[yellow]No highlights found.[/yellow]")
    return highlights


def _write_manifest(out_dir: Path, video: Path, results: list[ClipResult]) -> None:
    clips = []
    for r in sorted(results, key=lambda x: x.highlight.start):
        h = r.highlight
        clips.append({
            "rank": h.rank, "start": round(h.start, 3), "end": round(h.end, 3),
            "peak": round(h.peak, 3), "score": round(float(h.score), 4),
            "reasons": {k: round(float(v), 4) for k, v in h.reasons.items()},
            "file": r.output.name if r.output else None,
            "transcript": r.transcript_text, "error": r.error,
            **r.extra,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clips.json").write_text(
        json.dumps({"video": video.name, "clips": clips}, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# per video


def _print_candidates_table(console: Console, cands: list[dict]) -> None:
    table = Table(title="AI candidates", header_style="bold cyan")
    for col, right in (("#", True), ("Time", False), ("Dur", True), ("Audio", True), ("AI", True),
                       ("Category", False), ("Hook", False), ("Kept", False)):
        table.add_column(col, justify="right" if right else "left")
    order = sorted(cands, key=lambda c: (c["rank"] is None, c["rank"] or 0, -c["audio_score"]))
    for c in order:
        v = c["verdict"]
        table.add_row(
            str(c["rank"]) if c["rank"] else "-", fmt_time(c["start"]), f"{c['end'] - c['start']:.0f}s",
            f"{c['audio_norm']:.1f}", f"{v['score']:.1f}" if v else "-",
            v["category"] if v else "-", (v["hook"] if v else "") or "",
            "[yellow]fill[/yellow]" if c.get("weak") else "[green]✓[/green]" if c["kept"] else "[red]✗[/red]")
    console.print(table)


def _signals(h: Highlight) -> str:
    top = sorted(h.reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return ", ".join(f"{k} {v:+.1f}" for k, v in top)


def _write_candidates(out_dir: Path, video: Path, records: list[dict]) -> None:
    rows = [{**r, "start": round(r["start"], 3), "end": round(r["end"], 3), "peak": round(r["peak"], 3),
             "audio_score": round(r["audio_score"], 4), "audio_norm": round(r["audio_norm"], 3),
             "final_score": round(r["final_score"], 3)} for r in records]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates.json").write_text(
        json.dumps({"video": video.name, "candidates": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def process_video(video: Path, cfg: Config, transcriber: Transcriber | None, encoder: str,
                  console: Console, dry_run: bool = False,
                  ranker: AIRanker | None = None) -> list[ClipResult]:
    info = media.probe(video)
    use_ai = ranker is not None and ranker.available and not dry_run
    n = cfg.highlights.num_clips
    hcfg = dataclasses.replace(cfg.highlights, num_clips=candidate_pool(n, cfg.ai)) if use_ai else None
    highlights = analyze_video(video, cfg, console, hcfg=hcfg, show_table=not use_ai)
    stem = video.stem
    out_dir = cfg.resolve(cfg.paths.output_dir) / stem
    if not highlights:
        return []

    if dry_run:
        results = [ClipResult(video, h.rank, h, None, None, "") for h in highlights]
        _write_manifest(out_dir, video, results)
        console.print(f"[dim]Dry run: planned windows written to {out_dir / 'clips.json'}[/dim]")
        return results

    out_dir.mkdir(parents=True, exist_ok=True)
    layout = compute_layout(info.width, info.height, cfg.layout, cfg.captions)
    # Streamers move or hide their cam mid-stream; clips where the AI sees no cam drop the band.
    gp_layout = compute_layout(info.width, info.height,
                               dataclasses.replace(cfg.layout, mode="gameplay_only"), cfg.captions)
    # Slot 1 = the configured cam spot, then any `facecam.alternates` (streamers move their cam).
    slot_layouts = [layout]
    for alt in ((cfg.layout.facecam.alternates or []) if layout.mode == "split" else []):
        fc = dataclasses.replace(cfg.layout.facecam, preset="custom", region=list(alt), alternates=None)
        slot_layouts.append(compute_layout(info.width, info.height,
                                           dataclasses.replace(cfg.layout, facecam=fc), cfg.captions))

    def _describe(lay) -> str:
        r = lay.facecam_crop
        return (f"{100 * r.x / info.width:.0f}-{100 * (r.x + r.w) / info.width:.0f}% across, "
                f"{100 * r.y / info.height:.0f}-{100 * (r.y + r.h) / info.height:.0f}% down the frame")

    facecam_slots = [_describe(lay) for lay in slot_layouts if lay.facecam_crop is not None]
    fps = cfg.render.fps or min(info.fps, 60.0)
    fonts_dir = cfg.resolve(cfg.captions.fonts_dir)
    snap = cfg.highlights.snap_window
    render_cfg = cfg.render
    if render_cfg.fps != fps:
        render_cfg = dataclasses.replace(render_cfg, fps=fps)
    results: list[ClipResult] = []
    pending: dict[Future, tuple[Highlight, Path, str, dict]] = {}

    def report(fut: Future) -> None:
        h, out_path, text, extra = pending.pop(fut)
        err = None
        try:
            fut.result()
        except Exception as e:  # noqa: BLE001
            err = _first_line(e)
            log.debug("render failed", exc_info=True)
        results.append(ClipResult(video, h.rank, h, None if err else out_path, err, text, extra))
        if err:
            console.print(f"  [red]x[/red] clip {h.rank:02d} render failed: {err}")
        else:
            console.print(f"  [green]ok[/green] clip {h.rank:02d} -> {out_path.name}")

    def drain(block: bool) -> None:
        while pending:
            done = [f for f in pending if f.done()]
            if not done and block:
                done_set, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                done = list(done_set)
            if not done:
                return
            for f in done:
                report(f)

    def transcribe_candidate(h: Highlight, cdir: Path) -> tuple[Highlight, Transcript | None]:
        transcript: Transcript | None = None
        if transcriber is not None:
            a_start = max(0.0, h.start - snap - 1.0)
            a_end = min(info.duration, h.end + snap + 1.0)
            console.print(f"  clip {h.rank:02d}: transcribing "
                          f"{fmt_time(h.start)} - {fmt_time(h.end)} ...")
            try:
                wav = media.extract_audio(video, a_start, a_end - a_start, cdir / "audio.wav",
                                          track=cfg.highlights.audio_track)
                transcript = transcriber.transcribe(wav, offset=a_start)
            except Exception as e:  # noqa: BLE001
                console.print(f"  [yellow]![/yellow] clip {h.rank:02d} transcription failed "
                              f"({_first_line(e)}); continuing without captions")
                transcript = None
            if transcript is not None:
                h = refine_with_transcript(h, transcript, cfg.highlights, info.duration)
        return h, transcript

    def submit_render(executor: ThreadPoolExecutor, h: Highlight, transcript: Transcript | None,
                      cdir: Path, hook: str, extra: dict, clip_layout=None) -> None:
        clip_layout = clip_layout or layout
        title_text = format_title(cfg.title.text, h.rank, stem, hook)
        if not cfg.title.text and use_ai and cfg.ai.titles and hook:
            title_text = hook
        ass_text = build_ass(transcript, h.start, h.end, clip_layout, cfg.captions,
                             cfg.title if title_text else None, title_text, cfg.watermark)
        ass_path = cdir / "clip.ass"
        ass_path.write_text(ass_text, encoding="utf-8")
        out_path = out_dir / f"{stem}_{h.rank:02d}_{fmt_time(h.start, '-')}.mp4"
        if transcript is not None and transcript.words:   # reused by `clipper export` (no re-transcription)
            out_path.with_suffix(".srt").write_text(transcript_to_srt(transcript.words, h.start,
                                                                      cfg.captions.censor),
                                                    encoding="utf-8")
        text = " ".join(w.text for w in transcript.words) if transcript else ""
        fut = executor.submit(render_clip, video, h.start, h.duration, clip_layout, out_path,
                              render_cfg, encoder, cdir, ass_path, fonts_dir)
        pending[fut] = (h, out_path, text, extra)

    def guard_overlap(h: Highlight, prev_end: float) -> Highlight:
        if h.start < prev_end:  # sentence snapping may push into the previous clip
            h = dataclasses.replace(h, start=min(prev_end, h.end - 1.0))
        return h

    with _workdir(cfg) as workdir:
        if cfg.keep_temp:
            console.print(f"[dim]Keeping temp files in {workdir}[/dim]")
        executor = ThreadPoolExecutor(max_workers=cfg.render.workers)
        judge_pool = ThreadPoolExecutor(max_workers=cfg.ai.concurrency) if use_ai else None
        try:
            prev_end = 0.0
            if not use_ai:
                for h in sorted(highlights, key=lambda x: x.start):
                    cdir = workdir / f"clip{h.rank:02d}"
                    try:
                        cdir.mkdir(parents=True, exist_ok=True)
                        h, transcript = transcribe_candidate(h, cdir)
                        if cfg.ai.block_adult and transcript and transcript_is_adult(
                                " ".join(w.text for w in transcript.words)):
                            console.print(f"  [yellow]![/yellow] clip {h.rank:02d} skipped: "
                                          "adult content in transcript")
                            continue
                        h = guard_overlap(h, prev_end)
                        prev_end = h.end
                        submit_render(executor, h, transcript, cdir, "", {})
                    except Exception as e:  # noqa: BLE001
                        msg = _first_line(e)
                        console.print(f"  [red]x[/red] clip {h.rank:02d} failed: {msg}")
                        results.append(ClipResult(video, h.rank, h, None, msg, ""))
                    drain(block=False)
                drain(block=True)
            else:
                assert ranker is not None and judge_pool is not None
                console.print(f"AI ranking: judging {len(highlights)} candidates with {cfg.ai.model} "
                              f"to keep the best {n}")
                cands: list[dict] = []   # h, transcript, text, cdir, fut
                for h in sorted(highlights, key=lambda x: x.start):
                    cdir = workdir / f"clip{h.rank:02d}"
                    try:
                        cdir.mkdir(parents=True, exist_ok=True)
                        h, transcript = transcribe_candidate(h, cdir)
                        h = guard_overlap(h, prev_end)
                        prev_end = h.end
                        text = " ".join(w.text for w in transcript.words) if transcript else ""
                        sheet: Path | None = None
                        try:
                            sheet = build_contact_sheet(video, h.start, h.end, cdir / "sheet.jpg",
                                                        cfg.ai.frames)
                        except Exception as e:  # noqa: BLE001
                            console.print(f"  [yellow]![/yellow] clip {h.rank:02d} contact sheet failed "
                                          f"({_first_line(e)}); judging on transcript only")
                        meta = {"video": video.name, "timestamp": fmt_time(h.start),
                                "duration": round(h.duration), "signals": _signals(h),
                                "facecam_slots": facecam_slots}
                        fut = judge_pool.submit(ranker.judge, sheet, text, meta)
                        cands.append({"h": h, "transcript": transcript, "text": text,
                                      "cdir": cdir, "fut": fut, "args": (sheet, text, meta)})
                    except Exception as e:  # noqa: BLE001
                        msg = _first_line(e)
                        console.print(f"  [red]x[/red] clip {h.rank:02d} failed: {msg}")
                        results.append(ClipResult(video, h.rank, h, None, msg, ""))
                verdicts: list[AIVerdict | None] = []
                for c in cands:
                    try:
                        verdicts.append(c["fut"].result())
                    except Exception:  # noqa: BLE001
                        verdicts.append(None)
                missing = [i for i, v in enumerate(verdicts) if v is None]
                if missing and ranker.available:
                    console.print(f"Retrying {len(missing)} failed AI call(s) ...")
                    for i in missing:  # one sequential retry pass (network blips)
                        verdicts[i] = ranker.judge(*cands[i]["args"])
                hs = [c["h"] for c in cands]
                blocked = ({i for i, c in enumerate(cands) if is_blocked(verdicts[i], c["text"])}
                           if cfg.ai.block_adult else set())
                if blocked:
                    console.print(f"[yellow]Adult-content filter: dropped {len(blocked)} candidate(s) "
                                  "(nudity / strip club / sexual content).[/yellow]")
                kept = select_final(hs, verdicts, n, cfg.ai, blocked)
                finals = final_scores([h.score for h in hs], verdicts, cfg.ai.weight)
                top = max((h.score for h in hs), default=0.0)
                rank_of = {i: r + 1 for r, (i, _) in enumerate(kept)}
                weak_of = {i: w for i, w in kept}
                records = []
                for i, c in enumerate(cands):
                    v = verdicts[i]
                    records.append({
                        "start": c["h"].start, "end": c["h"].end, "peak": c["h"].peak,
                        "audio_score": float(c["h"].score),
                        "audio_norm": 10.0 * c["h"].score / top if top > 0 else 0.0,
                        "reasons": {k: round(float(x), 4) for k, x in c["h"].reasons.items()},
                        "verdict": v.to_dict() if v else None, "final_score": finals[i],
                        "kept": i in rank_of, "weak": weak_of.get(i, False), "rank": rank_of.get(i),
                        "blocked_adult": i in blocked, "transcript": c["text"]})
                _print_candidates_table(console, records)
                console.print(f"AI: {ranker.calls} calls, {ranker.failures} failed, "
                              f"${ranker.spent_usd:.3f}")
                n_good = sum(1 for _, w in kept if not w)
                if n_good < n:
                    if len(kept) > n_good:
                        console.print(f"[yellow]Only {n_good} of {n} clips passed ai.min_score "
                                      f"{cfg.ai.min_score}; filled {len(kept) - n_good} with the next best "
                                      f"(marked weak).[/yellow]")
                    else:
                        console.print(f"[yellow]Only {n_good} of {n} clips passed the AI filter "
                                      f"(ai.min_score {cfg.ai.min_score}).[/yellow]")
                _write_candidates(out_dir, video, records)
                for i in sorted(rank_of, key=lambda j: hs[j].start):
                    c, v = cands[i], verdicts[i]
                    h = dataclasses.replace(c["h"], rank=rank_of[i])
                    extra = {"ai_score": round(v.score, 2) if v else None,
                             "category": v.category if v else None,
                             "hook": v.hook if v else None,
                             "reason": v.reason if v else None,
                             "final_score": round(finals[i], 3),
                             "weak": weak_of[i]}
                    clip_layout = layout
                    if v is not None and layout.mode == "split":
                        if v.facecam_slot == 0:
                            clip_layout = gp_layout
                            extra["layout"] = "gameplay_only"
                            console.print(f"  clip {h.rank:02d}: no webcam at any facecam position "
                                          "-> gameplay-only layout")
                        elif 1 < v.facecam_slot <= len(slot_layouts):
                            clip_layout = slot_layouts[v.facecam_slot - 1]
                            extra["layout"] = f"facecam_slot_{v.facecam_slot}"
                            console.print(f"  clip {h.rank:02d}: webcam at alternate position "
                                          f"{v.facecam_slot}")
                    try:
                        submit_render(executor, h, c["transcript"], c["cdir"], v.hook if v else "",
                                      extra, clip_layout)
                    except Exception as e:  # noqa: BLE001
                        msg = _first_line(e)
                        console.print(f"  [red]x[/red] clip {h.rank:02d} failed: {msg}")
                        results.append(ClipResult(video, h.rank, h, None, msg, "", extra))
                    drain(block=False)
                drain(block=True)
        except BaseException:
            for f in pending:
                f.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            if judge_pool is not None:
                judge_pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
            if judge_pool is not None:
                judge_pool.shutdown(wait=True)

    _write_manifest(out_dir, video, results)
    if cfg.publish.enabled and cfg.publish.creator.strip():
        try:
            export_clips(out_dir / "clips.json", cfg, console, transcriber)
        except Exception as e:  # noqa: BLE001 - packaging must never fail the run
            console.print(f"[yellow]Publish export failed: {_first_line(e)}[/yellow]")
    return sorted(results, key=lambda r: r.index)


# ---------------------------------------------------------------------------
# batch


def _setup_ai(cfg: Config, console: Console, dry_run: bool) -> AIRanker | None:
    """Create the AI ranker when enabled and usable; print why not otherwise."""
    mode = cfg.ai.enabled
    if mode == "false":
        return None
    if dry_run:
        console.print("[dim]AI ranking skipped in dry run (it needs transcripts).[/dim]")
        return None
    ranker = AIRanker(cfg.ai, cfg.base_dir)
    if ranker.available:
        return ranker
    if mode == "true":
        console.print(f"[yellow]AI ranking unavailable: {ranker.unavailable_reason}. "
                      f"Using audio-only ranking.[/yellow]")
    else:
        console.print(f"[dim]AI ranking off: {ranker.unavailable_reason}.[/dim]")
    return None


def run_batch(inputs: list[Path], cfg: Config, console: Console, dry_run: bool = False) -> list[ClipResult]:
    t0 = time.monotonic()
    encoder = "libx264"
    transcriber: Transcriber | None = None
    ranker = _setup_ai(cfg, console, dry_run)
    if not dry_run:
        encoder = media.detect_encoder(cfg.render.encoder)
        console.print(f"Encoder: [bold]{encoder}[/bold]")
        if needs_transcription(cfg):
            transcriber = Transcriber(cfg.transcription)

    results: list[ClipResult] = []
    summary: list[tuple[Path, int, int, Path]] = []
    for video in inputs:
        try:
            res = process_video(video, cfg, transcriber, encoder, console, dry_run=dry_run, ranker=ranker)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            msg = _first_line(e)
            console.print(f"[red]x {video.name}: {msg}[/red]")
            log.debug("video failed", exc_info=True)
            res = [ClipResult(video, 0, Highlight(0.0, 0.0, 0.0, 0.0), None, msg, "")]
        results.extend(res)
        ok = sum(1 for r in res if r.error is None)
        summary.append((video, ok, len(res) - ok, cfg.resolve(cfg.paths.output_dir) / video.stem))

    table = Table(title="Summary", header_style="bold cyan")
    table.add_column("Video")
    table.add_column("Clips ok", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Output")
    for video, ok, bad, out in summary:
        table.add_row(video.name, str(ok), f"[red]{bad}[/red]" if bad else "0", str(out))
    console.print(table)
    console.print(f"Elapsed: {fmt_time(time.monotonic() - t0)}")
    return results
