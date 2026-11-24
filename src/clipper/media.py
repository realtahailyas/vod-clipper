"""Thin wrappers around the ffmpeg / ffprobe binaries."""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".flv", ".ts", ".webm", ".m4v", ".avi"}


class FFmpegError(RuntimeError):
    pass


@functools.cache
def ffmpeg_exe() -> str:
    exe = os.environ.get("CLIPPER_FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        raise FFmpegError("ffmpeg not found on PATH (or set CLIPPER_FFMPEG).")
    return exe


@functools.cache
def ffprobe_exe() -> str:
    exe = os.environ.get("CLIPPER_FFPROBE") or shutil.which("ffprobe")
    if not exe:
        raise FFmpegError("ffprobe not found on PATH (or set CLIPPER_FFPROBE).")
    return exe


def run(cmd: list[str], cwd: str | Path | None = None, desc: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Run a command, raise FFmpegError with the stderr tail on failure."""
    proc = subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
        raise FFmpegError(f"{desc} failed (exit {proc.returncode}):\n{tail}")
    return proc


@dataclass
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    audio_streams: int


def _parse_rate(r: str | None) -> float:
    if not r or r == "0/0":
        return 0.0
    if "/" in r:
        n, d = r.split("/")
        return float(n) / float(d) if float(d) else 0.0
    return float(r)


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    proc = run([ffprobe_exe(), "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)], desc="ffprobe")
    data = json.loads(proc.stdout)
    vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"
                and not s.get("disposition", {}).get("attached_pic")]
    astreams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not vstreams:
        raise FFmpegError(f"No video stream in {path}")
    v = vstreams[0]
    duration = float(data.get("format", {}).get("duration") or v.get("duration") or 0.0)
    fps = _parse_rate(v.get("avg_frame_rate")) or _parse_rate(v.get("r_frame_rate")) or 30.0
    w, h = int(v["width"]), int(v["height"])
    rot = v.get("tags", {}).get("rotate") or next(
        (sd.get("rotation") for sd in v.get("side_data_list", []) if "rotation" in sd), 0)
    if abs(int(float(rot or 0))) in (90, 270):
        w, h = h, w
    return MediaInfo(path=path, duration=duration, width=w, height=h, fps=fps,
                     audio_streams=len(astreams))


def extract_audio(video: str | Path, start: float, duration: float, out_wav: str | Path,
                  track: int = 0, sample_rate: int = 16000) -> Path:
    """Cut a mono 16 kHz WAV (what Whisper wants) for [start, start+duration)."""
    out_wav = Path(out_wav)
    run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{max(0.0, start):.3f}", "-i", str(video), "-t", f"{duration:.3f}",
         "-map", f"0:a:{track}", "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(out_wav)], desc="audio extract")
    return out_wav


# ---------------------------------------------------------------------------
# encoders

HW_ENCODER_ORDER = ("h264_nvenc", "h264_qsv", "h264_amf")


@functools.cache
def encoder_works(encoder: str) -> bool:
    """Really try the encoder: being compiled in doesn't mean the GPU/driver is there."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30:d=0.3",
           "-c:v", encoder, "-f", "null", "-"]
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=30).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


@functools.cache
def qsv_low_power_works(encoder: str) -> bool:
    """Intel VDENC (low_power) path: ~2x faster encode on Gen11+/Xe iGPUs, same quality at q+3.

    Older iGPUs/drivers reject it, so test once and fall back to the regular QSV path.
    """
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=256x256:r=30:d=0.3,format=nv12",
           "-c:v", encoder, "-low_power", "1", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=30).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def detect_encoder(preferred: str = "auto") -> str:
    if preferred != "auto":
        if not encoder_works(preferred):
            raise FFmpegError(f"Encoder '{preferred}' is not usable on this machine.")
        return preferred
    for enc in HW_ENCODER_ORDER:
        if encoder_works(enc):
            return enc
    return "libx264"


def encoder_args(encoder: str, quality: int) -> list[str]:
    q = str(quality)
    if encoder.endswith("_nvenc"):
        return ["-c:v", encoder, "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", q, "-b:v", "0", "-maxrate", "20M", "-bufsize", "40M"]
    if encoder.endswith("_qsv"):
        # Benchmarked on i5-1245U / Xe iGPU, 1080x1920@60: low_power + veryfast renders a 45 s clip
        # ~25% faster than the default path; at global_quality q+3 SSIM matches the normal path at q.
        if qsv_low_power_works(encoder):
            return ["-c:v", encoder, "-low_power", "1", "-preset", "veryfast",
                    "-global_quality", str(quality + 3), "-look_ahead", "0"]
        return ["-c:v", encoder, "-preset", "veryfast", "-global_quality", q, "-look_ahead", "0"]
    if encoder.endswith("_amf"):
        return ["-c:v", encoder, "-quality", "balanced", "-rc", "cqp",
                "-qp_i", q, "-qp_p", q, "-qp_b", q]
    if encoder == "libx265":
        return ["-c:v", "libx265", "-preset", "fast", "-crf", q, "-tag:v", "hvc1"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", q, "-profile:v", "high"]


def list_videos(paths: list[str | Path]) -> list[Path]:
    """Expand files/directories into a sorted, de-duplicated list of video files."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS and f.is_file()))
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        k = p.resolve()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq
