from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper import media
from clipper.captions import build_ass
from clipper.config import CaptionsConfig, LayoutConfig, RenderConfig, TitleConfig, TranscriptionConfig
from clipper.layout import compute_layout
from clipper.models import Segment, Transcript, Word
from clipper.render import (_escape_filter_value, build_filtergraph, render_clip, render_preview,
                            render_source_preview)
from clipper.transcribe import Transcriber, resolve_device

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available")


# ---- pure graph tests ------------------------------------------------------

def test_graph_split():
    lay = compute_layout(1920, 1080, LayoutConfig())
    g = build_filtergraph(lay, 30, "caps.ass", None)
    fc, gp = lay.facecam_crop, lay.gameplay_crop
    assert "[0:v]split=2[fc][gp]" in g
    assert f"crop={fc.w}:{fc.h}:{fc.x}:{fc.y}" in g
    assert f"crop={gp.w}:{gp.h}:{gp.x}:{gp.y}" in g
    assert f"scale=1080:{lay.facecam_h}:flags=lanczos" in g
    assert f"scale=1080:{lay.gameplay_h}:flags=lanczos" in g
    assert "vstack=inputs=2[st]" in g
    assert f"pad=1080:1920:0:{lay.header_h}:color=black[base]" in g
    assert "subtitles=filename='caps.ass'" in g
    assert "fps=30" in g and g.endswith("format=yuv420p[vout]")


def test_graph_gameplay_only_and_no_ass():
    lay = compute_layout(1920, 1080, LayoutConfig(mode="gameplay_only"))
    g = build_filtergraph(lay, None, None, None)
    assert "split=2" not in g and "vstack" not in g and "subtitles" not in g and "fps=" not in g
    assert f"pad=1080:1920:0:{lay.header_h}" in g
    assert g.endswith("[vout]")


def test_graph_blur():
    cfg = LayoutConfig()
    cfg.gameplay.fit = "blur"
    lay = compute_layout(1920, 1080, cfg)
    g = build_filtergraph(lay, None, None, None)
    gh = lay.gameplay_h
    assert "split[g1][g2]" in g
    assert f"scale=1080:{gh}:force_original_aspect_ratio=increase,crop=1080:{gh},boxblur=20:2[bg]" in g
    assert "[g2]scale=1080:-2[fg]" in g
    assert "overlay=(W-w)/2:(H-h)/2" in g


def test_fonts_dir_escaping():
    assert _escape_filter_value("C:\\Users\\me\\Fonts") == "'C\\:/Users/me/Fonts'"
    lay = compute_layout(1920, 1080, LayoutConfig())
    g = build_filtergraph(lay, None, "a.ass", "C:\\My Fonts")
    assert "fontsdir='C\\:/My Fonts'" in g


# ---- transcribe (no model) ---------------------------------------------------

def test_resolve_device(monkeypatch):
    import clipper.transcribe as t
    monkeypatch.setattr(t, "_cuda_count", lambda: 0)
    auto = TranscriptionConfig(compute_type="auto")
    assert resolve_device(auto) == ("cpu", "int8")
    monkeypatch.setattr(t, "_cuda_count", lambda: 1)
    assert resolve_device(auto) == ("cuda", "float16")
    assert resolve_device(TranscriptionConfig()) == ("cuda", "int8")  # explicit int8 default kept
    assert resolve_device(TranscriptionConfig(device="cpu")) == ("cpu", "int8")
    assert resolve_device(TranscriptionConfig(device="cuda", compute_type="int8")) == ("cuda", "int8")
    assert Transcriber(TranscriptionConfig(device="cpu")).device == "cpu"


# ---- integration -------------------------------------------------------------

def _ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                          "-show_format", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    d = tmp_path_factory.mktemp("src")
    p = d / "source.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


def _fake_ass(tmp_path: Path, layout) -> Path:
    d = tmp_path / "dir with space"
    d.mkdir()
    words = [Word("hello", 0.2, 0.6), Word("there", 0.65, 1.0), Word("big", 1.05, 1.4),
             Word("world", 1.45, 2.0)]
    tr = Transcript(segments=[Segment("hello there big world", 0.2, 2.0, words)])
    ass = build_ass(tr, 0.0, 3.0, layout, CaptionsConfig(), TitleConfig(), "Test title")
    p = d / "my captions.ass"
    p.write_text(ass, encoding="utf-8")
    return p


@needs_ffmpeg
@pytest.mark.parametrize("fit", ["crop", "blur"])
def test_render_clip_integration(tmp_path, source, fit):
    cfg_l = LayoutConfig()
    cfg_l.gameplay.fit = fit
    layout = compute_layout(1920, 1080, cfg_l, CaptionsConfig())
    ass = _fake_ass(tmp_path, layout)
    work = tmp_path / "work dir"
    out = tmp_path / "out" / "clip.mp4"
    res = render_clip(source, 0.5, 3.0, layout, out, RenderConfig(hwaccel="none"),
                      "libx264", work, ass_path=ass)
    assert res == out.resolve() and out.exists()
    assert not out.with_suffix(".part.mp4").exists()
    info = _ffprobe(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert float(info["format"]["duration"]) == pytest.approx(3.0, abs=0.4)


@needs_ffmpeg
def test_render_clip_fallback_from_bad_hwaccel(tmp_path, source):
    layout = compute_layout(1920, 1080, LayoutConfig(mode="gameplay_only"))
    out = tmp_path / "fb.mp4"
    render_clip(source, 0, 1.5, layout, out, RenderConfig(hwaccel="bogus_accel"),
                "libx264", tmp_path / "w")
    assert out.exists()


@needs_ffmpeg
def test_render_failure_raises(tmp_path):
    layout = compute_layout(1920, 1080, LayoutConfig())
    with pytest.raises(media.FFmpegError):
        render_clip(tmp_path / "missing.mp4", 0, 1, layout, tmp_path / "o.mp4",
                    RenderConfig(hwaccel="none"), "libx264", tmp_path / "w")
    assert not (tmp_path / "o.part.mp4").exists() and not (tmp_path / "o.mp4").exists()


@needs_ffmpeg
def test_render_preview_and_source_preview(tmp_path, source):
    layout = compute_layout(1920, 1080, LayoutConfig(), CaptionsConfig())
    ass = _fake_ass(tmp_path, layout)
    png = render_preview(source, 1.0, layout, tmp_path / "p" / "prev.png", tmp_path / "w",
                         ass_path=ass)
    assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"
    v = _ffprobe(png)["streams"][0]
    assert (v["width"], v["height"]) == (1080, 1920)
    src_png = render_source_preview(source, 1.0, layout, tmp_path / "p" / "src.png")
    assert src_png.exists() and src_png.read_bytes()[:4] == b"\x89PNG"
    v = _ffprobe(src_png)["streams"][0]
    assert (v["width"], v["height"]) == (1920, 1080)


def _yavg(png: Path, x: int, y: int, w: int, h: int) -> float:
    import re
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(png),
                        "-vf", f"crop={w}:{h}:{x}:{y},signalstats,metadata=print:key=lavfi.signalstats.YAVG",
                        "-f", "null", "-"], capture_output=True, text=True, check=True)
    return float(re.search(r"YAVG=([0-9.]+)", r.stderr).group(1))


@needs_ffmpeg
def test_watermark_changes_pixels_in_preview(tmp_path, source):
    from clipper.captions import watermark_position
    from clipper.config import WatermarkConfig
    caps = CaptionsConfig(enabled=False)
    layout = compute_layout(1920, 1080, LayoutConfig(), caps)
    wm = WatermarkConfig(opacity=0.9)
    x, y, _ = watermark_position(layout, wm, caps)
    means = {}
    for name, w in (("on", wm), ("off", WatermarkConfig(enabled=False))):
        ass = build_ass(None, 0.0, 3.0, layout, caps, None, "", w)
        p = tmp_path / f"{name}.ass"
        p.write_text(ass, encoding="utf-8")
        png = render_preview(source, 1.0, layout, tmp_path / f"{name}.png", tmp_path / f"w_{name}",
                             ass_path=p)
        means[name] = _yavg(png, x - 330, y - 40, 330, 40)
    assert means["on"] != means["off"]
