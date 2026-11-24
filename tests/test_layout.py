from __future__ import annotations

import pytest

from clipper.config import CaptionsConfig, FacecamConfig, LayoutConfig
from clipper.layout import compute_layout, facecam_region

SOURCES = [(1920, 1080), (2560, 1440)]
PRESETS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def _inside(r, w, h):
    return r.x >= 0 and r.y >= 0 and r.x + r.w <= w and r.y + r.h <= h


@pytest.mark.parametrize("src", SOURCES)
@pytest.mark.parametrize("preset", PRESETS)
def test_presets_split(src, preset):
    sw, sh = src
    cfg = LayoutConfig(facecam=FacecamConfig(preset=preset))
    lay = compute_layout(sw, sh, cfg)
    for r in (lay.facecam_crop, lay.gameplay_crop):
        assert r is not None
        assert _inside(r, sw, sh)
        assert all(v % 2 == 0 for v in (r.x, r.y, r.w, r.h))
    assert lay.header_h + lay.facecam_h + lay.gameplay_h == lay.out_h
    assert lay.gameplay_y == lay.header_h + lay.facecam_h
    fc = lay.facecam_crop
    assert fc.w / fc.h == pytest.approx(lay.out_w / lay.facecam_h, rel=0.01)
    gp = lay.gameplay_crop
    assert gp.w / gp.h == pytest.approx(lay.out_w / lay.gameplay_h, rel=0.01)


@pytest.mark.parametrize("preset,xs,ys", [
    ("top-left", "left", "top"), ("top-right", "right", "top"),
    ("bottom-left", "left", "bottom"), ("bottom-right", "right", "bottom")])
def test_region_anchored(preset, xs, ys):
    r = facecam_region(1920, 1080, FacecamConfig(preset=preset))
    assert r.x == (0 if xs == "left" else 1920 - r.w)
    assert r.y == (0 if ys == "top" else 1080 - r.h)
    assert r.w == round(0.26 * 1920) and r.h == round(0.30 * 1080)


def test_region_margin():
    r = facecam_region(1920, 1080, FacecamConfig(preset="top-left", margin=[0.1, 0.1]))
    assert (r.x, r.y) == (192, 108)


def test_gameplay_only():
    lay = compute_layout(1920, 1080, LayoutConfig(mode="gameplay_only"))
    assert lay.facecam_h == 0 and lay.facecam_crop is None
    assert lay.gameplay_y == lay.header_h
    assert lay.gameplay_h == lay.out_h - lay.header_h
    assert lay.caption_y == round(lay.header_h + lay.gameplay_h * 0.2)


def test_custom_region_pixels_and_fractions():
    px = facecam_region(1920, 1080, FacecamConfig(preset="custom", region=[100, 50, 400, 300]))
    assert (px.x, px.y, px.w, px.h) == (100, 50, 400, 300)
    fr = facecam_region(1920, 1080, FacecamConfig(preset="custom", region=[0.5, 0.5, 0.25, 0.25]))
    assert (fr.x, fr.y, fr.w, fr.h) == (960, 540, 480, 270)


def test_custom_region_clamped():
    r = facecam_region(1920, 1080, FacecamConfig(preset="custom", region=[1800, 1000, 400, 300]))
    assert _inside(r, 1920, 1080)


def test_gameplay_zoom_and_center():
    cfg = LayoutConfig()
    cfg.gameplay.zoom = 2.0
    cfg.gameplay.center_x = 0.0
    lay = compute_layout(1920, 1080, cfg)
    gp = lay.gameplay_crop
    assert gp.h == 540 and gp.x == 0 and _inside(gp, 1920, 1080)


def test_gameplay_narrow_source_clamps_width():
    cfg = LayoutConfig(width=1080, height=1920, header_height=0, facecam_height=200)
    lay = compute_layout(1000, 1000, cfg)
    gp = lay.gameplay_crop
    assert gp.w <= 1000 and _inside(gp, 1000, 1000)
    assert gp.w / gp.h == pytest.approx(1080 / lay.gameplay_h, rel=0.01)


def test_blur_keeps_source_aspect():
    cfg = LayoutConfig()
    cfg.gameplay.fit = "blur"
    lay = compute_layout(1920, 1080, cfg)
    gp = lay.gameplay_crop
    assert (gp.w, gp.h) == (1920, 1080) and lay.gameplay_fit == "blur"
    cfg.gameplay.zoom = 1.5
    gp = compute_layout(1920, 1080, cfg).gameplay_crop
    assert gp.w / gp.h == pytest.approx(16 / 9, rel=0.01) and _inside(gp, 1920, 1080)


def test_caption_positions():
    cfg = LayoutConfig()
    seam = compute_layout(1920, 1080, cfg, CaptionsConfig(position="seam"))
    assert seam.caption_y == cfg.header_height + cfg.facecam_height
    gp = compute_layout(1920, 1080, cfg, CaptionsConfig(position="gameplay"))
    assert gp.caption_y == round(gp.gameplay_y + 0.30 * gp.gameplay_h)
    bot = compute_layout(1920, 1080, cfg, CaptionsConfig(position="bottom"))
    assert bot.caption_y == round(1920 - 0.18 * 1920)
    off = compute_layout(1920, 1080, cfg, CaptionsConfig(position="seam", y_offset=25))
    assert off.caption_y == seam.caption_y + 25
