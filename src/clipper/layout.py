"""Source/output geometry: where the facecam is, what gets cropped, where captions go."""
from __future__ import annotations

from .config import CaptionsConfig, FacecamConfig, LayoutConfig
from .models import Layout, Rect


def _even_floor(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def _even_round(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def facecam_region(src_w: int, src_h: int, cfg: FacecamConfig) -> Rect:
    """Rectangle (source pixels) where the streamer's facecam sits."""
    if cfg.preset == "custom":
        if not cfg.region or len(cfg.region) != 4:
            raise ValueError("facecam.region must be [x, y, w, h] for preset 'custom'")
        x, y, w, h = cfg.region
        x = x * src_w if x <= 1 else x
        w = w * src_w if w <= 1 else w
        y = y * src_h if y <= 1 else y
        h = h * src_h if h <= 1 else h
    else:
        w = cfg.size[0] * src_w
        h = cfg.size[1] * src_h
        mx = cfg.margin[0] * src_w
        my = cfg.margin[1] * src_h
        x = mx if cfg.preset.endswith("left") else src_w - w - mx
        y = my if cfg.preset.startswith("top") else src_h - h - my
    wi = int(_clamp(round(w), 2, src_w))
    hi = int(_clamp(round(h), 2, src_h))
    xi = int(_clamp(round(x), 0, src_w - wi))
    yi = int(_clamp(round(y), 0, src_h - hi))
    return Rect(xi, yi, wi, hi)


def _fit_aspect(region: Rect, aspect: float, src_w: int, src_h: int) -> Rect:
    """Shrink `region` (centred) to `aspect` (w/h), even integer values, inside the source."""
    if region.w / region.h > aspect:
        h = region.h
        w = h * aspect
    else:
        w = region.w
        h = w / aspect
    h_e = _even_floor(h)
    w_e = _even_round(h_e * aspect)
    if w_e > region.w:  # rounding pushed us out; shrink the other way
        w_e = _even_floor(region.w)
        h_e = _even_round(w_e / aspect)
    cx = region.x + region.w / 2.0
    cy = region.y + region.h / 2.0
    x = int(_clamp(cx - w_e / 2.0, 0, max(0, src_w - w_e))) // 2 * 2
    y = int(_clamp(cy - h_e / 2.0, 0, max(0, src_h - h_e))) // 2 * 2
    return Rect(x, y, min(w_e, src_w), min(h_e, src_h))


def compute_layout(src_w: int, src_h: int, cfg: LayoutConfig,
                   captions: CaptionsConfig | None = None) -> Layout:
    """Compute all pixel geometry for one source resolution."""
    out_w, out_h = cfg.width, cfg.height
    header_h = max(0, cfg.header_height)
    gameplay_only = cfg.mode == "gameplay_only"
    facecam_h = 0 if gameplay_only else max(0, cfg.facecam_height)
    gameplay_y = header_h + facecam_h
    gameplay_h = out_h - gameplay_y

    facecam_crop: Rect | None = None
    if facecam_h > 0:
        region = facecam_region(src_w, src_h, cfg.facecam)
        facecam_crop = _fit_aspect(region, out_w / facecam_h, src_w, src_h)

    gp = cfg.gameplay
    zoom = max(1.0, gp.zoom)
    if gp.fit == "blur":
        w = src_w / zoom
        h = src_h / zoom
    else:
        aspect = out_w / gameplay_h
        h = src_h / zoom
        w = h * aspect
        if w > src_w:
            w = src_w
            h = w / aspect
    w_e = min(_even_floor(w), _even_floor(src_w))
    h_e = min(_even_floor(h), _even_floor(src_h))
    x = int(_clamp(gp.center_x * src_w - w_e / 2.0, 0, src_w - w_e)) // 2 * 2
    y = int(_clamp(gp.center_y * src_h - h_e / 2.0, 0, src_h - h_e)) // 2 * 2
    gameplay_crop = Rect(x, y, w_e, h_e)

    position = captions.position if captions else "seam"
    if position == "gameplay":
        cap_y = gameplay_y + 0.30 * gameplay_h
    elif position == "bottom":
        cap_y = out_h - 0.18 * out_h
    else:  # seam
        cap_y = header_h + gameplay_h * 0.2 if gameplay_only else header_h + facecam_h
    if captions:
        cap_y += captions.y_offset

    return Layout(
        src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h, mode=cfg.mode,
        header_h=header_h, facecam_h=facecam_h, gameplay_y=gameplay_y, gameplay_h=gameplay_h,
        facecam_crop=facecam_crop, gameplay_crop=gameplay_crop, gameplay_fit=gp.fit,
        caption_y=int(round(cap_y)),
        background_color=cfg.background_color,
    )
