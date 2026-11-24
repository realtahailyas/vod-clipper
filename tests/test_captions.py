from __future__ import annotations

import re

import pytest

from clipper.captions import ass_color, ass_time, build_ass, chunk_words, escape_text
from clipper.config import CaptionsConfig, LayoutConfig, TitleConfig
from clipper.layout import compute_layout
from clipper.models import Segment, Transcript, Word


def W(text, s, e):
    return Word(text, s, e)


def _layout(header=140):
    return compute_layout(1920, 1080, LayoutConfig(header_height=header))


def _secs(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _events(ass):
    out = []
    for line in ass.splitlines():
        if line.startswith("Dialogue:"):
            f = line.split(",", 9)
            out.append((f[3], _secs(f[1]), _secs(f[2]), f[9]))
    return out


def _plain(text):
    return re.sub(r"\{[^}]*\}", "", text)


def test_color_conversion():
    assert ass_color("#FF0000") == "&H000000FF"
    assert ass_color("#FFE600") == "&H0000E6FF"
    assert ass_color("#112233") == "&H00332211"
    with pytest.raises(ValueError):
        ass_color("red")


def test_time_format():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(3661.239) == "1:01:01.24"
    assert ass_time(59.999) == "0:01:00.00"


def test_escape():
    assert escape_text("a{b}c\\d") == "a\\{b\\}c\\\\d"


def test_chunk_max_words():
    ws = [W(f"w{i}", i * 0.3, i * 0.3 + 0.2) for i in range(7)]
    ch = chunk_words(ws, 3, 99)
    assert [len(c) for c in ch] == [3, 3, 1]


def test_chunk_max_chars():
    ws = [W("hello", 0, 0.2), W("wonderful", 0.25, 0.5), W("world", 0.55, 0.8)]
    ch = chunk_words(ws, 5, 15)
    assert [[w.text for w in c] for c in ch] == [["hello", "wonderful"], ["world"]]


def test_chunk_gap_and_sentence_end():
    ws = [W("a", 0, 0.2), W("b", 1.5, 1.7), W("c.", 1.75, 1.9), W("d", 1.95, 2.1)]
    ch = chunk_words(ws, 5, 99)
    assert [[w.text for w in c] for c in ch] == [["a"], ["b", "c."], ["d"]]


def _tr():
    words = [W("hello", 10.0, 10.3), W("there", 10.35, 10.7), W("big", 10.72, 11.0),
             W("world.", 11.05, 11.5), W("next", 14.0, 14.3), W("one", 14.31, 14.6)]
    return Transcript(segments=[Segment("x", 10.0, 14.6, words)])


def test_events_relative_and_no_overlap():
    lay = _layout()
    ass = build_ass(_tr(), 9.0, 20.0, lay, CaptionsConfig())
    ev = [e for e in _events(ass) if e[0] == "Caption"]
    assert ev
    assert ev[0][1] == pytest.approx(1.0, abs=0.011)  # 10.0 - 9.0
    assert all(e[1] < e[2] for e in ev)
    for a, b in zip(ev, ev[1:]):
        assert a[2] <= b[1] + 1e-9, (a, b)
    first_chunk = [e for e in ev if _plain(e[3]) == "HELLO THERE BIG"]
    assert len(first_chunk) == 3
    for a, b in zip(first_chunk, first_chunk[1:]):
        assert a[2] == b[1]  # touch exactly: no gap, no overlap
    assert ev[-1][2] <= 11.0 + 1e-9  # last chunk lingers <= 0.6 s past last word


def test_active_word_markup_and_reset():
    lay = _layout()
    ass = build_ass(_tr(), 9.0, 20.0, lay, CaptionsConfig())
    ev = [e for e in _events(ass) if e[0] == "Caption"]
    t = ev[0][3]
    assert t.startswith("{\\an5\\pos(540,%d)}" % lay.caption_y)
    assert "\\c&H00E6FF&" in t and "\\t(0,90,\\fscx118\\fscy118)" in t
    assert t.index("HELLO") < t.index("{\\r}") < t.index("THERE")
    ass2 = build_ass(_tr(), 9.0, 20.0, lay, CaptionsConfig(pop=False))
    assert "\\fscx118" not in ass2


def test_words_outside_clip_dropped():
    lay = _layout()
    ass = build_ass(_tr(), 13.0, 15.0, lay, CaptionsConfig())
    plain = " ".join(_plain(e[3]) for e in _events(ass))
    assert "HELLO" not in plain and "NEXT" in plain
    assert _events(ass)[0][1] == pytest.approx(1.0, abs=0.011)


def test_header_structure():
    ass = build_ass(_tr(), 9.0, 20.0, _layout(), CaptionsConfig())
    assert "ScriptType: v4.00+" in ass and "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "WrapStyle: 2" in ass and "ScaledBorderAndShadow: yes" in ass
    fmt = next(ln for ln in ass.splitlines() if ln.startswith("Format: Name"))
    assert len(fmt[len("Format: "):].split(",")) == 23
    styles = [ln for ln in ass.splitlines() if ln.startswith("Style:")]
    assert len(styles) == 2
    for st in styles:
        assert len(st[len("Style: "):].split(",")) == 23
    assert "Style: Caption,Arial Black,84,&H00FFFFFF" in ass


def test_disabled_captions_title_only():
    ass = build_ass(_tr(), 9.0, 20.0, _layout(), CaptionsConfig(enabled=False), TitleConfig(), "Hi")
    assert [e[0] for e in _events(ass)] == ["Title"]
    none = build_ass(_tr(), 9.0, 20.0, _layout(), CaptionsConfig(enabled=False))
    assert _events(none) == []


def test_no_transcript():
    ass = build_ass(None, 0, 10, _layout(), CaptionsConfig(), TitleConfig(), "Title")
    assert [e[0] for e in _events(ass)] == ["Title"]


def test_title_position_and_wrapping():
    lay = _layout(header=140)
    t = _events(build_ass(None, 0, 10, lay, CaptionsConfig(), TitleConfig(), "short"))[0][3]
    assert "\\pos(540,70)" in t and "\\N" not in t and "SHORT" in t
    long = "this is a really long title that must wrap onto two lines"
    t = _events(build_ass(None, 0, 10, lay, CaptionsConfig(), TitleConfig(), long))[0][3]
    assert t.count("\\N") == 1
    fs = int(re.search(r"\\fs(\d+)", t).group(1))
    assert fs <= 60
    longest = max(len(x) for x in _plain(t).split("\\N"))
    assert 0.6 * fs * longest <= 0.9 * 1080 + 1
    lay0 = _layout(header=0)
    t = _events(build_ass(None, 0, 10, lay0, CaptionsConfig(), TitleConfig(), "x"))[0][3]
    assert "\\pos(540,%d)" % round(0.06 * 1920) in t


def test_title_and_word_escaping():
    tr = Transcript(segments=[Segment("", 0, 1, [W("a{b}", 0.0, 0.5)])])
    ass = build_ass(tr, 0, 2, _layout(), CaptionsConfig(), TitleConfig(), "x{y}")
    assert "A\\{B\\}" in ass and "X\\{Y\\}" in ass


# ---------------------------------------------------------------------------
# watermark

from clipper.captions import ass_alpha, watermark_position  # noqa: E402
from clipper.config import WatermarkConfig  # noqa: E402


def _wm_events(ass):
    return [e for e in _events(ass) if e[0] == "Watermark"]


def _wm_ass(wm, layout=None, caps=None, tr=None):
    layout = layout or compute_layout(1920, 1080, LayoutConfig(), CaptionsConfig())
    return build_ass(tr, 0.0, 10.0, layout, caps or CaptionsConfig(), None, "", wm)


def test_alpha_conversion():
    assert ass_alpha(0.38) == "&H9E"
    assert ass_alpha(1.0) == "&H00"
    assert ass_alpha(0.0) == "&HFF"


def test_watermark_style_and_dialogue():
    ass = _wm_ass(WatermarkConfig())
    style = next(ln for ln in ass.splitlines() if ln.startswith("Style: Watermark"))
    f = style.split(",")
    assert f[3] == "&H9EFFFFFF" and f[5] == "&H9E000000" and f[6] == "&H9E000000"
    assert f[7] == "-1"
    evs = _wm_events(ass)
    assert len(evs) == 1
    assert evs[0][1] == 0 and evs[0][2] == pytest.approx(10.0)
    line = next(ln for ln in ass.splitlines() if ",Watermark,,0,0,0,," in ln)
    assert line.startswith("Dialogue: 5,")
    assert "@YourChannel" in evs[0][3] and r"\an3\pos(" in evs[0][3]


def test_watermark_absent():
    assert "Watermark" not in _wm_ass(None)
    assert "Watermark" not in _wm_ass(WatermarkConfig(enabled=False))
    assert "Watermark" not in _wm_ass(WatermarkConfig(text="  "))


def test_watermark_only_without_captions():
    caps = CaptionsConfig(enabled=False)
    ass = _wm_ass(WatermarkConfig(), caps=caps)
    assert len(_wm_events(ass)) == 1 and len(_events(ass)) == 1


def test_watermark_escaping():
    ass = _wm_ass(WatermarkConfig(text=r"a{b}\c"))
    assert _wm_events(ass)[0][3].endswith(r"a\{b\}\\c")


def test_watermark_position_default_collision():
    layout = compute_layout(1920, 1080, LayoutConfig(), CaptionsConfig())
    assert layout.mode == "split"
    seam = layout.header_h + layout.facecam_h
    caps = CaptionsConfig()
    x, y, an = watermark_position(layout, WatermarkConfig(), caps)
    assert (x, an) == (1080 - 18, 3)
    assert abs(layout.caption_y - seam) < caps.font_size * 1.5
    assert y == round(layout.caption_y - caps.font_size * 0.75 - 10) < seam - 18


def test_watermark_position_no_collision():
    layout = compute_layout(1920, 1080, LayoutConfig(), CaptionsConfig())
    caps = CaptionsConfig(position="bottom")
    layout = compute_layout(1920, 1080, LayoutConfig(), caps)
    x, y, an = watermark_position(layout, WatermarkConfig(), caps)
    assert y == layout.header_h + layout.facecam_h - 18 and an == 3


def test_watermark_position_other_modes():
    layout = compute_layout(1920, 1080, LayoutConfig(), CaptionsConfig())
    caps = CaptionsConfig()
    wm = WatermarkConfig(position="facecam_top_right")
    assert watermark_position(layout, wm, caps) == (1062, layout.header_h + 18 + 34, 3)
    wm = WatermarkConfig(position="header_right")
    assert watermark_position(layout, wm, caps) == (1062, layout.header_h // 2, 6)
    wm = WatermarkConfig(position="bottom_right")
    assert watermark_position(layout, wm, caps) == (1062, 1920 - 384, 3)
    wm = WatermarkConfig(position="custom", x=500, y=600)
    assert watermark_position(layout, wm, caps) == (500, 600, 3)
    nohdr = compute_layout(1920, 1080, LayoutConfig(header_height=0), caps)
    wm = WatermarkConfig(position="header_right")
    assert watermark_position(nohdr, wm, caps) == (1062, 0 + 18 + 34, 3)


def test_watermark_position_gameplay_only_fallback():
    cfg_l = LayoutConfig()
    cfg_l.mode = "gameplay_only"
    layout = compute_layout(1920, 1080, cfg_l, CaptionsConfig())
    assert layout.facecam_h == 0
    assert watermark_position(layout, WatermarkConfig(), CaptionsConfig()) == (1062, 1536, 3)


def test_censor_text_masks_slurs_and_strong_swears():
    from clipper.captions import censor_text
    assert censor_text("fuck!") == "f***!"
    assert censor_text("Motherfucker") == "M***********"
    assert censor_text("oh shit") == "oh sh**"
    assert censor_text("nigga") == "n****" and censor_text("NIGGAS,") == "N*****,"
    assert censor_text("wtf") == "w*f"
    for clean in ["hello", "class", "assess", "shiitake", "Dickens", "scunthorpe", "shift", "fun"]:
        assert censor_text(clean) == clean, clean


def test_srt_censor_flag():
    from clipper.models import Word
    from clipper.publish import transcript_to_srt
    ws = [Word("oh", 0.0, 0.4), Word("fuck", 0.4, 0.8)]
    assert "f***" in transcript_to_srt(ws, 0.0, censor=True)
    assert "fuck" in transcript_to_srt(ws, 0.0)
