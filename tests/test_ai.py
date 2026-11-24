import json
import math
import subprocess
from types import SimpleNamespace

import pytest

from clipper import media
from clipper.ai import ranker as rk
from clipper.ai.ranker import AIRanker, AIVerdict, build_contact_sheet, candidate_pool, final_scores, select_final
from clipper.config import AIConfig, Config, ConfigError, load_config, validate
from clipper.models import Highlight
from clipper.pipeline import format_title

GOOD = {"score": 8.5, "keep": True, "hook": "He did NOT expect that", "category": "chaos",
        "streamer_reaction": True, "reason": "Loud scream then crash."}


class FakeResponses:
    def __init__(self, text, tin=1000, tout=100, exc=None):
        self.text, self.tin, self.tout, self.exc, self.n = text, tin, tout, exc, 0

    def create(self, **kw):
        self.n += 1
        self.kw = kw
        if self.exc:
            raise self.exc
        return SimpleNamespace(output_text=self.text,
                               usage=SimpleNamespace(input_tokens=self.tin, output_tokens=self.tout))


def make(text, cfg=None, **kw):
    resp = FakeResponses(text, **kw)
    r = AIRanker(cfg or AIConfig(), ".", client=SimpleNamespace(responses=resp))
    return r, resp


def v(score, keep=True):
    return AIVerdict(score, keep, "h", "funny", True, "r")


def hl(score, start=0.0):
    return Highlight(start, start + 40, start + 20, score)


# ---------------------------------------------------------------- judge


def test_verdict_parsing_and_request_shape(tmp_path):
    sheet = tmp_path / "s.jpg"
    sheet.write_bytes(b"\xff\xd8fake")
    r, resp = make(json.dumps(GOOD))
    out = r.judge(sheet, "hello", {"video": "a.mp4", "timestamp": "0:01:00", "duration": 40})
    assert out == AIVerdict(8.5, True, "He did NOT expect that", "chaos", True, "Loud scream then crash.")
    assert r.calls == 1 and r.failures == 0
    assert r.spent_usd == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)
    kw = resp.kw
    assert kw["model"] == "gpt-6-luna" and kw["reasoning"] == {"effort": "low"}
    fmt = kw["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "clip_verdict"
    img = kw["input"][1]["content"][1]
    assert img["type"] == "input_image" and img["detail"] == "low"
    assert img["image_url"].startswith("data:image/jpeg;base64,")


def test_hook_is_cleaned_and_score_clamped():
    bad = dict(GOOD, score=14, hook='"One two #wow three four five six seven eight nine ten eleven"',
               category="weird")
    r, _ = make(json.dumps(bad))
    out = r.judge(None, "", {})
    assert out.score == 10.0 and out.category == "other"
    assert out.hook == "One two three four five six seven eight nine ten"  # quotes/hashtag gone, capped at 10


@pytest.mark.parametrize("text", ["not json", "[]", json.dumps({"score": 5}), json.dumps(dict(GOOD, score="high"))])
def test_invalid_json_returns_none_and_counts_failure(text):
    r, _ = make(text)
    assert r.judge(None, "", {}) is None
    assert r.failures == 1 and r.calls == 1


def test_api_error_returns_none():
    r, _ = make("", exc=RuntimeError("boom"))
    assert r.judge(None, "", {}) is None
    assert r.failures == 1


def test_budget_stop_skips_call():
    cfg = AIConfig(max_cost_usd=0.001)
    r, resp = make(json.dumps(GOOD), cfg, tin=100000, tout=1000)  # ~0.0105 per call
    assert r.judge(None, "", {}) is not None
    assert r.spent_usd >= cfg.max_cost_usd
    assert r.judge(None, "", {}) is None
    assert resp.n == 1


def test_transcript_truncated():
    r, resp = make(json.dumps(GOOD))
    r.judge(None, "word " * 2000, {})
    text = resp.kw["input"][1]["content"][0]["text"]
    assert len(text) < 1800


# ---------------------------------------------------------------- key lookup


def test_env_file_parsing(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# c\n\nA=1\nexport B = \"two words\"\nC='x' \nD=val # note\nbroken\n", encoding="utf-8")
    assert rk.parse_env_file(f) == {"A": "1", "B": "two words", "C": "x", "D": "val"}
    assert rk.parse_env_file(tmp_path / "nope") == {}


def test_key_lookup_env_then_dotenv(tmp_path, monkeypatch):
    cfg = AIConfig(api_key_env="CLIPPER_TEST_KEY")
    monkeypatch.delenv("CLIPPER_TEST_KEY", raising=False)
    assert rk.find_api_key(cfg, tmp_path) is None
    (tmp_path / ".env").write_text("CLIPPER_TEST_KEY=from-file\n", encoding="utf-8")
    assert rk.find_api_key(cfg, tmp_path) == "from-file"
    monkeypatch.setenv("CLIPPER_TEST_KEY", "from-env")
    assert rk.find_api_key(cfg, tmp_path) == "from-env"


def test_missing_key_makes_ranker_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "openai_installed", lambda: True)
    monkeypatch.delenv("CLIPPER_TEST_KEY", raising=False)
    r = AIRanker(AIConfig(api_key_env="CLIPPER_TEST_KEY"), tmp_path)
    assert r.available is False and "key" in r.unavailable_reason
    assert r.judge(None, "", {}) is None and r.calls == 0


def test_missing_package_makes_ranker_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(rk, "openai_installed", lambda: False)
    r = AIRanker(AIConfig(), tmp_path)
    assert r.available is False and "openai" in r.unavailable_reason


# ---------------------------------------------------------------- scoring


def test_final_scores_normalises_audio_and_handles_none():
    out = final_scores([2.0, 4.0, 1.0], [v(10), v(0), None], 0.75)
    assert out[0] == pytest.approx(0.75 * 10 + 0.25 * 5)
    assert out[1] == pytest.approx(0.25 * 10)
    assert out[2] == pytest.approx(2.5)          # audio-only: 10 * 1/4
    assert final_scores([0.0], [None], 0.5) == [0.0]


def test_select_final_drops_low_keeps_none_orders():
    cfg = AIConfig(min_score=4.0, weight=0.75, fill=False)
    cands = [hl(1.0), hl(1.0), hl(0.2), hl(1.0), hl(1.0)]
    verdicts = [v(9), v(3.9), None, v(6), v(4.0)]
    assert select_final(cands, verdicts, 10, cfg) == [(0, False), (3, False), (4, False), (2, False)]
    assert select_final(cands, verdicts, 2, cfg) == [(0, False), (3, False)]


def test_select_final_fill_tops_up_with_weak():
    cands = [hl(1.0), hl(1.0), hl(1.0), hl(1.0), hl(1.0)]
    verdicts = [v(9), v(3.0), v(2.0), v(3.9), v(1.0)]
    fill = AIConfig(min_score=4.0, weight=0.75, fill=True)
    assert select_final(cands, verdicts, 3, fill) == [(0, False), (3, True), (1, True)]
    assert select_final(cands, verdicts, 10, fill) == [(0, False), (3, True), (1, True), (2, True), (4, True)]
    assert select_final(cands, verdicts, 1, fill) == [(0, False)]
    strict = AIConfig(min_score=4.0, weight=0.75, fill=False)
    assert select_final(cands, verdicts, 3, strict) == [(0, False)]


def test_build_system_prompt_story_scenes():
    from clipper.ai.ranker import SYSTEM_PROMPT, build_system_prompt
    assert build_system_prompt(AIConfig()) == SYSTEM_PROMPT
    p = build_system_prompt(AIConfig(story_scenes=True))
    assert p.startswith(SYSTEM_PROMPT) and "story cutscenes" in p and "newly released" in p
    assert "story" in rk.CATEGORIES and "story" in rk.VERDICT_SCHEMA["properties"]["category"]["enum"]


# ---------------------------------------------------------------- pool / config / title


def test_candidate_pool():
    cfg = AIConfig()
    assert candidate_pool(25, cfg) == math.ceil(25 * 2.5) == 63
    assert candidate_pool(100, cfg) == 120
    assert candidate_pool(40, AIConfig(candidate_multiplier=1.0)) == 40


@pytest.mark.parametrize("raw,expected", [("auto", "auto"), ("TRUE", "true"), ("False", "false")])
def test_ai_enabled_values(raw, expected):
    cfg = Config()
    cfg.ai.enabled = raw
    assert validate(cfg).ai.enabled == expected


def test_ai_enabled_yaml_booleans_and_invalid(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("ai:\n  enabled: true\n", encoding="utf-8")
    assert load_config(p).ai.enabled == "true"
    p.write_text("ai:\n  enabled: false\n", encoding="utf-8")
    assert load_config(p).ai.enabled == "false"
    p.write_text("ai:\n  enabled: maybe\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ai.enabled"):
        load_config(p)


@pytest.mark.parametrize("field,value", [("candidate_multiplier", 0.5), ("frames", 0), ("frames", 10),
                                         ("weight", 1.5), ("min_score", 11), ("concurrency", 0),
                                         ("image_detail", "ultra"), ("reasoning_effort", "max"),
                                         ("timeout", 0), ("max_cost_usd", -1)])
def test_ai_numeric_validation(field, value):
    cfg = Config()
    setattr(cfg.ai, field, value)
    with pytest.raises(ConfigError, match=f"ai.{field}"):
        validate(cfg)


def test_format_title_hook():
    assert format_title("{hook} #{n}", 2, "s", "Wild") == "Wild #2"
    assert format_title("{hook}", 2, "s") == ""


# ---------------------------------------------------------------- contact sheet


def _probe_size(path):
    out = subprocess.run([media.ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


@pytest.fixture(scope="module")
def testvideo(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "t.mp4"
    media.run([media.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=6",
               "-pix_fmt", "yuv420p", str(p)])
    return p


@pytest.mark.parametrize("frames,grid", [(4, (2, 2)), (1, (1, 1)), (6, (3, 2)), (2, (2, 1))])
def test_contact_sheet_grid(testvideo, tmp_path, frames, grid):
    out = build_contact_sheet(testvideo, 0.0, 6.0, tmp_path / "x" / "s.jpg", frames=frames, width=320)
    assert out.exists() and out.stat().st_size > 0
    w, h = _probe_size(out)
    assert w == 320 * grid[0] and h == 180 * grid[1]


# ---------------------------------------------------------------- pipeline integration (fake client)


def test_pipeline_ai_path_end_to_end(tmp_path):
    from rich.console import Console
    from clipper import pipeline

    vid = tmp_path / "in" / "s.mp4"
    vid.parent.mkdir()
    media.run([media.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=240",
               "-f", "lavfi", "-i", "sine=frequency=300:duration=240", "-af",
               "volume='1+3*lt(mod(t,60),8)':eval=frame", "-pix_fmt", "yuv420p", "-shortest", str(vid)])
    cfg = Config(base_dir=tmp_path)
    cfg.paths.output_dir = str(tmp_path / "out")
    cfg.paths.cache_dir = str(tmp_path / "cache")
    h = cfg.highlights
    h.num_clips, h.skip_start, h.skip_end, h.min_gap, h.snap_to_sentences = 2, 10, 10, 30, False
    cfg.captions.enabled = False
    cfg.layout.mode = "gameplay_only"
    cfg.render.encoder = "libx264"
    cfg.ai.enabled = "true"
    cfg.ai.concurrency = 2
    validate(cfg)
    resp = FakeResponses(json.dumps(GOOD))
    ranker = AIRanker(cfg.ai, tmp_path, client=SimpleNamespace(responses=resp))
    res = pipeline.process_video(vid, cfg, None, "libx264", Console(quiet=True), ranker=ranker)
    assert 1 <= len(res) <= 2 and all(r.error is None for r in res)
    assert resp.n >= len(res)                     # judged at least the kept ones (pool > n)
    out = tmp_path / "out" / "s"
    clips = json.loads((out / "clips.json").read_text(encoding="utf-8"))["clips"]
    assert clips[0]["hook"] == "He did NOT expect that" and clips[0]["ai_score"] == 8.5
    assert {"category", "reason", "final_score"} <= set(clips[0])
    cands = json.loads((out / "candidates.json").read_text(encoding="utf-8"))["candidates"]
    assert len(cands) == resp.n and cands[0]["verdict"]["score"] == 8.5


def test_clean_hook_keeps_apostrophes_and_full_phrase():
    from clipper.ai.ranker import _clean_hook
    assert _clean_hook("He’s trapped in the tightest crawlspace") == "He's trapped in the tightest crawlspace"
    assert _clean_hook('"He couldn\'t explain it" #gaming 💀') == "He couldn't explain it"
    assert _clean_hook("He realizes this has to be a trap") == "He realizes this has to be a trap"


def test_facecam_slot_parsed_with_fallbacks():
    from clipper.ai.ranker import parse_verdict, VERDICT_SCHEMA
    assert "facecam_slot" in VERDICT_SCHEMA["required"]
    assert parse_verdict(json.dumps(dict(GOOD, facecam_slot=2))).facecam_slot == 2
    assert parse_verdict(json.dumps(dict(GOOD, facecam_slot=0))).facecam_visible is False
    assert parse_verdict(json.dumps(dict(GOOD, facecam_visible=False))).facecam_slot == 0  # legacy field
    assert parse_verdict(json.dumps(GOOD)).facecam_slot == 1   # no field -> primary position
    assert parse_verdict(json.dumps(dict(GOOD, facecam_slot=-3))).facecam_slot == 0


def test_parse_verdict_adult_content_flag():
    assert rk.parse_verdict(json.dumps({**GOOD, "facecam_slot": 1, "adult_content": True})).adult_content
    assert not rk.parse_verdict(json.dumps({**GOOD, "facecam_slot": 1, "adult_content": False})).adult_content
    assert not rk.parse_verdict(json.dumps(GOOD)).adult_content            # legacy answer: not flagged
    assert "adult_content" in rk.VERDICT_SCHEMA["required"]


def test_transcript_adult_terms():
    for t in ["bro we're in the strip club", "she is NAKED", "lap dance time", "a stripper on the pole",
              "why is she topless", "this is basically porn", "NIGHTCLUBS! bro no", "a nightclub",
              "girls in bikinis", "he's in his underwear"]:
        assert rk.transcript_is_adult(t), t
    for t in ["", "escort the convoy", "police chase on the highway", "sussex county", "nudge it",
              "that's a strip mall"]:
        assert not rk.transcript_is_adult(t), t


def test_select_final_never_picks_blocked_even_when_filling():
    cands = [hl(1.0)] * 4
    verdicts = [v(9), v(8), v(2), None]
    fill = AIConfig(min_score=4.0, weight=0.75, fill=True)
    assert sorted(select_final(cands, verdicts, 4, fill, blocked={0, 2})) == [(1, False), (3, False)]
    flagged = AIVerdict(9, True, "x", "chaos", True, "r", adult_content=True)
    assert rk.is_blocked(flagged, "") and rk.is_blocked(None, "strip club") and not rk.is_blocked(v(9), "car")
