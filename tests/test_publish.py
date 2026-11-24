import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from clipper import publish as pb
from clipper.cli import apply_overrides, build_parser
from clipper.config import AIConfig, Config, PublishConfig, load_config
from clipper.models import Word

ROOT = Path(__file__).resolve().parent.parent
CON = Console(quiet=True)


class FakeResponses:
    def __init__(self, payload, exc=None):
        self.payload, self.exc, self.n = payload, exc, 0

    def create(self, **kw):
        self.n += 1
        self.kw = kw
        if self.exc:
            raise self.exc
        return SimpleNamespace(output_text=json.dumps(self.payload),
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200))


def pcfg(**kw):
    base = dict(creator="xQc", creator_url="https://www.twitch.tv/xqc", game="GTA 6",
                hashtags=["#Shorts", "#GTA6"], ai_metadata=False)
    base.update(kw)
    return PublishConfig(**base)


def make_cfg(tmp_path, **kw):
    cfg = Config(base_dir=tmp_path)
    cfg.publish = pcfg(root=str(tmp_path / "pub"), **kw)
    return cfg


def make_output(tmp_path, clips):
    out = tmp_path / "out" / "vid"
    out.mkdir(parents=True)
    items = []
    for i, c in enumerate(clips):
        c = dict(c)
        c.setdefault("file", f"vid_{i}.mp4")
        c.setdefault("start", 100.0 * (i + 1))
        c.setdefault("end", c["start"] + 40)
        c.setdefault("hook", f"Hook number {i}")
        c.setdefault("category", "funny")
        c.setdefault("transcript", "hello there")
        c.setdefault("error", None)
        if c["file"]:
            (out / c["file"]).write_bytes(b"mp4data")
            (out / c["file"]).with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        items.append(c)
    (out / "clips.json").write_text(json.dumps({"video": "vid.mp4", "clips": items}))
    return out / "clips.json"


def test_safe_name():
    assert pb.safe_name('a<b>c:d"e/f\\g|h?i*j') == "abcdefghij"
    assert pb.safe_name("hello...  ") == "hello"
    assert pb.safe_name("x" * 200) == "x" * 80
    assert pb.safe_name("???") == "untitled"
    assert pb.safe_name("CON") == "_CON"


def test_transcript_to_srt_format():
    words = [Word("Hello", 10.0, 10.4), Word("world.", 10.5, 11.0), Word("Next", 12.0, 12.5)]
    srt = pb.transcript_to_srt(words, 10.0)
    assert srt == ("1\n00:00:00,000 --> 00:00:01,000\nHello world.\n\n"
                   "2\n00:00:02,000 --> 00:00:02,500\nNext\n")
    assert pb.transcript_to_srt([], 0) == ""


def test_postprocess_title_and_tags():
    payload = {"title": '"WOW THIS IS INSANE #Shorts ' + "very " * 30 + '"',
               "description": "Line one\nLine two\n\nSubscribe whatever\n#foo",
               "tags": ["xQc", "XQC", "a" * 31, "#gta 6"] + [f"tag number {i}" for i in range(40)]}
    writer = pb.MetadataWriter(AIConfig(), ".", client=SimpleNamespace(responses=FakeResponses(payload)))
    meta = pb.generate_metadata({"hook": "x"}, pcfg(ai_metadata=True), AIConfig(), ".", writer=writer)
    assert len(meta["title"]) <= 70 and "#" not in meta["title"] and '"' not in meta["title"]
    assert not meta["title"].isupper()
    lines = meta["description"].splitlines()
    assert lines[:2] == ["Line one", "Line two"] and lines[2] == ""
    assert lines[3] == "\U0001F3A5 Credit: xQc — https://www.twitch.tv/xqc"
    assert lines[4] == "Subscribe for daily GTA 6 highlights: @YourChannel"
    for h in ("#Shorts", "#GTA6", "#xQc"):
        assert h in lines[5]
    tags = meta["tags"]
    assert len(tags) <= 15 and len(", ".join(tags)) <= 450 and all(len(t) <= 30 for t in tags)
    assert len({t.casefold() for t in tags}) == len(tags)


def test_ai_cost_and_budget():
    resp = FakeResponses({"title": "T", "description": "a\nb", "tags": ["x"]})
    w = pb.MetadataWriter(AIConfig(max_cost_usd=0.0002), ".", client=SimpleNamespace(responses=resp))
    assert w.write({"hook": "h"}) is not None
    assert w.spent_usd > 0 and resp.kw["text"]["format"]["strict"] is True
    assert w.write({"hook": "h"}) is None and resp.n == 1       # budget reached


def test_fallback_template():
    meta = pb.generate_metadata({"hook": "He Fell Off The Roof", "category": "fail", "reason": "Big fall."},
                                pcfg(), AIConfig(), ".")
    assert meta["title"] == "He Fell Off The Roof | xQc GTA 6"
    d = meta["description"].splitlines()
    assert d[0] == "He Fell Off The Roof" and d[1] == "Big fall."
    assert "#Shorts" in d[-1] and "#xQc" in d[-1]
    assert "xQc" in meta["tags"] and "GTA 6" in meta["tags"]
    long = pb.generate_metadata({"hook": "word " * 30}, pcfg(), AIConfig(), ".")
    assert len(long["title"]) <= 70


def test_export_layout_numbering_idempotent_manifest(tmp_path):
    cj = make_output(tmp_path, [{}, {}])
    cfg = make_cfg(tmp_path)
    pk = pb.export_clips(cj, cfg, CON)
    assert len(pk) == 2
    auto = tmp_path / "pub"
    base = auto / "xQc"
    names = sorted(d.name for d in base.iterdir())
    assert names[0].startswith("001 - ") and names[1].startswith("002 - ")
    f = base / names[0]
    for n in ("title.txt", "description.txt", "tags.txt", "captions.srt", "metadata.json"):
        assert (f / n).is_file()
    meta = json.loads((f / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "ready" and (f / meta["video_file"]).is_file()
    assert (f / "captions.srt").read_text().startswith("1\n")
    man = json.loads((auto / "manifest.json").read_text(encoding="utf-8"))
    assert len(man) == 2 and man[0]["folder"].startswith("xQc/001 - ")
    rows = list(csv.DictReader((auto / "manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 2 and rows[0]["id"] == man[0]["id"]
    assert (auto / "HANDOVER.md").is_file()
    pb.export_clips(cj, cfg, CON)                      # idempotent
    assert len(json.loads((auto / "manifest.json").read_text(encoding="utf-8"))) == 2
    assert len(list(base.iterdir())) == 2
    cj2 = make_output(tmp_path / "b", [{"start": 999.0}])   # a new run appends
    pb.export_clips(cj2, cfg, CON)
    assert sorted(d.name for d in base.iterdir())[-1].startswith("003 - ")
    assert len(json.loads((auto / "manifest.json").read_text(encoding="utf-8"))) == 3


def test_flat_layout(tmp_path):
    cj = make_output(tmp_path, [{}])
    pk = pb.export_clips(cj, make_cfg(tmp_path), CON)
    assert pk[0]["folder"].startswith("xQc/001 - ")
    assert (tmp_path / "pub" / pk[0]["folder"] / "title.txt").is_file()


def test_export_skips_weak_and_failed(tmp_path):
    cj = make_output(tmp_path, [{"weak": True}, {"error": "boom", "file": None}, {}])
    assert len(pb.export_clips(cj, make_cfg(tmp_path), CON)) == 1
    assert len(pb.export_clips(cj, make_cfg(tmp_path, skip_weak=False), CON)) == 2


def test_export_requires_creator(tmp_path):
    cj = make_output(tmp_path, [{}])
    with pytest.raises(ValueError):
        pb.export_clips(cj, make_cfg(tmp_path, creator=""), CON)


def test_cli_parse_export_and_no_publish():
    a = build_parser().parse_args(["export", "output/x", "--creator", "A", "--creator-url", "u",
                                   "--game", "G", "--root", "r", "--include-weak"])
    cfg = apply_overrides(Config(), a)
    assert cfg.publish.creator == "A" and cfg.publish.creator_url == "u" and cfg.publish.game == "G"
    assert cfg.publish.skip_weak is False and Path(cfg.publish.root).is_absolute()
    r = apply_overrides(Config(), build_parser().parse_args(["run", "--no-publish"]))
    assert r.publish.enabled is False


def test_configs_load_with_publish_fields():
    cfg = load_config([ROOT / "configs/creators/xqc_gta6.yaml", ROOT / "configs/games/gta6.yaml"])
    p = cfg.publish
    assert (p.creator, p.creator_url, p.game) == ("xQc", "https://www.twitch.tv/xqc", "GTA 6")
    assert p.hashtags == ["#Shorts", "#GTA6", "#GTAVI"]
    for f in (ROOT / "configs/creators").glob("*.yaml"):
        assert load_config(f).publish.creator
    assert load_config(ROOT / "configs/preset_top_left.yaml").publish.creator == ""


def test_description_includes_cta_line_after_credit():
    from clipper.config import PublishConfig
    from clipper.publish import _build_description
    cfg = PublishConfig(creator="Tomo", creator_url="https://www.youtube.com/@Tomovids", game="GTA 6",
                        channel="@X", cta="Subscribe to Tomo on YouTube: @Tomovids")
    lines = _build_description(["line one", "line two"], cfg, []).splitlines()
    assert lines[3].startswith("🎥 Credit: Tomo") and lines[4] == "Subscribe to Tomo on YouTube: @Tomovids"
    assert "Subscribe to" not in _build_description(["a"], PublishConfig(creator="T"), []).split("Credit")[0]


def test_handover_names_its_own_drive_folder_and_channel():
    from clipper.config import PublishConfig
    txt = pb.handover_text(PublishConfig(channel="@MyClipsChannel", drive_folder="MyClipsFolder"))
    assert 'My Drive/MyClipsFolder' in txt and "@MyClipsChannel" in txt
    assert "Automation/Shorts" not in txt and "@@" not in txt
    assert "My Drive" not in pb.handover_text(PublishConfig(drive_folder=""))
