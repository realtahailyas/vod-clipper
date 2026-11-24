from pathlib import Path

import pytest

from clipper.cli import apply_overrides, build_parser, main
from clipper.config import Config, ConfigError


def test_init_config_writes_and_refuses_overwrite(tmp_path):
    dest = tmp_path / "c.yaml"
    assert main(["init-config", str(dest)]) == 0
    assert dest.exists() and "highlights:" in dest.read_text(encoding="utf-8")
    assert main(["init-config", str(dest)]) == 1
    dest.write_text("x")
    assert main(["init-config", str(dest), "--force"]) == 0
    assert "highlights:" in dest.read_text(encoding="utf-8")


def test_doctor_returns_zero():
    assert main(["doctor"]) == 0


def test_apply_overrides():
    args = build_parser().parse_args([
        "run", "a.mp4", "-n", "3", "--min", "20", "--max", "45", "--facecam", "top-right",
        "--layout", "gameplay_only", "--title", "Clip {n}", "--no-captions", "--model", "tiny",
        "--encoder", "libx264", "--chat", "chat.json", "--keep-temp", "--dry-run",
    ])
    base = Config()
    cfg = apply_overrides(base, args)
    assert cfg.highlights.num_clips == 3
    assert cfg.highlights.min_duration == 20 and cfg.highlights.max_duration == 45
    assert cfg.layout.facecam.preset == "top-right"
    assert cfg.layout.mode == "gameplay_only"
    assert cfg.title.text == "Clip {n}"
    assert cfg.captions.enabled is False
    assert cfg.transcription.model == "tiny"
    assert cfg.render.encoder == "libx264"
    assert Path(cfg.highlights.chat.file).name == "chat.json"
    assert cfg.keep_temp is True
    assert base.highlights.num_clips == 25  # original untouched


def test_apply_overrides_none_keeps_config():
    args = build_parser().parse_args(["run"])
    assert apply_overrides(Config(), args) == Config()


def test_apply_overrides_validates():
    args = build_parser().parse_args(["run", "--min", "90", "--max", "30"])
    with pytest.raises(ConfigError):
        apply_overrides(Config(), args)


def test_bad_config_exit_code(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("nope: 1\n")
    assert main(["doctor", "-c", str(p)]) == 2


def test_large_clip_counts_and_ai_flags():
    for cmd in ("run", "analyze"):
        argv = [cmd, "a.mp4", "-n", "40"]
        cfg = apply_overrides(Config(), build_parser().parse_args(argv))
        assert cfg.highlights.num_clips == 40
    cfg = apply_overrides(Config(), build_parser().parse_args(["run", "-n", "30"]))
    assert cfg.highlights.num_clips == 30
    assert apply_overrides(Config(), build_parser().parse_args(["run", "--ai"])).ai.enabled == "true"
    assert apply_overrides(Config(), build_parser().parse_args(["run", "--no-ai"])).ai.enabled == "false"
    assert apply_overrides(Config(), build_parser().parse_args(["run"])).ai.enabled == "auto"


def test_config_flag_repeatable():
    args = build_parser().parse_args(["run", "-c", "a.yaml", "-c", "b.yaml"])
    assert args.config == ["a.yaml", "b.yaml"]
    assert build_parser().parse_args(["run", "-c", "a.yaml"]).config == ["a.yaml"]
    assert build_parser().parse_args(["run"]).config is None
    assert build_parser().parse_args(["analyze", "v.mp4", "-c", "a", "-c", "b"]).config == ["a", "b"]


def test_fill_strict_flags():
    from clipper.config import Config as C
    base = C()
    base.ai.fill = False
    assert apply_overrides(base, build_parser().parse_args(["run", "--fill"])).ai.fill is True
    assert apply_overrides(C(), build_parser().parse_args(["run", "--strict"])).ai.fill is False
    assert apply_overrides(base, build_parser().parse_args(["run"])).ai.fill is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--fill", "--strict"])


def test_watermark_flags():
    p = build_parser()
    cfg = apply_overrides(Config(), p.parse_args(["run", "a.mp4", "--watermark", "@Me"]))
    assert cfg.watermark.text == "@Me" and cfg.watermark.enabled
    cfg = apply_overrides(Config(), p.parse_args(["preview", "a.mp4", "--no-watermark"]))
    assert cfg.watermark.enabled is False
    assert Config().watermark.enabled is True
    with pytest.raises(SystemExit):
        p.parse_args(["run", "a.mp4", "--watermark", "x", "--no-watermark"])
