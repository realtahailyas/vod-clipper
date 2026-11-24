from pathlib import Path

import pytest
import yaml

from clipper.config import Config, ConfigError, load_config

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "config.yaml", ROOT / "src" / "clipper" / "default_config.yaml"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_shipped_config_equals_defaults(path):
    cfg = load_config(path)
    expected = Config(base_dir=cfg.base_dir)
    assert cfg == expected


def test_root_and_bundled_identical():
    assert FILES[0].read_bytes() == FILES[1].read_bytes()


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"highlights": {"num_clipz": 3}}))
    with pytest.raises(ConfigError, match="num_clipz"):
        load_config(p)


def test_unknown_top_level_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bogus: 1\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_bad_preset_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"layout": {"facecam": {"preset": "middle"}}}))
    with pytest.raises(ConfigError, match="preset"):
        load_config(p)


def test_partial_config_ok(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("highlights:\n  num_clips: 3\n")
    cfg = load_config(p)
    assert cfg.highlights.num_clips == 3
    assert cfg.base_dir == tmp_path.resolve()


# --- extends / cpu_threads ---

def _w(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


def test_extends_merge(tmp_path):
    _w(tmp_path / "base.yaml", {"highlights": {"num_clips": 3, "min_gap": 99.0},
                                "layout": {"facecam": {"size": [0.1, 0.1]}}, "keep_temp": True})
    child = _w(tmp_path / "child.yaml", {"extends": "base.yaml", "highlights": {"num_clips": 5},
                                         "layout": {"facecam": {"size": [0.3, 0.4]}}})
    cfg = load_config(child)
    assert cfg.highlights.num_clips == 5
    assert cfg.highlights.min_gap == 99.0
    assert cfg.layout.facecam.size == [0.3, 0.4]
    assert cfg.keep_temp is True


def test_extends_chain_and_base_dir(tmp_path):
    _w(tmp_path / "a" / "root.yaml", {"highlights": {"num_clips": 2, "hop": 1.0}})
    _w(tmp_path / "b" / "mid.yaml", {"extends": "../a/root.yaml", "highlights": {"hop": 2.0}})
    child = _w(tmp_path / "c" / "leaf.yaml", {"extends": str(tmp_path / "b" / "mid.yaml")})
    cfg = load_config(child)
    assert (cfg.highlights.num_clips, cfg.highlights.hop) == (2, 2.0)
    # unset default paths live next to the root config (project folder), not next to the preset
    assert cfg.base_dir == (tmp_path / "a").resolve()


def test_extends_paths_resolve_against_defining_file(tmp_path):
    _w(tmp_path / "config.yaml", {"paths": {"output_dir": "out"}})
    preset = _w(tmp_path / "configs" / "p.yaml",
                {"extends": "../config.yaml", "paths": {"cache_dir": "pcache"}})
    cfg = load_config(preset)
    assert cfg.resolve(cfg.paths.output_dir) == (tmp_path / "out").resolve()
    assert cfg.resolve(cfg.paths.cache_dir) == (tmp_path / "configs" / "pcache").resolve()
    assert cfg.resolve(cfg.paths.input_dir) == (tmp_path / "input").resolve()  # default


def test_extends_cycle(tmp_path):
    _w(tmp_path / "a.yaml", {"extends": "b.yaml"})
    _w(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_extends_missing_parent(tmp_path):
    child = _w(tmp_path / "c.yaml", {"extends": "nope.yaml"})
    with pytest.raises(ConfigError, match="nope.yaml"):
        load_config(child)


def test_cpu_threads_negative(tmp_path):
    p = _w(tmp_path / "c.yaml", {"transcription": {"cpu_threads": -1}})
    with pytest.raises(ConfigError, match="cpu_threads"):
        load_config(p)


def test_multi_config_merge_order_and_paths(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "one.yaml").write_text(yaml.safe_dump({
        "highlights": {"num_clips": 5, "min_gap": 10}, "ai": {"min_score": 5.0},
        "paths": {"output_dir": "out1"}}))
    (b / "two.yaml").write_text(yaml.safe_dump({
        "highlights": {"num_clips": 9}, "ai": {"fill": False, "story_scenes": True},
        "paths": {"cache_dir": "cache2"}}))
    cfg = load_config([a / "one.yaml", b / "two.yaml"])
    assert cfg.highlights.num_clips == 9           # later wins
    assert cfg.highlights.min_gap == 10            # earlier kept when not overridden
    assert cfg.ai.min_score == 5.0 and cfg.ai.fill is False and cfg.ai.story_scenes is True
    assert cfg.base_dir == a.resolve()             # base_dir from the first file
    assert cfg.resolve(cfg.paths.output_dir) == (a / "out1").resolve()
    assert Path(cfg.paths.cache_dir) == (b / "cache2").resolve()   # 2nd file: absolutized against its own folder


def test_multi_config_single_item_list_and_missing(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("highlights:\n  num_clips: 7\n")
    assert load_config([p]).highlights.num_clips == 7
    with pytest.raises(ConfigError):
        load_config([p, tmp_path / "nope.yaml"])


def test_game_profile_layers_on_creator_preset():
    cfg = load_config([ROOT / "configs" / "preset_top_left.yaml", ROOT / "configs" / "games" / "gta6.yaml"])
    assert cfg.layout.facecam.preset == "top-left"
    assert cfg.ai.story_scenes is True and cfg.ai.fill is True and cfg.ai.min_score == 3.5
    assert "GTA 6" in cfg.ai.context
    assert cfg.base_dir == ROOT


@pytest.mark.parametrize("body", ["opacity: 1.5", "opacity: -0.1", "font_size: 0", "position: nowhere"])
def test_watermark_validation(tmp_path, body):
    p = tmp_path / "c.yaml"
    p.write_text(f"watermark:\n  {body}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)
