"""Typed configuration loaded from a single YAML file.

Every field has a default, so an empty or partial config.yaml is valid.
Unknown keys raise ConfigError so typos don't silently do nothing.
"""
from __future__ import annotations

import dataclasses as dc
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class PathsConfig:
    input_dir: str = "input"
    output_dir: str = "output"
    temp_dir: str | None = None          # None -> system temp
    cache_dir: str = ".cache"            # audio-analysis cache (re-runs skip the long scan)


@dataclass
class WeightsConfig:
    loudness: float = 0.45               # overall level vs. local baseline
    vocal: float = 0.35                  # 300-3400 Hz speech-band energy (yelling, laughing)
    excitement: float = 0.20             # sudden upward jumps in loudness


@dataclass
class ChatConfig:
    enabled: bool = True
    file: str | None = None              # explicit chat file; otherwise auto-detected next to the video
    weight: float = 0.6                  # how much chat activity adds to the audio score
    window: float = 10.0                 # seconds; message-rate smoothing window
    delay: float = 4.0                   # chat reacts late: shift chat activity earlier by this many seconds


@dataclass
class HighlightsConfig:
    num_clips: int = 25
    min_duration: float = 35.0
    max_duration: float = 60.0
    pre_roll: float = 30.0               # seconds kept before the peak (build-up)
    post_roll: float = 15.0              # seconds kept after the peak (reaction)
    min_gap: float = 60.0                # min seconds between two selected peaks (60: back-to-back action in GTA-type streams)
    skip_start: float = 300.0            # ignore the first N seconds (starting-soon screen)
    skip_end: float = 120.0
    snap_window: float = 3.0             # max seconds a boundary may move to land on silence / sentence edge
    snap_to_sentences: bool = True       # use the transcript to avoid cutting mid-sentence
    audio_track: int = 0                 # audio stream index used for analysis (e.g. a mic-only track)
    hop: float = 0.5                     # analysis resolution in seconds
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)


FACECAM_PRESETS = ("top-left", "top-right", "bottom-left", "bottom-right", "custom")


@dataclass
class FacecamConfig:
    preset: str = "bottom-left"          # top-left | top-right | bottom-left | bottom-right | custom
    size: list[float] = field(default_factory=lambda: [0.26, 0.30])  # [w, h] as fraction of source, for presets
    margin: list[float] = field(default_factory=lambda: [0.0, 0.0])  # [x, y] inset from the corner, fraction of source
    region: list[float] | None = None    # custom [x, y, w, h]; values <= 1 are fractions, > 1 are pixels
    alternates: list[list[float]] | None = None  # other [x, y, w, h] cam spots the streamer switches to;
                                                 # with AI on, each clip uses the spot where the cam is seen


@dataclass
class GameplayConfig:
    zoom: float = 1.0                    # 1.0 = full source height; >1 zooms in
    center_x: float = 0.5                # horizontal focus point (0..1)
    center_y: float = 0.5
    fit: str = "crop"                    # crop (fill, cut sides) | blur (whole frame over blurred fill)


@dataclass
class LayoutConfig:
    width: int = 1080
    height: int = 1920
    mode: str = "split"                  # split | gameplay_only
    header_height: int = 140             # top band for the title/hook text (0 to disable)
    facecam_height: int = 620            # facecam section; gameplay gets the remaining height
    background_color: str = "#000000"
    facecam: FacecamConfig = field(default_factory=FacecamConfig)
    gameplay: GameplayConfig = field(default_factory=GameplayConfig)


@dataclass
class CaptionsConfig:
    enabled: bool = True
    font: str = "Arial Black"
    fonts_dir: str | None = None         # folder with custom .ttf/.otf files
    font_size: int = 84
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFE600"     # the word being spoken
    outline_color: str = "#000000"
    outline: float = 7.0
    shadow: float = 3.0
    uppercase: bool = True
    censor: bool = True                  # mask slurs and strong swears on screen and in the .srt (age-restriction risk)
    max_words: int = 3                   # words shown at once
    max_chars: int = 18
    position: str = "seam"               # seam | gameplay | bottom
    y_offset: int = 0                    # extra pixels, + moves down
    pop: bool = True                     # scale "pop" on the active word


@dataclass
class TitleConfig:
    text: str = ""                       # "" = no title. Placeholders: {n} clip number, {stem} video name, {hook} AI hook
    font: str = "Arial Black"
    font_size: int = 60
    color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline: float = 4.0
    uppercase: bool = True


WATERMARK_POSITIONS = ("facecam_bottom_right", "facecam_top_right", "header_right",
                       "bottom_right", "custom")


@dataclass
class WatermarkConfig:
    enabled: bool = True
    text: str = "@YourChannel"
    font: str = "Arial Black"
    font_size: int = 34
    color: str = "#FFFFFF"
    opacity: float = 0.38                # 0..1 (0.35-0.40 = subtle)
    outline_color: str = "#000000"
    outline: float = 2.0                 # thin dark outline
    shadow: float = 1.0
    position: str = "facecam_bottom_right"  # facecam_bottom_right | facecam_top_right | header_right | bottom_right | custom
    margin: int = 18                     # px from the edges of the chosen area
    x: int = 0                           # custom only: output pixels, anchor = bottom-right of the text
    y: int = 0


@dataclass
class PublishConfig:
    enabled: bool = True                 # after `clipper run`, package the clips into root (needs publish.creator)
    root: str = "publish"                # local folder; point it at a Google Drive for Desktop folder to auto-sync
    platform: str = "YouTube"
    game: str = ""                       # e.g. "GTA 6" (set by configs/games/*.yaml)
    creator: str = ""                    # display name, e.g. xQc (set by configs/creators/*.yaml)
    creator_url: str = ""                # credit link, e.g. https://www.twitch.tv/xqc
    channel: str = "@YourChannel"
    skip_weak: bool = True               # don't package clips the AI marked weak
    hashtags: list[str] = field(default_factory=lambda: ["#Shorts"])  # always added; game/creator tags are added automatically
    ai_metadata: bool = True             # write title/description/tags with the AI model (falls back to a template)
    cta: str = ""                        # extra description line after the credit, e.g. a campaign's required call to action
    drive_folder: str = "Automation/Shorts"  # Drive mirror named in HANDOVER.md ("" = none)


@dataclass
class TranscriptionConfig:
    model: str = "small.en"              # CPU: small.en / base.en; GPU: large-v3 / distil-large-v3
    device: str = "auto"                 # auto | cuda | cpu
    compute_type: str = "int8"           # int8 (CPU) | float16 (NVIDIA) | auto
    cpu_threads: int = 6                 # 0 = auto; caps whisper CPU threads so batch runs don't lock up / throttle the laptop
    language: str | None = "en"          # "en" for *.en models; None = auto-detect
    beam_size: int = 1
    vad_filter: bool = False               # VAD drops screams buried under game audio


ENCODERS = ("auto", "h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_amf", "hevc_amf", "libx264", "libx265")


@dataclass
class RenderConfig:
    encoder: str = "auto"                # auto picks the first working HW encoder, else libx264
    quality: int = 21                    # CRF / CQ (lower = better, bigger)
    fps: float | None = None             # None keeps source fps (capped at 60)
    audio_bitrate: str = "192k"
    audio_track: int = 0                 # audio stream used in the output clip
    normalize_audio: bool = True         # EBU R128 loudnorm to -14 LUFS (platform loudness)
    hwaccel: str = "none"                # decode acceleration: none | auto | cuda | qsv | d3d11va ...
    workers: int = 1                     # parallel ffmpeg renders


AI_ENABLED = ("auto", "true", "false")


@dataclass
class AIConfig:
    enabled: str = "auto"                # auto = on when the API key is found | true | false
    model: str = "gpt-6-luna"            # cheapest current OpenAI model with image input ($0.10 in / $0.50 out per 1M)
    api_key_env: str = "OPENAI_API_KEY"  # env var (or .env entry) holding the key
    reasoning_effort: str = "low"        # none | low | medium | high
    candidate_multiplier: float = 2.5    # judge ~2.5x num_clips candidates (25 clips -> 63 candidates)
    max_candidates: int = 120            # hard cap on judged candidates
    frames: int = 4                      # frames per candidate, combined into one contact sheet
    image_detail: str = "low"            # low | high | auto
    min_score: float = 4.0               # drop clips the AI scores below this (0-10)
    fill: bool = True                    # fewer than num_clips pass min_score -> fill up with the next best (flagged weak)
    story_scenes: bool = False           # newly released story game: cutscenes / plot moments are valuable clips
    block_adult: bool = True             # never keep clips with nudity / strip clubs / sexual content (AI + transcript)
    weight: float = 0.75                 # final = weight*ai_score + (1-weight)*audio_score (both 0-10)
    titles: bool = True                  # use the AI hook as the title when title.text is empty, or via {hook}
    context: str = ""                    # e.g. "Kai Cenat GTA 6 stream" - helps the AI judge
    max_cost_usd: float = 1.0            # per-run budget; AI stops (audio ranking continues) once reached
    price_input_per_m: float = 0.10      # USD per 1M input tokens
    price_output_per_m: float = 0.50     # USD per 1M output tokens
    concurrency: int = 4                 # parallel API calls
    timeout: float = 60.0                # seconds per API call


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    highlights: HighlightsConfig = field(default_factory=HighlightsConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    captions: CaptionsConfig = field(default_factory=CaptionsConfig)
    title: TitleConfig = field(default_factory=TitleConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    keep_temp: bool = False
    base_dir: Path = field(default=Path("."), repr=False, compare=False)  # set to config file's folder

    def resolve(self, p: str | Path | None) -> Path | None:
        """Resolve a config-relative path."""
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else (self.base_dir / p).resolve()


# ---------------------------------------------------------------------------
# loading / validation


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dc.is_dataclass(tp)


def _coerce(value: Any, tp: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], where)
    if _is_dataclass_type(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return _build(tp, value, where)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        (inner,) = typing.get_args(tp) or (Any,)
        return [_coerce(v, inner, f"{where}[{i}]") for i, v in enumerate(value)]
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ConfigError(f"{where}: expected an integer")
        return int(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number")
        return float(value)
    if tp is str:
        if not isinstance(value, (str, int, float)):
            raise ConfigError(f"{where}: expected text")
        return str(value)
    return value


def _build(cls: type, data: dict, where: str = "config") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dc.fields(cls) if f.name != "base_dir"}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def validate(cfg: Config) -> Config:
    h, lay, r = cfg.highlights, cfg.layout, cfg.render
    errs: list[str] = []
    if h.num_clips < 1:
        errs.append("highlights.num_clips must be >= 1")
    if not (0 < h.min_duration <= h.max_duration):
        errs.append("highlights: need 0 < min_duration <= max_duration")
    if h.hop <= 0:
        errs.append("highlights.hop must be > 0")
    if lay.mode not in ("split", "gameplay_only"):
        errs.append("layout.mode must be 'split' or 'gameplay_only'")
    if lay.facecam.preset not in FACECAM_PRESETS:
        errs.append(f"layout.facecam.preset must be one of {FACECAM_PRESETS}")
    if lay.facecam.preset == "custom" and (not lay.facecam.region or len(lay.facecam.region) != 4):
        errs.append("layout.facecam.region must be [x, y, w, h] when preset is 'custom'")
    if len(lay.facecam.size) != 2 or len(lay.facecam.margin) != 2:
        errs.append("layout.facecam.size and margin must be [w, h] / [x, y]")
    if lay.width % 2 or lay.height % 2:
        errs.append("layout.width/height must be even")
    used = lay.header_height + (lay.facecam_height if lay.mode == "split" else 0)
    if used >= lay.height - 200:
        errs.append("layout: header_height + facecam_height leaves no room for gameplay")
    if lay.gameplay.fit not in ("crop", "blur"):
        errs.append("layout.gameplay.fit must be 'crop' or 'blur'")
    if lay.gameplay.zoom < 1.0:
        errs.append("layout.gameplay.zoom must be >= 1.0")
    if cfg.captions.position not in ("seam", "gameplay", "bottom"):
        errs.append("captions.position must be seam | gameplay | bottom")
    if r.encoder not in ENCODERS:
        errs.append(f"render.encoder must be one of {ENCODERS}")
    if r.workers < 1:
        errs.append("render.workers must be >= 1")
    if cfg.transcription.cpu_threads < 0:
        errs.append("transcription.cpu_threads must be >= 0")
    w = cfg.watermark
    if not (0 <= w.opacity <= 1):
        errs.append("watermark.opacity must be between 0 and 1")
    if w.font_size <= 0:
        errs.append("watermark.font_size must be > 0")
    if w.position not in WATERMARK_POSITIONS:
        errs.append(f"watermark.position must be one of {', '.join(WATERMARK_POSITIONS)}")
    a = cfg.ai
    a.enabled = a.enabled.strip().lower()
    if a.enabled not in AI_ENABLED:
        errs.append("ai.enabled must be auto, true or false")
    if a.reasoning_effort not in ("none", "low", "medium", "high"):
        errs.append("ai.reasoning_effort must be none | low | medium | high")
    if a.image_detail not in ("low", "high", "auto"):
        errs.append("ai.image_detail must be low | high | auto")
    if a.candidate_multiplier < 1:
        errs.append("ai.candidate_multiplier must be >= 1")
    if a.max_candidates < 1:
        errs.append("ai.max_candidates must be >= 1")
    if not (1 <= a.frames <= 9):
        errs.append("ai.frames must be between 1 and 9")
    if not (0 <= a.min_score <= 10):
        errs.append("ai.min_score must be between 0 and 10")
    if not (0 <= a.weight <= 1):
        errs.append("ai.weight must be between 0 and 1")
    if a.max_cost_usd < 0 or a.price_input_per_m < 0 or a.price_output_per_m < 0:
        errs.append("ai.max_cost_usd and ai.price_* must be >= 0")
    if a.concurrency < 1:
        errs.append("ai.concurrency must be >= 1")
    if a.timeout <= 0:
        errs.append("ai.timeout must be > 0")
    if not a.model.strip() or not a.api_key_env.strip():
        errs.append("ai.model and ai.api_key_env must not be empty")
    if errs:
        raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errs))
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    """Child over parent: dicts merge recursively; lists and scalars replace."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_PATH_KEYS = (("paths", "input_dir"), ("paths", "output_dir"), ("paths", "temp_dir"),
              ("paths", "cache_dir"), ("publish", "root"), ("captions", "fonts_dir"), ("highlights", "chat", "file"))


def _absolutize_paths(data: dict, folder: Path) -> None:
    """Make relative path values absolute against the folder of the file that set them,
    so a preset in configs/ that extends ../config.yaml still writes to ./output."""
    for keys in _PATH_KEYS:
        node = data
        for k in keys[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict) and isinstance(node.get(keys[-1]), str):
            p = Path(node[keys[-1]])
            if not p.is_absolute():
                node[keys[-1]] = str((folder / p).resolve())


def _root_dir(path: Path) -> Path:
    """Folder of the top-most ancestor in an `extends` chain (where unset default paths live)."""
    seen: set[Path] = set()
    cur = path.resolve()
    while cur not in seen:
        seen.add(cur)
        try:
            ref = (yaml.safe_load(cur.read_text(encoding="utf-8")) or {}).get("extends")
        except (OSError, yaml.YAMLError, AttributeError):
            break
        if not isinstance(ref, str):
            break
        nxt = Path(ref) if Path(ref).is_absolute() else cur.parent / ref
        if not nxt.exists():
            break
        cur = nxt.resolve()
    return cur.parent


def _load_yaml_tree(path: Path, seen: tuple[Path, ...] = (), absolutize_root: bool = False) -> dict:
    """Load a YAML file, resolving a top-level `extends:` chain. The result has no `extends` key.
    With absolutize_root the chain's root file also gets its relative paths made absolute
    (used for layered configs after the first, which have no base_dir of their own)."""
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(p) for p in (*seen, resolved))
        raise ConfigError(f"Config 'extends' cycle: {chain}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        if absolutize_root:
            _absolutize_paths(data, resolved.parent)
        return data  # root of the chain: its relative paths resolve against base_dir (= its folder)
    _absolutize_paths(data, resolved.parent)
    if not isinstance(parent_ref, str) or not parent_ref.strip():
        raise ConfigError(f"{path}: 'extends' must be a file path")
    parent = Path(parent_ref)
    if not parent.is_absolute():
        parent = resolved.parent / parent
    if not parent.exists():
        raise ConfigError(f"{path}: extends file not found: {parent}")
    return _deep_merge(_load_yaml_tree(parent, (*seen, resolved), absolutize_root), data)


def load_config(path: str | Path | list[str | Path] | None = None) -> Config:
    """Load config.yaml (or defaults if path is None / missing default file).

    A file may start with `extends: <path>` (relative to that file, or absolute) to inherit
    from another YAML file; the child's keys are deep-merged over the parent's.
    A list of files is loaded in order (each with its own extends chain) and deep-merged,
    later files winning; base_dir comes from the first file's chain.
    """
    if isinstance(path, (list, tuple)):
        paths = [Path(p) for p in path]
        if not paths:
            path = None
        elif len(paths) == 1:
            path = paths[0]
        else:
            for p in paths:
                if not p.exists():
                    raise ConfigError(f"Config file not found: {p}")
            data = _load_yaml_tree(paths[0])
            for p in paths[1:]:
                data = _deep_merge(data, _load_yaml_tree(p, absolutize_root=True))
            cfg = _build(Config, data)
            cfg.base_dir = _root_dir(paths[0])
            return validate(cfg)
    if path is None:
        default = Path("config.yaml")
        if not default.exists():
            cfg = Config(base_dir=Path.cwd())
            return validate(cfg)
        path = default
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    data = _load_yaml_tree(path)
    cfg = _build(Config, data)
    cfg.base_dir = _root_dir(path)
    return validate(cfg)
