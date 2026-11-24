# Architecture

```
VOD (.mp4/.mkv) ──► probe ──► audio scan (streamed, cached) ──► score curve ◄── chat log (optional)
                                                                   │
                                                          peak picking + NMS
                                                                   │
                         ┌───────────── per highlight ─────────────┘
                         ▼
          extract padded audio ─► faster-whisper (word timestamps)
                         │
          refine boundaries (silence / sentence edges)
                         │
          ASS captions + title ─► ffmpeg single-pass filter graph ─► output/<stem>/<stem>_01_….mp4
                                   (crop facecam + crop gameplay + vstack + pad + subtitles + loudnorm,
                                    HW encoder auto-detected; renders run in a thread pool while the
                                    next clip is being transcribed)
```

Optional AI stage (`ai.enabled`, needs the `openai` extra and an API key): `find_highlights` is asked for
`ceil(num_clips * ai.candidate_multiplier)` candidates (capped by `ai.max_candidates`); every candidate is
transcribed, gets a contact sheet (one ffmpeg call, 2x2 for 4 frames) and is judged by an OpenAI vision model
in a thread pool that overlaps with the next transcription. `select_final` drops clips below `ai.min_score`,
ranks by `weight*ai + (1-weight)*audio` and keeps `num_clips`; only those are rendered, with the AI hook as
title. Extra output: `candidates.json`, and `ai_score/category/hook/reason/final_score` in `clips.json`.
Without the package/key, or with `--no-ai`/`--dry-run`, the flow above runs unchanged.

Stack: Python ≥3.10, FFmpeg/ffprobe binaries, NumPy, faster-whisper (CTranslate2; CUDA float16 or CPU int8), PyYAML, rich.

## Package layout

```
src/clipper/
  config.py          dataclass config + YAML loader/validator           (done)
  models.py          Highlight, Word, Segment, Transcript, Rect, Layout, parse_time, fmt_time (done)
  media.py           ffprobe/ffmpeg wrappers, extract_audio, encoder detection (done)
  analysis/
    audio.py         streaming audio feature extraction + cache
    chat.py          chat log discovery/parsing -> activity curve
    highlights.py    scoring, peak picking, clip windows, transcript-based refinement
  transcribe.py      faster-whisper wrapper
  layout.py          source/output geometry
  captions.py        ASS subtitle generation (word-by-word captions + title)
  render.py          ffmpeg filter graph, clip render, preview still
  ai/
    ranker.py        AIRanker (OpenAI Responses API, budget, thread-safe), build_contact_sheet,
                     final_scores / select_final / candidate_pool, .env key lookup
  pipeline.py        per-video orchestration + batch (AI candidate pool, judge, select, render)
  cli.py             argparse entry point (`clipper ...`)
tests/               pytest, pure-logic tests (no real video needed)
```

## Module contracts (exact signatures — implement exactly these)

All times are **seconds in source (VOD) time** unless a name says otherwise.
Use `from __future__ import annotations`, type hints, and short docstrings.
`cfg` objects are the dataclasses from `clipper.config`.

### clipper/analysis/audio.py
```python
@dataclass
class AudioFeatures:
    hop: float                 # seconds per frame (cfg.highlights.hop)
    duration: float            # seconds actually analysed
    loudness_db: np.ndarray    # float32, per hop, RMS level in dBFS (floor -90)
    vocal_db: np.ndarray       # float32, per hop, 300-3400 Hz band energy in dB (floor -90)
    def times(self) -> np.ndarray            # centre time of each frame
    def index(self, t: float) -> int         # clamp to valid range

def compute_features(video: Path, hop: float = 0.5, audio_track: int = 0,
                     cache_dir: Path | None = None,
                     progress: Callable[[float], None] | None = None) -> AudioFeatures
```
- Decode with `ffmpeg -i video -map 0:a:{track} -vn -ac 1 -ar 16000 -f s16le -` via `subprocess.Popen`
  and read stdout in chunks (~30 s of audio); never load a 6 h file into memory at once.
- Short-time frames of 50 ms (800 samples) -> rfft -> total power and 300-3400 Hz band power;
  aggregate per `hop` (mean power) -> dB. Vectorise with numpy per chunk.
- `progress(fraction)` called occasionally (needs duration: use `media.probe`).
- Cache as `.npz` in `cache_dir`, key = sha1(abs path, size, mtime_ns, hop, track). Load if present.
- If the file has no audio stream -> raise `ValueError` with a clear message.

### clipper/analysis/chat.py
```python
def find_chat_file(video: Path) -> Path | None
    # looks next to the video for: <stem>.chat.json, <stem>_chat.json, <stem>.json, <stem>.chat.csv,
    # <stem>_chat.csv, <stem>.csv (first match wins)
def load_chat_timestamps(path: Path) -> np.ndarray   # sorted float64 seconds from VOD start
def chat_activity(timestamps: np.ndarray, duration: float, hop: float,
                  window: float, delay: float) -> np.ndarray  # length = ceil(duration/hop), float32
```
- JSON formats to support: TwitchDownloaderCLI (`{"comments":[{"content_offset_seconds":..}]}`),
  chat-downloader / yt-dlp live_chat style (list of objects with `time_in_seconds` or
  `time_text` or `timestamp` in µs relative/absolute — if absolute epoch, subtract the first one),
  generic list of objects with any of `offset`, `time`, `timestamp`, `seconds`, `t`; values may be numbers
  or "HH:MM:SS" strings (`models.parse_time`). Also JSON-lines (one object per line).
- CSV: header row; pick the first column named like time/timestamp/offset/seconds/time_in_seconds;
  values numeric or HH:MM:SS.
- Activity = message counts per hop, smoothed with a moving-average of `window` seconds, then shifted
  EARLIER by `delay` seconds (chat reacts late). Unparseable file -> raise ValueError.

### clipper/analysis/highlights.py
```python
def score_curve(features: AudioFeatures, cfg: HighlightsConfig,
                chat: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]
    # returns (combined score per hop, {"loudness":..,"vocal":..,"excitement":..,"chat":..} component curves)
def select_highlights(features: AudioFeatures, score: np.ndarray, components: dict[str, np.ndarray],
                      cfg: HighlightsConfig, duration: float) -> list[Highlight]
def find_highlights(video: Path, duration: float, cfg: HighlightsConfig, cache_dir: Path | None,
                    chat_path: Path | None = None,
                    progress: Callable[[float], None] | None = None
                    ) -> tuple[list[Highlight], AudioFeatures]
def refine_with_transcript(h: Highlight, transcript: Transcript, cfg: HighlightsConfig,
                           duration: float) -> Highlight
```
Scoring:
- Each feature is normalised **relative to its local baseline** so a loud game section doesn't win by
  default: robust z-score vs. a rolling median / MAD over ~5 minutes (compute medians on ~15 s blocks
  and interpolate — must be fast for 6 h = 43 200 hops). Clip z to [-3, 6].
- `excitement` = positive part of the derivative of ~2 s-smoothed loudness (sudden jumps), then z-scored.
- combined = Σ weight·z (weights from `cfg.weights`) + `cfg.chat.weight`·chat_z (only if chat given),
  smoothed with a ~3 s moving average. Silence (loudness < -60 dBFS) never scores high.
Selection:
- Ignore `skip_start` / `skip_end`. Greedy: take the global max, suppress ±max(`min_gap`, clip length)
  around it, repeat until `num_clips` or nothing left above a sane floor (score > 0).
- Window = [peak − pre_roll, peak + post_roll]; extend the end while the smoothed score stays above 60 %
  of the peak (reaction continues) but never beyond max_duration; enforce min_duration by growing the
  side with more headroom; clamp to [0, duration].
- Snap start/end to the quietest hop within ±`snap_window` (so cuts land on pauses).
- No two windows may overlap. Return sorted by score desc with rank 1..n; `reasons` = component values
  at the peak.
Refinement (`refine_with_transcript`):
- If `cfg.snap_to_sentences`: move start to the start of the segment that contains/is nearest to the
  start (only if within `snap_window`), and end to the end of the segment containing/nearest the end
  (within `snap_window`), plus 0.15 s / 0.35 s breathing room. Keep min/max duration, clamp to [0, duration].
  Empty transcript -> return h unchanged.

### clipper/transcribe.py
```python
class Transcriber:
    def __init__(self, cfg: TranscriptionConfig) -> None   # lazy: model loads on first transcribe()
    @property
    def device(self) -> str                                # resolved "cuda" / "cpu"
    def transcribe(self, wav: Path, offset: float = 0.0) -> Transcript
        # word_timestamps=True, vad_filter=cfg.vad_filter; add `offset` to every time so results are in
        # SOURCE time. Strip whitespace from words; drop empty words.
def resolve_device(cfg: TranscriptionConfig) -> tuple[str, str]   # (device, compute_type)
```
- device "auto": cuda if `ctranslate2.get_cuda_device_count() > 0` else cpu. compute_type "auto":
  float16 on cuda, int8 on cpu. If CUDA load fails, fall back to cpu/int8 with a warning (logging).
- After a language is detected on the first clip and `cfg.language` is None, reuse it for later clips
  (speeds up and avoids per-clip mis-detection).

### clipper/layout.py
```python
def facecam_region(src_w: int, src_h: int, cfg: FacecamConfig) -> Rect
def compute_layout(src_w: int, src_h: int, cfg: LayoutConfig, captions: CaptionsConfig | None = None) -> Layout
```
- Presets: rect of size `size` (fractions) anchored at the named corner, inset by `margin`.
  custom `region`: values ≤ 1 are fractions of src, > 1 pixels. Clamp inside the frame.
- Output bands: header = `header_height`; split mode: facecam band = `facecam_height`, gameplay = rest.
  gameplay_only: facecam_h = 0, facecam_crop = None.
- Facecam crop: shrink the region (centred) to the facecam band's aspect ratio (crop-to-fill, no stretch).
- Gameplay crop (`fit="crop"`): height = src_h/zoom, width = height·(out_w/gameplay_h); if wider than
  src, clamp to src_w and reduce height to keep the aspect. Centre at (center_x, center_y), clamp inside.
  (`fit="blur"`): crop = the zoomed full frame keeping src aspect (renderer scales it to width and puts it
  over a blurred fill of the same crop).
- All crop values even integers (yuv420p). caption_y: seam → y of the seam between facecam and gameplay
  (header_h + facecam_h; in gameplay_only → header_h + gameplay_h*0.2); gameplay → gameplay_y + 0.30·gameplay_h;
  bottom → out_h − 0.18·out_h. Add `captions.y_offset`.

### clipper/captions.py
```python
def build_ass(transcript: Transcript | None, clip_start: float, clip_end: float, layout: Layout,
              captions: CaptionsConfig, title: TitleConfig | None = None, title_text: str = "") -> str
def chunk_words(words: list[Word], max_words: int, max_chars: int) -> list[list[Word]]
```
- PlayResX/PlayResY = layout.out_w/out_h, `ScaledBorderAndShadow: yes`, WrapStyle 2.
- Styles "Caption" and "Title" from config (colors "#RRGGBB" → ASS `&H00BBGGRR`). Bold -1.
- Word times converted to clip-relative (`t - clip_start`), words outside the clip dropped/clamped.
- Chunking: break when max_words or max_chars reached, or a gap > 0.6 s between words, or the word ends
  with . ? ! (sentence end). Each chunk is shown from its first word start until the next chunk's start
  (max +0.6 s after its last word). For each word in a chunk emit one Dialogue for that word's
  active interval in which the full chunk is drawn and the active word gets `highlight_color` and (if `pop`)
  `\t` scale animation 100→118→108 %. Position with `{\an5\pos(out_w/2, caption_y)}`. Uppercase if set.
  Escape `\`, `{`, `}` in word text.
- Title (if `title_text`): one Dialogue for the whole clip, `\an5\pos(out_w/2, header_h/2)`; if header_h
  is 0 place it at y = 0.06·out_h. Shrink font (`\fs`) for long text so it fits ~90 % width (estimate
  0.6·font_size px per char), allow at most 2 lines (insert `\N` at the most central space).
- captions.enabled False → only title (or an ASS with no events).

### clipper/render.py
```python
def build_filtergraph(layout: Layout, fps: float | None, ass_name: str | None,
                      fonts_dir: str | None) -> str                  # final video label is [vout]
def render_clip(video: Path, start: float, duration: float, layout: Layout, out_path: Path,
                cfg: RenderConfig, encoder: str, workdir: Path, ass_path: Path | None = None,
                fonts_dir: Path | None = None) -> Path
def render_preview(video: Path, at: float, layout: Layout, out_png: Path, workdir: Path,
                   ass_path: Path | None = None, fonts_dir: Path | None = None) -> Path
```
- Split: `[0:v]split=2[fc][gp]; [fc]crop=..,scale=W:FH:flags=lanczos,setsar=1[fcs];
  [gp]crop=..,scale=W:GH:flags=lanczos,setsar=1[gps]; [fcs][gps]vstack=2[st];
  [st]pad=W:H:0:header_h:color=<bg>[base]; [base]subtitles=<ass>[...]; fps; format=yuv420p[vout]`.
  gameplay_only: no facecam branch. fit=blur: gameplay = blurred crop scaled to fill + sharp crop scaled to
  width overlaid centred.
- **Windows path safety**: run ffmpeg with `cwd=workdir` and refer to the .ass by bare filename;
  fonts_dir passed via `fontsdir=` with ffmpeg filter escaping (`\` → `/`, `:` → `\:`, `'` quoted).
- Input: `-ss {start} [-hwaccel X] -i video -t {duration}`, `-map [vout] -map 0:a:{track}?`,
  audio `aac` at `audio_bitrate`, 48 kHz, `loudnorm=I=-14:TP=-1.5:LRA=11` if `normalize_audio`,
  `media.encoder_args(encoder, quality)`, `-pix_fmt yuv420p` for non-HW encoders, `-movflags +faststart`.
  fps: cfg.fps or min(source fps, 60) — renderer receives fps already resolved as `fps` param of graph.
- Write to `out_path.with_suffix('.part.mp4')` then rename (no half files on crash).
- If the render fails and `cfg.hwaccel != "none"`, retry once without hwaccel; if a HW encoder fails,
  retry once with libx264. Raise `media.FFmpegError` otherwise.
- Preview: same graph, single frame (`-frames:v 1`) to PNG, plus draw a 4 px red box (`drawbox`) on the
  source facecam region *before* cropping? No — preview shows the final composition only; the CLI also
  writes a second PNG of the raw source frame with the facecam rect drawn (`render_source_preview`).
```python
def render_source_preview(video: Path, at: float, layout: Layout, out_png: Path) -> Path
```

### clipper/pipeline.py
```python
@dataclass
class ClipResult:
    video: Path; index: int; highlight: Highlight; output: Path | None; error: str | None; transcript_text: str
def process_video(video: Path, cfg: Config, transcriber: Transcriber | None, encoder: str,
                  console: rich.console.Console, dry_run: bool = False) -> list[ClipResult]
def run_batch(inputs: list[Path], cfg: Config, console: Console, dry_run: bool = False) -> list[ClipResult]
```
- Per video: probe → chat file (cfg.highlights.chat.file resolved, else auto-detect if enabled) →
  find_highlights (rich progress bar) → print table → for each highlight in **chronological** order:
  extract audio for [start − snap_window − 1, end + snap_window + 1] → transcribe (offset = that start) →
  refine_with_transcript → build_ass → submit render_clip to a ThreadPoolExecutor(cfg.render.workers)
  → collect. Transcription of clip i+1 overlaps with rendering of clip i.
- Output: `<output_dir>/<stem>/<stem>_<rank:02d>_<H-MM-SS>.mp4` and `<output_dir>/<stem>/clips.json`
  (rank, start, end, peak, score, reasons, file, transcript text).
- Temp: `tempfile.TemporaryDirectory(dir=cfg.paths.temp_dir)`; if `cfg.keep_temp`, use mkdtemp and keep.
- One failing clip/video must not stop the batch; record the error and continue.
- Captions disabled AND snap_to_sentences False → skip transcription entirely (transcriber None).

### clipper/cli.py
`clipper <command>` with argparse, `main(argv=None) -> int`, also `python -m clipper`.
- `run [INPUTS...] [-c CONFIG] [-n/--clips N] [--min D] [--max D] [--facecam PRESET] [--layout split|gameplay_only]
   [--title TEXT] [--no-captions] [--model NAME] [--encoder E] [--chat FILE] [--keep-temp] [--dry-run]`
  INPUTS default: cfg.paths.input_dir. CLI flags override config values.
- `analyze VIDEO [-c] [-n N] [--chat FILE]` → highlights table only (no render, no transcription).
- `preview VIDEO [-c] [--at TIME] [-o DIR] [--facecam PRESET]` → writes `<stem>_preview.png` (final 9:16
  composition with sample caption + title) and `<stem>_source.png` (source frame with facecam box).
- `doctor` → ffmpeg/ffprobe found + version, working encoders, CUDA device count, whisper importable.
- `init-config [PATH]` → copy the bundled default config.yaml (refuse to overwrite unless `--force`).
Exit code non-zero if any clip failed.
