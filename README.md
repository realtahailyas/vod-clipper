# vod-clipper — AI VOD to YouTube Shorts / TikTok clipper

> Twitch VOD clipper that finds highlights automatically, adds word-by-word Whisper captions and renders 9:16 vertical shorts with ffmpeg. Local and offline-first.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

Turns long 16:9 gaming VODs (`.mp4` / `.mkv`) into ready-to-publish 1080x1920 vertical highlight clips for YouTube Shorts, TikTok and Reels, fully local. Drop VODs in `input/`, run one command, get clips in `output/`.

## Quick start

```
git clone https://github.com/realtahailyas/vod-clipper.git
cd vod-clipper
pip install -e .          # needs ffmpeg + ffprobe on PATH
clipper run input/my_vod.mp4
```

Clips (and a `clips.json`) land in `output/<vod name>/`. Use `clipper preview VOD` to check the facecam crop first, and the presets in `configs/` (`-c configs/creators/xqc.yaml`) for common streamer layouts.

## Features

- Automatic highlight detection from audio energy (loudness, speech-band energy, sudden jumps), optionally boosted by a chat log.
- Split layout: title header band, facecam, gameplay (or gameplay only).
- Word-by-word burned-in captions from local Whisper (faster-whisper), with the active word highlighted.
- Cuts land on pauses / sentence edges instead of mid-word.
- Hardware encoding auto-detected (NVENC, QSV, AMF), falls back to libx264.
- Batch mode, analysis cache, parallel renders, loudness normalisation.

## How it works

```
VOD -> probe -> audio scan (streamed, cached) -> score curve <- chat log (optional)
                                                     |
                                         peak picking + min-gap suppression
                                                     |
            per highlight: extract audio -> Whisper (word timestamps) -> refine cut points
                                                     |
                    ASS captions + title -> one ffmpeg pass (crop, stack, subtitles, loudnorm)
                                                     |
                                   output/<stem>/<stem>_01_0-12-34.mp4 + clips.json
```

Transcription of the next clip overlaps with rendering of the current one.

## Requirements

- Python 3.10+
- FFmpeg and ffprobe on `PATH` (or set `CLIPPER_FFMPEG` / `CLIPPER_FFPROBE`)
- Optional: NVIDIA GPU with CUDA 12 + cuDNN 9 for fast Whisper
- Optional: NVENC / QSV / AMF capable GPU for fast encoding

## Install

Windows (PowerShell):

```powershell
winget install Gyan.FFmpeg
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Linux (bash):

```bash
sudo apt install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Check the setup with `clipper doctor` (shows ffmpeg, which encoders really work, CUDA devices, and which encoder `auto` picks).

## Quick start

```
clipper init-config            # writes config.yaml you can edit (optional)
# put your VODs in input/
clipper preview input/stream.mp4 --facecam bottom-left   # check the layout
clipper run                    # process everything in input/
clipper run input/stream.mp4 -n 5 --title "CLUTCH #{n}"
```

`preview` writes `<stem>_preview.png` (final 9:16 composition with sample caption and title) and `<stem>_source.png` (source frame with the facecam box drawn) into `output/_preview/` (or `-o DIR`). Adjust until the facecam box is right, then run.

Other commands:

- `clipper analyze VIDEO [-n N] [--chat FILE]`: print the highlight table only.
- `clipper run --dry-run`: analysis + `clips.json` with planned windows, no transcription or rendering.
- Flags on `run`: `-c CONFIG -n --min --max --facecam --layout --title --no-captions --watermark --no-watermark --model --encoder --chat --ai --no-ai --keep-temp -v`.

CLI flags override `config.yaml`. Exit code is non-zero if any clip failed (130 on Ctrl+C, 2 on config errors).

## Configuration highlights

`config.yaml` is optional; every key has a default and unknown keys are rejected (typo protection). `clipper init-config` writes the full commented file.

- **Facecam**: `layout.facecam.preset` is `top-left | top-right | bottom-left | bottom-right | custom`. Presets use `size: [w, h]` and `margin: [x, y]` as fractions of the source. For `custom`, set `region: [x, y, w, h]` (values <= 1 are fractions, > 1 are pixels).
  - Streamers often move or hide the cam mid-stream. List the other spots in `facecam.alternates: [[x, y, w, h], ...]`; with AI ranking on, every clip is cropped at the spot where the AI sees the cam, and clips with no visible cam render gameplay-only (see `configs/creators/xqc_gta6.yaml`).
- **Layout**: `layout.mode: split | gameplay_only`; `header_height` (title band) and `facecam_height` in output pixels, gameplay gets the rest. `gameplay.zoom`, `center_x`, `center_y`, and `fit: crop | blur`.
- **Captions**: `font`, `fonts_dir`, `font_size`, `primary_color`, `highlight_color`, `outline`, `shadow`, `uppercase`, `max_words`, `max_chars`, `position: seam | gameplay | bottom`, `y_offset`, `pop`.
- **Clip defaults** (tuned for high-action open-world streams such as GTA): `highlights.num_clips: 25`, `min_gap: 60` (back-to-back action sequences), `min_duration: 35`, `max_duration: 60`, `pre_roll: 30`, `post_roll: 15`, so clips land around 45-55 s. Use `-n 30` / `-n 40` for more, or lower `min_gap` / raise it for calmer streams.
- **Title**: `title.text` with `{n}` (clip rank), `{stem}` (video name) and `{hook}` (AI hook, see below), e.g. `"{stem} - clip {n}"`. Empty means no title, unless AI ranking is on and `ai.titles` is true (then the AI hook is the title).
- **Watermark**: `watermark.text` (default `@YourChannel`) is burned into every 9:16 render and `clipper preview` as an extra ASS style above the captions. Keys: `enabled`, `text`, `font`, `font_size`, `color`, `opacity` (0..1, 0.35-0.40 is subtle), `outline_color`, `outline`, `shadow`, `position`, `margin`, and `x`/`y` for `custom` (anchor = bottom-right of the text). Positions: `facecam_bottom_right` (default, moves up so it never overlaps the captions at the seam; falls back to `bottom_right` in gameplay_only), `facecam_top_right`, `header_right` (falls back to `facecam_top_right` when there is no header), `bottom_right`, `custom`. Safe zones: the `facecam_*` positions sit clear of the Shorts/TikTok bottom UI and right-side action icons; `bottom_right` is placed 20% above the bottom edge but is NOT guaranteed to stay clear of the icons. CLI: `--watermark TEXT` overrides the text, `--no-watermark` turns it off (both on `run` and `preview`).
- **Whisper model** (`transcription.model`, default `small.en` + `int8`, `beam_size: 1`, `cpu_threads: 6`):
  - CPU only: `small.en` (`base.en` is faster but invents lines over loud game audio).
  - NVIDIA GPU: `large-v3` or `distil-large-v3` with `compute_type: float16`.
  - `*.en` models need `language: en`; use a multilingual model with `language: null` for other languages.
  - `vad_filter` defaults to `false`: on gaming streams Silero VAD drops screams and reactions that sit under game audio.
  - `cpu_threads` caps whisper's CPU threads (0 = auto) so batch runs don't lock up or throttle the laptop.

## AI ranking (optional)

Audio peaks are a good first filter but cannot tell a streamer screaming from a cutscene or loud music. The optional AI stage fixes that: it detects a bigger pool of candidates (2.5x `num_clips`, so 25 clips -> 63 candidates), transcribes each one, builds a 2x2 contact sheet of four frames, and asks an OpenAI vision model to score it for short-form virality, label it and write a hook title. The best clips (`final = 0.75 * ai_score + 0.25 * audio_score`, both 0-10) are kept, re-ranked 1..N and rendered; the rest are dropped. AI calls run in parallel with transcription of the next candidate.

Setup:

1. `pip install -e .[ai]` (installs the `openai` package).
2. Provide an API key, either in a `.env` file in the project folder (next to `config.yaml`, already gitignored, never commit it):
   ```
   OPENAI_API_KEY=sk-...
   ```
   or as an environment variable (Windows: `setx OPENAI_API_KEY "sk-..."`, then open a new terminal).
3. `clipper doctor` shows whether the package and key were found (the key itself is never printed).

`ai.enabled: auto` (default) turns the stage on when the key is found. Force it with `clipper run --ai` or switch it off with `--no-ai`. If the package or key is missing the tool prints one line saying why and runs the normal audio-only pipeline. `--dry-run` skips AI (it needs transcripts).

Model: `ai.model` defaults to `gpt-6-luna`, the cheapest current OpenAI model with image input ($0.10 in / $0.50 out per 1M tokens). Any OpenAI vision model that supports the Responses API and structured output works; change `ai.model` and the `ai.price_*` values (used for the cost counter). Put a short description of the stream in `ai.context` (for example `Kai Cenat GTA 6 stream`) to help the judge.

Cost: roughly a few cents per 5-hour stream at the default 63 candidates with `image_detail: low` (an estimate, not a guarantee; it depends on transcript length and the model's reasoning tokens). `ai.max_cost_usd` (default 1.0) is a hard per-run budget: once reached, remaining candidates are ranked by audio score only. The run prints `AI: <calls> calls, <failures> failed, $<spent>` and a candidates table.

Tuning: `ai.min_score` drops clips the AI rates below it, `ai.weight` balances AI vs audio, `ai.candidate_multiplier` / `ai.max_candidates` control the pool, `ai.frames` and `ai.image_detail` trade image quality for cost. `output/<stem>/candidates.json` lists every candidate with its verdict, so you can see why a clip was kept or dropped.

`ai.fill` (default true) always delivers `num_clips`: if fewer clips reach `ai.min_score`, the next best by final score fill the gap and are flagged `"weak": true` in `clips.json` / `candidates.json` (candidates table: `✓` kept, `fill` filled, `✗` dropped). `--strict` (or `ai.fill: false`) drops them instead; `--fill` forces it on.

`ai.story_scenes` (default false) tells the judge that this is a newly released story game, so major cutscenes, plot reveals and set pieces are valuable (new `story` category) instead of "game audio / boring".

## Presets / extends

Ready-made layouts in `configs/` (they `extends: ../config.yaml`, so your clip count, model and encoder settings still apply):

| Preset | Facecam | Notes |
|---|---|---|
| `configs/preset_top_left.yaml` | top-left, 25% x 24.5% | Tuned on a real 1920x1080 Kai Cenat VOD; gameplay zoom 1.08 trims the top-centre viewer counter |
| `configs/preset_bottom_left.yaml` | bottom-left, 25% x 25% | Generic; check with `clipper preview` |

```
clipper preview input/stream.mp4 -c configs/preset_top_left.yaml --at 1:00:00
clipper run -c configs/preset_top_left.yaml
```

Layer files by repeating `-c` (later wins, deep merge; each file keeps its own `extends` chain and its own relative paths): a creator preset plus a game profile.

```
clipper run -c configs/creators/<creator>.yaml -c configs/games/gta6.yaml
```

`configs/games/gta6.yaml` sets 25 clips, `ai.story_scenes: true`, `ai.fill: true`, `min_score: 3.5` and a GTA 6 `ai.context`. Game profiles have no `extends`, so they only add on top of the preset.

Any config file can start with `extends: base.yaml` (relative to that file, or absolute). Nested mappings merge, lists and scalars replace; chains are allowed. Relative paths resolve against the file that sets them; unset defaults (`input/`, `output/`, `.cache/`) resolve next to the root config.

## Chat logs (optional)

Chat activity boosts the score. Supported:

- JSON: TwitchDownloaderCLI (`{"comments": [{"content_offset_seconds": ...}]}`), chat-downloader / yt-dlp live chat, generic lists of objects with `offset`, `time`, `timestamp`, `seconds` or `t` (numbers or `HH:MM:SS`), and JSON lines.
- CSV with a header and a time / timestamp / offset / seconds column.

Auto-detection looks next to the video for, in order: `<stem>.chat.json`, `<stem>_chat.json`, `<stem>.json`, `<stem>.chat.csv`, `<stem>_chat.csv`, `<stem>.csv`. Or pass `--chat FILE` / set `highlights.chat.file`. `chat.delay` shifts activity earlier because chat reacts late.

## Publishing / Drive export

After `clipper run`, every rendered short is packaged for YouTube Shorts (needs `publish.creator`, set by the
creator presets; skip with `--no-publish`). Re-package an existing output folder any time:

    clipper export output\xqc_gta6_launch -c configs\creators\xqc_gta6.yaml -c configs\games\gta6.yaml

Result: `<publish.root>/<creator>/<NNN - hook>/` (e.g. `publish/xQc/001 - ...`, optionally mirrored to a Drive folder via `publish.drive_folder`) with the mp4, `title.txt`,
`description.txt`, `tags.txt`, `captions.srt` and `metadata.json`, plus `manifest.json|csv` and
`HANDOVER.md` directly in the root (a guide for the agent that posts them). Runs append with a running number; the same clip is never
duplicated. Weak clips are skipped unless `--include-weak`. Title/description/tags are written by the AI model
(falls back to a template; counted against `ai.max_cost_usd`). Point `publish.root` at a Google Drive for Desktop
folder to sync automatically.

## Output

```
output/<stem>/
  <stem>_01_0-12-34.mp4     rank 01, starts at 0:12:34 in the VOD
  <stem>_02_1-05-10.mp4
  clips.json                rank, start, end, peak, score, reasons, file, transcript, error
                            (+ ai_score, category, hook, reason, final_score when AI ranking ran)
  candidates.json           AI runs only: every judged candidate with its verdict, for tuning
```

## Performance tips

Measured on a 12th-gen i5-1245U laptop (15 W, Intel Xe iGPU, no discrete GPU) with a real 5h19m 1920x1080@60 Twitch VOD (17.1 GB):

| Step | Result |
|---|---|
| Audio scan of the full 5h19m VOD | 26 s, peak RAM ~125 MB (python) + ~120 MB (ffmpeg); cached afterwards |
| Seek + extract 50 s of audio at 5:15:00 | 0.3 s (MP4 and MKV), same as at the start of the file |
| Whisper `small.en` int8, 6 threads | ~4-10 s per 45 s clip (overlaps with rendering) |
| Render 45 s 1080x1920@60 clip, `h264_qsv` low-power | ~13-17 s (libx264: ~31 s) |
| **End to end, 10 clips from the 5h VOD** | **3 min 33 s** (~19 s per clip) |

- Encoder `auto` picks NVENC, then QSV, then AMF, each verified with a real test encode; libx264 is used only if none works. On Intel, the low-power (VDENC) QSV path is used when the driver supports it (~25% faster, same SSIM).
- `render.hwaccel: none` is the default: hardware decode was slower here because the crop/scale/caption filters run on the CPU anyway.
- `render.workers: 1` is the default: on a laptop iGPU there is one QSV engine and a 15 W CPU, so 2-3 parallel renders were slower (and hotter). Raise it on a desktop with NVENC.
- Clips are cut with `-ss` before `-i` (fast seek), so a clip at hour 5 costs the same as one at minute 5.
- The first run scans the whole audio; results are cached in `.cache/`, so re-runs with different clip/layout settings skip the scan.
- Set `captions.enabled: false` and `highlights.snap_to_sentences: false` to skip transcription entirely.
- Single-track VODs mix the streamer's mic with game audio, so game cutscenes can score high. If your OBS recording has a separate mic track, set `highlights.audio_track` to it.

## Troubleshooting

- **"no audio track"**: the VOD has no audio stream, so there is nothing to score. Try another file, or set `highlights.audio_track` if the audio is on a different stream index.
- **Facecam in the wrong place**: run `clipper preview`, look at `*_source.png`, then tune `layout.facecam` (or use `custom` with `region`).
- **Fonts**: captions use `captions.font`. On Windows installed fonts work directly; on Linux install the font or put `.ttf`/`.otf` files in a folder and set `captions.fonts_dir`. A missing font silently falls back to a default one.
- **CUDA / cuDNN DLL errors** (`cublas64_12.dll`, `cudnn_*` not found): install CUDA 12 + cuDNN 9, or set `transcription.device: cpu`.
- **Encoder not usable**: run `clipper doctor`; set `render.encoder: libx264` to force software encoding.
- **Leftover files**: temp files are removed automatically; use `--keep-temp` to inspect them.


## Responsible use

Only clip content you have the right to use. Get the creator's permission, or use official clipping programs that explicitly allow it, and follow the rules of the platform you post to (YouTube, TikTok, Twitch). Credit the original creator. The optional YouTube upload scripts in `tools/` need your own OAuth client; keep `secrets/` out of git (it is gitignored, as are `.env*`, tokens and `*.local.*` files).

## Contributing

Issues and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
