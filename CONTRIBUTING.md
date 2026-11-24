# Contributing

Thanks for helping out.

1. Fork and clone, then `pip install -e ".[dev]"` (ffmpeg and ffprobe must be on PATH).
2. Make one logical change per pull request. Add or update tests in `tests/`.
3. Run `pytest -q` and make sure it passes.
4. Use Conventional Commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
5. Never commit secrets, `.env*`, OAuth tokens, VODs or rendered output.

New streamer layouts go in `configs/creators/` (check them with `clipper preview`). Bug reports: include your OS, Python version, ffmpeg version and the command you ran.
