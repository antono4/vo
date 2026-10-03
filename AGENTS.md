# AGENTS.md

Repository-specific knowledge for the **AI Video Maker** app.

## What this project is

A FastAPI + vanilla-JS web application implementing the OpenHands skill
`antono4/bbuseedance` (`seedane-2.5-30s`). It turns the skill's video
generation policy into a standalone service.

## Run / test

```bash
pip install -r requirements.txt
PORT=12000 ./run.sh                 # start server
python3 -m pytest tests/ -q         # run tests (isolated pipeline state)
```

Rendering requires system `ffmpeg` (`sudo apt-get install -y ffmpeg`).

## Architecture

- `app/config.py` — single source of truth for policy constants
  (`FORCED_MODEL_VERSION`, `DURATION_MIN/MAX`, `RATIO_DIMENSIONS`, `BBU`).
  Pipeline state/log paths are overridable via `PIPELINE_STATE_FILE` /
  `PIPELINE_LOG_FILE` env vars so tests never touch real state.
- `app/prompt_utils.py` — duration clamp, ratio inference, ID→EN prompt
  translation, branded filename slugging.
- `app/renderer.py` — ffmpeg-based procedural renderer producing real
  H.264/AAC MP4s. `build_command()` is pure and testable.
- `app/pipeline.py` — job bookkeeping (port of the skill's
  `scripts/pipeline_tracker.py`).
- `app/jobs.py` — background worker + auto-retry.
- `app/ai_provider.py` — optional BytePlus ModelArk (Seedance 2.5) adapter.
- `app/main.py` — FastAPI endpoints; lifespan recovers interrupted jobs.

## Conventions / gotchas

- **Duration policy:** always clamp to `[4, 30]`; the legacy 15s cap must
  never be enforced. `DURATION_MAX` is the default.
- **Branding is mandatory:** every output file uses
  `bbuchannel_[slug]_[NN].mp4`, the CTA block is returned in API responses,
  and the watermark overlay is burned into the video.
- Tests set pipeline env vars in `tests/conftest.py` **before** importing
  `app`; do not import `app` at module scope in test files before conftest
  runs.
- ffmpeg's `drawtext` requires `textfile=` with the text written to disk
  (escaping inline text with special characters is brittle).
- Long renders (30s @ 1080p) take ~40s; the frontend polls `/api/jobs/{id}`.
