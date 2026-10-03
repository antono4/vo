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
- `app/scenes.py` — storyboard planner for the omni engine. Free LLM splits the
  idea into shots; `heuristic_plan()` is the offline fallback.
- `app/omni.py` — multi-shot renderer: per-shot camera move, xfade chain and a
  synthesized ambient bed with a downbeat on each cut. `build_command()` is pure.
- `app/gemini_omni.py` — real Gemini Omni Flash adapter (`GEMINI_API_KEY`).
- `app/free_ai.py` — keyless free AI ported from MarbelAIv2.1: multi-provider
  chat (prompt enhancement) and free image generation, with sequential
  failover. All calls are best-effort and degrade to offline behavior.
- `app/main.py` — FastAPI endpoints; lifespan recovers interrupted jobs.

## Conventions / gotchas

- **Duration policy:** always clamp to `[4, 30]`; the legacy 15s cap must
  never be enforced. `DURATION_MAX` is the default.
- **Batch size:** `MAX_COUNT` (default 5, `MAX_COUNT` env override) bounds the
  `count` field and `meta.max_count`. The frontend reads it dynamically.
- **Branding is mandatory:** every output file uses
  `bbuchannel_[slug]_[NN].mp4`, the CTA block is returned in API responses,
  and the watermark overlay is burned into the video.
- Tests set pipeline env vars in `tests/conftest.py` **before** importing
  `app`; do not import `app` at module scope in test files before conftest
  runs.
- ffmpeg's `drawtext` requires `textfile=` with the text written to disk
  (escaping inline text with special characters is brittle).
- **ffmpeg filter gotchas (learned the hard way):**
  - `zoompan` has no `t` variable; derive clip time from `on/fps`.
  - The eval parser rejects `(t>1)`; use `gt(t,1)`. Ungated `exp(-k*(t-b))`
    explodes for `t < b`, so gate every beat term.
  - A filter with no inputs (e.g. `sine`) must not be prefixed with a link
    label; emit one output label per chain.
- **Engines:** `classic` (1 image + Ken Burns), `omni` (multi-shot, default),
  `gemini` (real Gemini Omni Flash, needs `GEMINI_API_KEY`), `auto`.
  `jobs.resolve_engine()` degrades `gemini`/`auto` to `omni` without a key so
  the reported engine always matches the one that ran.
- **Omni duration maths:** `n` shots of `seg` overlapped by `xfade` must satisfy
  `n*seg - (n-1)*xfade == duration` (`omni.segment_geometry`).
- Long renders (30s @ 1080p) take ~40s; the frontend polls `/api/jobs/{id}`.
- **Free AI is optional:** `FREE_AI_ENABLED=0` forces offline mode. Network
  failures must never fail a job — `jobs._prepare_prompt()` catches them and
  keeps the original prompt. Tests stub `free_ai._read` to stay offline.
