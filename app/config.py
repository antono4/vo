"""Application configuration and constants.

Mirrors the hard policies of the upstream `seedane-2.5-30s` skill:
forced model version, true duration range [4, 30], ratio inference,
BBU CHANNEL branding.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR = DATA_DIR / "uploads"

for _d in (DATA_DIR, OUTPUT_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Forced model (skill policy: seedance_2.5 only) ---
FORCED_MODEL_VERSION = "seedance_2.5"

# --- True model duration range (skill rejects the legacy 15s tool cap) ---
DURATION_MIN = 4
DURATION_MAX = 30
DURATION_DEFAULT = 30
LEGACY_TOOL_CAP = 15  # informational only, never enforced
MAX_COUNT = int(os.getenv("MAX_COUNT", "5"))

# --- Generation output ---
FPS = int(os.getenv("VIDEO_FPS", "24"))
MAX_RETRIES = int(os.getenv("VIDEO_MAX_RETRIES", "3"))
DEFAULT_RESOLUTION = os.getenv("VIDEO_RESOLUTION", "1080p")

# ratio -> (width, height) at the 1080p tier
RATIO_DIMENSIONS: dict[str, tuple[int, int]] = {
    "21:9": (1920, 824),
    "16:9": (1920, 1080),
    "4:3": (1440, 1080),
    "1:1": (1080, 1080),
    "3:4": (1080, 1440),
    "9:16": (1080, 1920),
}
SUPPORTED_RATIOS = list(RATIO_DIMENSIONS)
DEFAULT_RATIO = "16:9"

# --- BBU CHANNEL branding (mandatory per skill policy) ---
BBU = {
    "channel_name": "BBU CHANNEL",
    "channel_url": "https://www.youtube.com/@BBUchannel13",
    "keyword": "BBUMANTAP",
    "file_prefix": "bbuchannel_",
    "cta": (
        "---\n"
        "🎬 Video dibuat dengan skill dari BBU CHANNEL\n\n"
        "📺 Tutorial lengkap & skill gratis lainnya:\n"
        "   youtube.com/@BBUchannel13\n\n"
        "🔑 Ketik BBUMANTAP di kolom komentar video untuk dapat\n"
        "   file skill ini secara gratis!\n"
        "---"
    ),
    "watermark_overlay_template": "BBU CHANNEL | youtube.com/@BBUchannel13",
}

# --- Pipeline ---
PIPELINE_STATE_FILE = Path(
    os.getenv("PIPELINE_STATE_FILE", str(DATA_DIR / "pipeline.json"))
)
PIPELINE_LOG_FILE = Path(
    os.getenv("PIPELINE_LOG_FILE", str(DATA_DIR / "pipeline.log"))
)

# --- Rendering engine ---
# 'procedural' renders a deterministic animated video with ffmpeg (works
# offline, always available). 'auto' additionally enables a real AI model
# provider when MODELARK_API_KEY is configured.
RENDER_ENGINE = os.getenv("RENDER_ENGINE", "procedural")
MODELARK_API_KEY = os.getenv("MODELARK_API_KEY")
MODELARK_BASE_URL = os.getenv("MODELARK_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3")
MODELARK_MODEL_ID = os.getenv("MODELARK_MODEL_ID", "dreamina-seedance-2-5")

# --- Free AI capabilities (ported from MarbelAIv2.1) ---
# Keyless, OpenAI-compatible providers used to (a) enhance the prompt and
# (b) generate a real reference image that the ffmpeg renderer animates.
# Enabled by default; set FREE_AI_ENABLED=0 to force fully-offline behavior.
FREE_AI_ENABLED = os.getenv("FREE_AI_ENABLED", "1") not in ("0", "false", "False")
FREE_AI_TIMEOUT = int(os.getenv("FREE_AI_TIMEOUT", "30"))
FREE_AI_CHAT_MODEL = os.getenv("FREE_AI_CHAT_MODEL", "qwen3.8-27b")
FREE_AI_CHAT_UPSTREAMS = [
    u.strip()
    for u in os.getenv(
        "FREE_AI_CHAT_UPSTREAMS",
        "https://hermes.ai.unturf.com,https://qwen.ai.unturf.com,"
        "https://text.pollinations.ai,https://api.free.ai",
    ).split(",")
    if u.strip()
]
FREE_AI_MODELS = ["qwen3.8-27b", "gpt-oss-20b", "qwen3-8b"]
FREE_AI_IMAGE_ASPECT = os.getenv("FREE_AI_IMAGE_ASPECT", "16:9")
# The free image tier is capped at 768px on its longest edge (a 768x768 request
# succeeds, 1024x1024 and 1280x720 are rejected with HTTP 402) and rate-limits
# bursts, so the longest requestable edge and the delay between requests are
# configurable.
FREE_AI_IMAGE_MAX_EDGE = int(os.getenv("FREE_AI_IMAGE_MAX_EDGE", "768"))
FREE_AI_IMAGE_DELAY = float(os.getenv("FREE_AI_IMAGE_DELAY", "1.0"))
FREE_AI_IMAGE_RETRIES = int(os.getenv("FREE_AI_IMAGE_RETRIES", "2"))
FREE_AI_IMAGE_BACKOFF = float(os.getenv("FREE_AI_IMAGE_BACKOFF", "3.0"))
# After a 402 the throttled endpoint is skipped for this long, so a multi-shot
# job fails over to the lower-resolution endpoint instead of stalling on retries.
FREE_AI_IMAGE_COOLDOWN = float(os.getenv("FREE_AI_IMAGE_COOLDOWN", "25"))

# --- Generation engine ---
# 'classic'  : one AI image + Ken Burns + title/watermark (the original engine)
# 'omni'     : multi-scene "Omni-lite" — several AI images, per-shot camera
#              motion, crossfades and a synchronized ambient audio bed
# 'gemini'   : real Gemini Omni Flash (gemini-omni-1.1-flash) via GEMINI_API_KEY
# 'auto'     : gemini when a key is present, otherwise omni
ENGINES = ["classic", "omni", "gemini"]
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "omni")

# --- Omni-lite engine tuning ---
OMNI_SCENE_SECONDS = int(os.getenv("OMNI_SCENE_SECONDS", "5"))
OMNI_MAX_SCENES = int(os.getenv("OMNI_MAX_SCENES", "6"))
OMNI_XFADE_SECONDS = float(os.getenv("OMNI_XFADE_SECONDS", "0.8"))
OMNI_TITLE_SECONDS = float(os.getenv("OMNI_TITLE_SECONDS", "3.0"))

# --- Realism (photographic look) ---
# Applied to every shot. These turn the flat, upscaled AI stills into
# something that reads as camera footage: a gentle film curve, a touch of
# local contrast, film grain and a soft vignette.
REALISM = os.getenv("REALISM", "1") not in ("0", "false", "False")
REALISM_GRAIN = float(os.getenv("REALISM_GRAIN", "7"))       # noise strength (0 disables)
REALISM_VIGNETTE = float(os.getenv("REALISM_VIGNETTE", "0.55"))
REALISM_SHARPEN = float(os.getenv("REALISM_SHARPEN", "1.0"))  # unsharp amount
REALISM_UPSCALE = os.getenv("REALISM_UPSCALE", "lanczos")
# Blend a small amount of the previous frame back in so stills gain subtle
# temporal texture even within a single shot (0 disables).
REALISM_TEMPORAL = float(os.getenv("REALISM_TEMPORAL", "0.12"))

# --- Real Gemini Omni Flash provider (https://ai.google.dev/gemini-api/docs/omni) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"
)
GEMINI_OMNI_MODEL = os.getenv("GEMINI_OMNI_MODEL", "gemini-omni-1.1-flash")
GEMINI_OMNI_RESOLUTION = os.getenv("GEMINI_OMNI_RESOLUTION", "1080p")
GEMINI_OMNI_POLL_SECONDS = int(os.getenv("GEMINI_OMNI_POLL_SECONDS", "10"))
GEMINI_OMNI_TIMEOUT = int(os.getenv("GEMINI_OMNI_TIMEOUT", "600"))
