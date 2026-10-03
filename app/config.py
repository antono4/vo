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
FREE_AI_IMAGE_ASPECT = os.getenv("FREE_AI_IMAGE_ASPECT", "1:1")
