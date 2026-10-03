"""Free AI capabilities ported from MarbelAIv2.1 (https://antono4.github.io/MarbelAIv2.1/).

MarbelAI is a chat UI + proxy over free, OpenAI-compatible providers that need
no API key. This module ports the same approach so the video maker can:

- enhance an Indonesian prompt into a richer English generation prompt
  (free chat completions), and
- generate a real reference image from the prompt (free image endpoint),
  which is then animated by the ffmpeg renderer (Ken Burns) — i.e. the app
  gains a real "AI image -> video" capability with zero credentials.

Providers are tried in order with per-request timeout and failover, mirroring
MarbelAI's sequential failover. Every network call is optional: when offline
(or when FREE_AI_ENABLED=0) the callers fall back to the existing local
behavior, so the app still works without any network access.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import config

# --- Free chat providers (no API key), mirrors MarbelAI DIRECT_UPSTREAMS ---
_UNCLOSEAI_MODEL = "turboderp/Qwen3.8-27B-exl3"
_CHAT_MODEL_MAP: dict[str, dict[str, str]] = {
    "https://hermes.ai.unturf.com": {
        "qwen3.8-27b": _UNCLOSEAI_MODEL,
        "gpt-oss-20b": _UNCLOSEAI_MODEL,
        "qwen3-8b": _UNCLOSEAI_MODEL,
    },
    "https://qwen.ai.unturf.com": {
        "qwen3.8-27b": _UNCLOSEAI_MODEL,
        "gpt-oss-20b": _UNCLOSEAI_MODEL,
        "qwen3-8b": _UNCLOSEAI_MODEL,
    },
    "https://text.pollinations.ai": {
        "qwen3.8-27b": "openai",
        "gpt-oss-20b": "openai",
        "qwen3-8b": "openai",
    },
    "https://api.free.ai": {
        "qwen3.8-27b": "qwen7b",
        "gpt-oss-20b": "qwen7b",
        "qwen3-8b": "qwen3-8b",
    },
}
# uncloseai runs vLLM/Qwen with thinking traces mixed into `content`;
# disable thinking so the answer is clean.
_CHAT_PAYLOAD: dict[str, dict] = {
    "https://hermes.ai.unturf.com": {"chat_template_kwargs": {"enable_thinking": False}},
    "https://qwen.ai.unturf.com": {"chat_template_kwargs": {"enable_thinking": False}},
}

# Free image endpoints, best quality first. Pollinations returns the largest
# free still (768px long edge) but throttles bursts with HTTP 402; a0.dev is
# lower resolution (~672px) but reliable. When pollinations throttles, its
# cooldown below routes the next shots straight to a0.dev instead of stalling.
_IMAGE_ENDPOINTS = [
    "https://image.pollinations.ai/prompt",
    "https://api.a0.dev/assets/image",
]

# endpoint -> unix time until which it should be skipped after a 402.
_ENDPOINT_COOLDOWN: dict[str, float] = {}

# Aspect -> (width, height) for the pollinations URL form, at full resolution.
# `image_size()` scales these down to the free tier's longest-edge cap.
_ASPECT_PX: dict[str, tuple[int, int]] = {
    "21:9": (1344, 576),
    "16:9": (1280, 720),
    "4:3": (1152, 864),
    "1:1": (1024, 1024),
    "3:4": (864, 1152),
    "9:16": (720, 1280),
}

# Quality tags appended to every image prompt. The free models default to a
# flat, illustration-like look, which reads as "AI slop" in motion; naming
# photographic specifics (lens, film, light, depth of field) is what pulls the
# output toward something that looks like real footage.
REALISM_TAGS = (
    "photorealistic, ultra detailed, natural lighting, shot on 35mm film, "
    "50mm lens, shallow depth of field, subtle film grain, cinematic color grading, "
    "sharp focus on the subject, high dynamic range"
)

_NEGATIVE_HINTS = "no text, no watermark, no logo, no caption"


def image_size(aspect: str, max_edge: int | None = None) -> tuple[int, int]:
    """Dimensions honouring the free tier's longest-edge cap, rounded to /16.

    Requesting 1280x720 from the keyless tier fails outright, so the aspect's
    full-resolution size is scaled down to `max_edge` on its long side.
    """
    width, height = _ASPECT_PX.get(aspect, _ASPECT_PX["16:9"])
    cap = max_edge or config.FREE_AI_IMAGE_MAX_EDGE
    longest = max(width, height)
    if cap and longest > cap:
        factor = cap / longest
        width, height = width * factor, height * factor
    snap = lambda v: max(16, int(math.floor(v / 16.0) * 16) or 16)  # noqa: E731
    return snap(width), snap(height)


def realism_prompt(prompt: str, *, aspect: str = "16:9") -> str:
    """Wrap a scene prompt in photographic quality tags, avoiding duplicates."""
    base = (prompt or "").strip().rstrip(".")
    low = base.lower()
    tags = [t for t in REALISM_TAGS.split(", ") if t.split()[0] not in low]
    if aspect in ("9:16", "3:4"):
        tags.append("vertical composition")
    return ", ".join([base] + tags)


class FreeAIError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    model: str
    provider: str


@dataclass
class ImageResult:
    path: str
    provider: str


def _sniff_ext(raw: bytes) -> str:
    """Pick the real image extension from magic bytes (free endpoints vary)."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if raw[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    if raw[:3] == b"GIF":
        return ".gif"
    return ".png"


def enabled() -> bool:
    return config.FREE_AI_ENABLED


def _chat_url(base: str) -> str:
    b = base.rstrip("/")
    if b == "https://text.pollinations.ai":
        return b + "/openai"
    return b + "/v1/chat/completions"


def _read(url: str, *, data: bytes | None = None, headers: dict | None = None,
          timeout: int | None = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout or config.FREE_AI_TIMEOUT) as resp:
        return resp.read()


def _fetch_image_with_retry(url: str) -> bytes:
    """GET an image, honouring the free tier's 402 rate limit.

    The keyless image tier answers `402 Payment Required` when requests come
    too fast. That is a throttle, not a hard failure, so wait and retry before
    giving up on the endpoint.
    """
    attempts = max(1, config.FREE_AI_IMAGE_RETRIES + 1)
    last: Exception | None = None
    for i in range(attempts):
        try:
            return _read(url, headers={"User-Agent": "ai-video-maker/1.0"})
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 402 and i < attempts - 1:
                time.sleep(config.FREE_AI_IMAGE_BACKOFF * (i + 1))
                continue
            raise
        except Exception as exc:  # noqa: BLE001 - network hiccup, retry
            last = exc
            if i < attempts - 1:
                time.sleep(config.FREE_AI_IMAGE_BACKOFF * (i + 1))
                continue
            raise
    raise FreeAIError(str(last) if last else "image fetch failed")


def _endpoint_available(endpoint: str) -> bool:
    return time.time() >= _ENDPOINT_COOLDOWN.get(endpoint, 0.0)


def _throttle_endpoint(endpoint: str) -> None:
    _ENDPOINT_COOLDOWN[endpoint] = time.time() + config.FREE_AI_IMAGE_COOLDOWN


def chat(messages: list[dict], model: str | None = None) -> ChatResult:
    """Free chat completion with sequential failover across providers."""
    model = model or config.FREE_AI_CHAT_MODEL
    errors: list[str] = []
    for base in config.FREE_AI_CHAT_UPSTREAMS:
        send_model = _CHAT_MODEL_MAP.get(base, {}).get(model, model)
        payload = {"model": send_model, "messages": messages, "stream": False}
        payload.update(_CHAT_PAYLOAD.get(base, {}))
        try:
            raw = _read(
                _chat_url(base),
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, str):
                text = data
            else:
                if data.get("error"):
                    err = data["error"]
                    raise FreeAIError(err.get("message") if isinstance(err, dict) else str(err))
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                text = message.get("content") or message.get("reasoning_content") or ""
            text = (text or "").strip()
            if not text:
                raise FreeAIError("model returned no content")
            return ChatResult(text=text, model=send_model, provider=base)
        except Exception as exc:  # noqa: BLE001 - try the next provider
            errors.append(f"{base} -> {exc}")
    raise FreeAIError("all chat providers failed: " + " | ".join(errors))


def enhance_prompt(user_prompt: str, *, model: str | None = None) -> str:
    """Turn an Indonesian/plain prompt into a richer English video prompt.

    Falls back to the original text when no provider is reachable.
    """
    if not enabled():
        return user_prompt
    system = (
        "You are a prompt engineer for the Seedance 2.5 text-to-video model. "
        "Rewrite the user's idea into ONE vivid English video prompt. "
        "Include subject, action, setting, lighting and camera motion. "
        "Keep it under 60 words, no quotes, no preamble, no lists."
    )
    try:
        result = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
        )
        return result.text.strip() or user_prompt
    except FreeAIError:
        return user_prompt


def generate_image(prompt: str, dest: str, *, aspect: str = "16:9", seed: int | None = None,
                   realistic: bool = True) -> ImageResult:
    """Fetch a free AI image for the prompt and save it to disk.

    The prompt is wrapped in photographic quality tags (unless `realistic=False`)
    and the requested size is clamped to the free tier's cap, so the returned
    still actually looks like footage rather than a flat illustration.

    The extension of the returned path reflects the real image format, since
    the free endpoints may answer with PNG, JPEG or WEBP regardless of the
    requested name.
    """
    from pathlib import Path

    request_prompt = realism_prompt(prompt, aspect=aspect) if realistic else prompt
    errors: list[str] = []
    endpoints = [e for e in _IMAGE_ENDPOINTS if _endpoint_available(e)] or list(_IMAGE_ENDPOINTS)
    for endpoint in endpoints:
        try:
            if endpoint.endswith("/image"):  # a0.dev
                query = urllib.parse.urlencode(
                    {"text": request_prompt + ", " + _NEGATIVE_HINTS,
                     "aspect": aspect, "seed": seed or 0}
                )
                url = f"{endpoint}?{query}"
            else:  # pollinations
                width, height = image_size(aspect)
                url = (
                    endpoint.rstrip("/")
                    + "/"
                    + urllib.parse.quote(request_prompt)
                    + f"?width={width}&height={height}&nologo=true&nofeed=true&seed={seed or 0}"
                )
            raw = _fetch_image_with_retry(url)
            if not raw:
                raise FreeAIError("empty image response")
            path = Path(dest).with_suffix(_sniff_ext(raw))
            path.write_bytes(raw)
            return ImageResult(path=str(path), provider=endpoint)
        except urllib.error.HTTPError as exc:
            if exc.code == 402:
                _throttle_endpoint(endpoint)
            errors.append(f"{endpoint} -> {exc}")
        except Exception as exc:  # noqa: BLE001 - try the next endpoint
            errors.append(f"{endpoint} -> {exc}")
        finally:
            if config.FREE_AI_IMAGE_DELAY > 0:
                time.sleep(config.FREE_AI_IMAGE_DELAY)
    raise FreeAIError("all image providers failed: " + " | ".join(errors))
