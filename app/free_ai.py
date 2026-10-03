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

# Free image endpoints: a0.dev (MarbelAI's choice) then pollinations.
_IMAGE_ENDPOINTS = [
    "https://api.a0.dev/assets/image",
    "https://image.pollinations.ai/prompt",
]

# Aspect -> (width, height) for the pollinations URL form.
_ASPECT_PX: dict[str, tuple[int, int]] = {
    "21:9": (1280, 548),
    "16:9": (1280, 720),
    "4:3": (1152, 864),
    "1:1": (1024, 1024),
    "3:4": (864, 1152),
    "9:16": (720, 1280),
}


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


def generate_image(prompt: str, dest: str, *, aspect: str = "1:1", seed: int | None = None) -> ImageResult:
    """Fetch a free AI image for the prompt and save it to disk.

    The extension of the returned path reflects the real image format, since
    the free endpoints may answer with PNG, JPEG or WEBP regardless of the
    requested name.
    """
    from pathlib import Path

    errors: list[str] = []
    for endpoint in _IMAGE_ENDPOINTS:
        try:
            if endpoint.endswith("/image"):  # a0.dev
                query = urllib.parse.urlencode({"text": prompt, "aspect": aspect, "seed": seed or 0})
                url = f"{endpoint}?{query}"
            else:  # pollinations
                width, height = _ASPECT_PX.get(aspect, _ASPECT_PX["1:1"])
                url = (
                    endpoint.rstrip("/")
                    + "/"
                    + urllib.parse.quote(prompt)
                    + f"?width={width}&height={height}&nologo=true&seed={seed or 0}"
                )
            raw = _read(url)
            if not raw:
                raise FreeAIError("empty image response")
            path = Path(dest).with_suffix(_sniff_ext(raw))
            path.write_bytes(raw)
            return ImageResult(path=str(path), provider=endpoint)
        except Exception as exc:  # noqa: BLE001 - try the next endpoint
            errors.append(f"{endpoint} -> {exc}")
    raise FreeAIError("all image providers failed: " + " | ".join(errors))
