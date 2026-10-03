"""Real Gemini Omni Flash provider (Google AI / Gemini API).

Docs: https://ai.google.dev/gemini-api/docs/omni

`gemini-omni-1.1-flash` generates a video (with audio) from a text prompt and
optional reference images, including first/last-frame interpolation. Unlike the
keyless free providers, this needs `GEMINI_API_KEY`; without it `available()`
is False and the app falls back to the free Omni-lite engine.

The response shape varies between Gemini API revisions, so the parser accepts
an inline base64 payload, a returned file URI, or a long-running operation that
must be polled. The function is intentionally defensive: any unrecognised
shape raises `GeminiOmniError` rather than silently producing a bad file.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config, prompt_utils


class GeminiOmniError(RuntimeError):
    pass


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


def _request(url: str, payload: dict | None = None, method: str = "POST") -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if config.GEMINI_API_KEY:
        headers["x-goog-api-key"] = config.GEMINI_API_KEY
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # pragma: no cover - network path
        body = e.read().decode("utf-8", "ignore")
        raise GeminiOmniError(f"Gemini HTTP {e.code}: {body[:400]}") from e
    except Exception as e:  # pragma: no cover - network path
        raise GeminiOmniError(f"Gemini request failed: {e}") from e


def _endpoint(model: str, action: str) -> str:
    base = config.GEMINI_API_BASE.rstrip("/")
    url = f"{base}/models/{model}:{action}"
    # Key may also be passed as a query parameter; header is preferred.
    return url


def _parts(prompt: str, image_paths: list[Path] | None) -> list[dict]:
    parts: list[dict] = [{"text": prompt}]
    for path in image_paths or []:
        raw = Path(path).read_bytes()
        mime = "image/png"
        if raw[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            mime = "image/webp"
        parts.append(
            {"inline_data": {"mime_type": mime, "data": base64.b64encode(raw).decode()}}
        )
    return parts


def _extract_video_bytes(payload: dict) -> bytes | None:
    """Pull base64 video out of any known response shape."""
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            inline = node.get("inlineData") or node.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type") or ""
                if "video" in mime or mime == "":
                    return base64.b64decode(inline["data"])
            for key in ("video", "output", "result", "response", "content",
                        "candidates", "parts"):
                if key in node:
                    stack.append(node[key])
        elif isinstance(node, list):
            stack.extend(node)
    return None


def _extract_file_uri(payload: dict) -> str | None:
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            uri = node.get("uri") or node.get("fileUri") or node.get("file_uri")
            if isinstance(uri, str) and uri.startswith("http"):
                return uri
            for key in ("file", "video", "output", "result", "response",
                        "content", "parts"):
                if key in node:
                    stack.append(node[key])
        elif isinstance(node, list):
            stack.extend(node)
    return None


def generate(*, prompt: str, duration: int, ratio: str,
             image_paths: list[Path] | None = None) -> Path:
    """Generate a video with Gemini Omni Flash and return the local MP4 path."""
    if not available():
        raise GeminiOmniError("GEMINI_API_KEY is not configured")

    payload = {
        "contents": [{"role": "user", "parts": _parts(prompt, image_paths)}],
        "generationConfig": {
            "responseModalities": ["VIDEO"],
            "videoConfig": {"resolution": config.GEMINI_OMNI_RESOLUTION},
        },
    }
    data = _request(_endpoint(config.GEMINI_OMNI_MODEL, "generateContent"), payload)

    # Long-running operation? Poll until it finishes.
    op_name = data.get("name")
    if op_name and not data.get("done", True):
        deadline = time.time() + config.GEMINI_OMNI_TIMEOUT
        while time.time() < deadline:
            time.sleep(config.GEMINI_OMNI_POLL_SECONDS)
            data = _request(
                f"{config.GEMINI_API_BASE.rstrip('/')}/{op_name}", method="GET"
            )
            if data.get("done"):
                break
        if not data.get("done"):
            raise GeminiOmniError("Gemini Omni operation timed out")

    dest = config.OUTPUT_DIR / (
        f"gemini_{prompt_utils.slugify(prompt, 24)}_{int(time.time())}.mp4"
    )
    raw = _extract_video_bytes(data)
    if raw:
        dest.write_bytes(raw)
        return dest

    uri = _extract_file_uri(data)
    if uri:
        if not uri.endswith(":download") and "alt=media" not in uri:
            uri = uri + ("&alt=media" if "?" in uri else "?alt=media")
        req = urllib.request.Request(uri, headers={"x-goog-api-key": config.GEMINI_API_KEY or ""})
        with urllib.request.urlopen(req, timeout=300) as resp:  # pragma: no cover
            dest.write_bytes(resp.read())
        return dest

    raise GeminiOmniError(f"no video in Gemini response: {json.dumps(data)[:400]}")
