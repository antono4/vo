"""Optional real-model provider (BytePlus ModelArk — Dreamina-Seedance-2.5).

The application works fully offline with the procedural engine. When
`MODELARK_API_KEY` is configured and RENDER_ENGINE=auto, generation is routed
to the real Seedance 2.5 model instead. The skill's hard policy of forcing
`model_version=seedance_2.5` and the [4, 30]s range is preserved here.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config


class ProviderError(RuntimeError):
    pass


def available() -> bool:
    return bool(config.MODELARK_API_KEY)


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.MODELARK_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # pragma: no cover - network path
        raise ProviderError(f"ModelArk HTTP {e.code}: {e.read()[:300]!r}") from e
    except Exception as e:  # pragma: no cover - network path
        raise ProviderError(f"ModelArk request failed: {e}") from e


def _get(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {config.MODELARK_API_KEY}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # pragma: no cover
        return json.loads(resp.read().decode("utf-8"))


def generate(*, prompt: str, duration: int, ratio: str, image_urls: list[str] | None = None) -> Path:
    """Submit a generation task and poll until the video is available.

    Returns the path to the downloaded MP4 inside OUTPUT_DIR.
    """
    if not available():
        raise ProviderError("MODELARK_API_KEY is not configured")

    content: list[dict] = [{"type": "text", "text": prompt}]
    for url in image_urls or []:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": config.MODELARK_MODEL_ID,
        "model_version": config.FORCED_MODEL_VERSION,
        "duration": int(duration),
        "ratio": ratio,
        "content": content,
    }
    created = _post(f"{config.MODELARK_BASE_URL}/contents/generations/tasks", payload)
    task_id = created.get("id")
    if not task_id:
        raise ProviderError(f"no task id in response: {created}")

    deadline = time.time() + 900
    while time.time() < deadline:
        status = _get(f"{config.MODELARK_BASE_URL}/contents/generations/tasks/{task_id}")
        state = status.get("status")
        if state in ("succeeded", "success"):
            video_url = (status.get("content") or {}).get("video_url")
            if not video_url:
                raise ProviderError(f"succeeded without video_url: {status}")
            return _download(video_url)
        if state in ("failed", "error"):
            raise ProviderError(f"task {task_id} failed: {status.get('error')}")
        time.sleep(5)
    raise ProviderError(f"task {task_id} timed out")


def _download(url: str) -> Path:  # pragma: no cover - network path
    from .prompt_utils import slugify

    dest = config.OUTPUT_DIR / f"modelark_{slugify(url, 12)}_{int(time.time())}.mp4"
    urllib.request.urlretrieve(url, dest)
    return dest
