"""Background generation worker with auto-retry and pipeline bookkeeping."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import ai_provider, config, pipeline, prompt_utils, renderer

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gen")
_active: set[str] = set()
_active_lock = threading.Lock()


def submit(*, job_id: str, prompt: str, duration: int, ratio: str,
           filename: str, mode: str, image_path: Path | None = None,
           image_urls: list[str] | None = None, use_ai: bool = False) -> None:
    with _active_lock:
        _active.add(job_id)
    _executor.submit(
        _run, job_id, prompt, duration, ratio, filename, mode,
        image_path, image_urls, use_ai,
    )


def _run(job_id, prompt, duration, ratio, filename, mode,
         image_path, image_urls, use_ai) -> None:
    try:
        pipeline.update(job_id, status="dispatched", attempts=1)
        last_error = ""
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                if use_ai and ai_provider.available():
                    path = ai_provider.generate(
                        prompt=prompt, duration=duration, ratio=ratio,
                        image_urls=image_urls,
                    )
                    filename = path.name
                else:
                    result = renderer.render(
                        prompt=prompt, duration=duration, ratio=ratio,
                        filename=filename, image_path=image_path,
                    )
                    path = result.path
                pipeline.update(
                    job_id,
                    status="done",
                    attempts=attempt,
                    url=f"/media/{filename}",
                    error=None,
                    filename=filename,
                )
                return
            except Exception as exc:  # noqa: BLE001 - surface any failure
                last_error = str(exc)
                pipeline.update(
                    job_id,
                    status="retry" if attempt < config.MAX_RETRIES else "failed",
                    attempts=attempt,
                    error=last_error[:500],
                )
        pipeline.update(job_id, status="failed", error=last_error[:500])
    finally:
        with _active_lock:
            _active.discard(job_id)


def active_count() -> int:
    with _active_lock:
        return len(_active)
