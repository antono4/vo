"""Background generation worker with auto-retry and pipeline bookkeeping."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import (ai_provider, config, free_ai, gemini_omni, omni, pipeline,
               prompt_utils, renderer, scenes)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gen")
_active: set[str] = set()
_active_lock = threading.Lock()


def resolve_engine(engine: str | None) -> str:
    """Map the requested engine to a concrete one, honouring `auto`.

    `gemini` without a key degrades to `omni` so the engine reported on the job
    matches the one that actually ran.
    """
    if engine == "gemini" or engine == "auto":
        return "gemini" if gemini_omni.available() else "omni"
    if engine in config.ENGINES:
        return engine
    return config.DEFAULT_ENGINE


def submit(*, job_id: str, prompt: str, duration: int, ratio: str,
           filename: str, mode: str, image_path: Path | None = None,
           image_urls: list[str] | None = None, use_ai: bool = False,
           use_free_ai: bool = True, ai_model: str | None = None,
           engine: str | None = None) -> None:
    with _active_lock:
        _active.add(job_id)
    _executor.submit(
        _run, job_id, prompt, duration, ratio, filename, mode,
        image_path, image_urls, use_ai, use_free_ai, ai_model, engine,
    )


def _prepare_prompt(job_id: str, prompt: str, mode: str, use_free_ai: bool,
                    ai_model: str | None) -> tuple[str, str | None]:
    """Enhance the prompt and/or fetch a real AI image via the free providers.

    Returns (render_prompt, ai_image_path). Any failure is non-fatal: the
    caller keeps the original prompt and the offline procedural render.
    """
    if not (use_free_ai and free_ai.enabled()):
        return prompt, None

    render_prompt = prompt
    try:
        enhanced = free_ai.enhance_prompt(prompt, model=ai_model)
        if enhanced and enhanced != prompt:
            render_prompt = enhanced
            pipeline.update(job_id, prompt_enhanced=enhanced, ai_provider="free_ai")
    except Exception as exc:  # noqa: BLE001 - enhancement is best-effort
        pipeline.update(job_id, ai_error=f"enhance: {exc}"[:300])

    ai_image_path: str | None = None
    if mode == "text_to_video":
        dest = config.UPLOAD_DIR / f"freeai_{job_id}.png"
        try:
            result = free_ai.generate_image(
                render_prompt, str(dest), aspect=config.FREE_AI_IMAGE_ASPECT
            )
            ai_image_path = result.path
            pipeline.update(
                job_id, ai_image=True, ai_image_provider=result.provider,
                ai_image_path=result.path,
            )
        except Exception as exc:  # noqa: BLE001 - image is optional
            pipeline.update(job_id, ai_error=f"image: {exc}"[:300])
    return render_prompt, ai_image_path


def _plan_scenes(job_id: str, prompt: str, duration: int, ratio: str,
                 ai_model: str | None) -> list[tuple[Path, str]]:
    """Build the storyboard and fetch one AI image per shot.

    Returns the shots that produced a usable image. Partial failures are fine:
    the caller renders however many shots succeeded. An empty list means the
    omni engine cannot run and the caller should fall back.
    """
    plan = scenes.plan_scenes(prompt, duration, model=ai_model)
    shots: list[tuple[Path, str]] = []
    for scene in plan.scenes:
        dest = config.UPLOAD_DIR / f"omni_{job_id}_{scene.index:02d}.png"
        try:
            result = free_ai.generate_image(
                scene.prompt, str(dest), aspect=ratio, seed=scene.index + 1
            )
        except Exception as exc:  # noqa: BLE001 - skip this shot, keep the rest
            pipeline.update(job_id, ai_error=f"scene {scene.index}: {exc}"[:300])
            continue
        shots.append((Path(result.path), scene.camera))
    pipeline.update(
        job_id,
        scene_count=len(shots),
        scene_prompts=[s.prompt for s in plan.scenes],
        scene_plan_source=plan.source,
    )
    return shots


def _run(job_id, prompt, duration, ratio, filename, mode,
         image_path, image_urls, use_ai, use_free_ai, ai_model, engine) -> None:
    try:
        pipeline.update(job_id, status="dispatched", attempts=1)
        resolved = resolve_engine(engine)
        pipeline.update(job_id, engine=resolved)

        render_prompt, ai_image_path = _prepare_prompt(
            job_id, prompt, mode, use_free_ai, ai_model
        )
        last_error = ""
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                if use_ai and ai_provider.available():
                    path = ai_provider.generate(
                        prompt=prompt, duration=duration, ratio=ratio,
                        image_urls=image_urls,
                    )
                    filename = path.name
                elif resolved == "gemini" and gemini_omni.available():
                    ref = [image_path] if image_path else None
                    path = gemini_omni.generate(
                        prompt=render_prompt, duration=duration, ratio=ratio,
                        image_paths=ref,
                    )
                    filename = path.name
                elif resolved == "omni" and use_free_ai and free_ai.enabled():
                    shots = _plan_scenes(job_id, prompt, duration, ratio, ai_model)
                    if not shots:
                        raise RuntimeError(
                            "omni engine produced no scene images; "
                            "falling back to classic"
                        )
                    result = omni.render(
                        prompt=prompt, duration=duration, ratio=ratio,
                        filename=filename, scene_images=shots,
                    )
                    path = result.path
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
