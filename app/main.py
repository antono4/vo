"""AI Video Maker — FastAPI application.

A self-contained AI video maker built on the `seedane-2.5-30s` skill
specification: built-in text_to_video / image_to_video, forced
model_version=seedance_2.5, true duration range [4, 30]s, ratio inference,
continuous pipelined generation with auto-retry, and mandatory BBU CHANNEL
branded delivery.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ai_provider, config, free_ai, jobs, pipeline, prompt_utils


def _recover_interrupted_jobs() -> None:
    """Requeue jobs that a previous process left mid-flight."""
    for job in pipeline.interrupted_jobs():
        image_path = None
        if job.get("image_urls"):
            image_path = config.UPLOAD_DIR / Path(job["image_urls"][0]).name
        pipeline.requeue(job["id"])
        jobs.submit(
            job_id=job["id"],
            prompt=job["prompt"],
            duration=job["duration_clamped"],
            ratio=job["ratio"],
            filename=job.get("filename") or f"{config.BBU['file_prefix']}{job['id']}.mp4",
            mode=job["mode"],
            image_path=image_path,
            image_urls=job.get("image_urls"),
            use_free_ai=job.get("use_free_ai", True),
            ai_model=job.get("ai_model"),
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _recover_interrupted_jobs()
    yield


app = FastAPI(title="AI Video Maker", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    duration: int = config.DURATION_DEFAULT
    ratio: str | None = None
    count: int = Field(1, ge=1, le=4)
    images: list[str] = Field(default_factory=list)
    use_ai: bool = False
    use_free_ai: bool = True
    ai_model: str | None = None
    title: str | None = None


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": config.FORCED_MODEL_VERSION,
        "duration_range": [config.DURATION_MIN, config.DURATION_MAX],
        "legacy_tool_cap": config.LEGACY_TOOL_CAP,
        "render_engine": config.RENDER_ENGINE,
        "ai_provider_available": ai_provider.available(),
        "free_ai": {
            "enabled": free_ai.enabled(),
            "default_model": config.FREE_AI_CHAT_MODEL,
            "models": config.FREE_AI_MODELS,
            "upstreams": config.FREE_AI_CHAT_UPSTREAMS,
            "source": "MarbelAIv2.1",
        },
        "active_jobs": jobs.active_count(),
    }


@app.get("/api/meta")
def meta() -> dict:
    return {
        "model_version": config.FORCED_MODEL_VERSION,
        "duration": {
            "min": config.DURATION_MIN,
            "max": config.DURATION_MAX,
            "default": config.DURATION_DEFAULT,
            "legacy_tool_cap": config.LEGACY_TOOL_CAP,
            "note": "Model asli mendukung 30s; cap 15s lama tidak dipakai.",
        },
        "ratios": config.SUPPORTED_RATIOS,
        "default_ratio": config.DEFAULT_RATIO,
        "max_count": 4,
        "modes": ["text_to_video", "image_to_video"],
        "free_ai": {
            "enabled": free_ai.enabled(),
            "default_model": config.FREE_AI_CHAT_MODEL,
            "models": config.FREE_AI_MODELS,
            "note": (
                "Kemampuan AI gratis dari MarbelAIv2.1: prompt enhancement + "
                "gambar referensi AI (keyless) yang dianimasikan oleh renderer."
            ),
        },
        "branding": config.BBU,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Tipe file tidak didukung: {file.content_type}")
    suffix = Path(file.filename or "image.png").suffix or ".png"
    name = f"upload_{uuid.uuid4().hex[:10]}{suffix}"
    dest = config.UPLOAD_DIR / name
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(400, "Ukuran file maksimal 25MB")
    dest.write_bytes(data)
    return {"url": f"/uploads/{name}", "name": file.filename}


@app.post("/api/generate")
def generate(req: GenerateRequest) -> JSONResponse:
    duration, note = prompt_utils.clamp_duration(req.duration)
    ratio = prompt_utils.infer_ratio(req.prompt, req.ratio)
    english_prompt = prompt_utils.translate_prompt(req.prompt)
    mode = "image_to_video" if req.images else "text_to_video"

    if mode == "image_to_video":
        for url in req.images:
            if not (config.UPLOAD_DIR / Path(url).name).exists():
                raise HTTPException(400, f"Gambar tidak ditemukan: {url}")

    created = []
    for i in range(1, req.count + 1):
        filename = prompt_utils.branded_filename(req.prompt, i)
        if (config.OUTPUT_DIR / filename).exists():
            stem = filename[:-4]
            filename = f"{stem}_{uuid.uuid4().hex[:4]}.mp4"
        jid = pipeline.enqueue(
            mode=mode,
            prompt=english_prompt,
            ratio=ratio,
            duration_requested=req.duration,
            duration_clamped=duration,
            image_urls=req.images or None,
            title=req.title or "",
            filename=filename,
        )
        pipeline.update(
            jid,
            use_free_ai=req.use_free_ai,
            ai_model=req.ai_model or config.FREE_AI_CHAT_MODEL,
        )
        image_path = None
        if req.images:
            image_path = config.UPLOAD_DIR / Path(req.images[0]).name
        jobs.submit(
            job_id=jid,
            prompt=english_prompt,
            duration=duration,
            ratio=ratio,
            filename=filename,
            mode=mode,
            image_path=image_path,
            image_urls=req.images or None,
            use_ai=req.use_ai,
            use_free_ai=req.use_free_ai,
            ai_model=req.ai_model,
        )
        created.append({"job_id": jid, "filename": filename, "mode": mode})

    return JSONResponse(
        {
            "jobs": created,
            "duration": duration,
            "duration_requested": req.duration,
            "ratio": ratio,
            "mode": mode,
            "english_prompt": english_prompt,
            "notes": [note] if note else [],
            "cta": config.BBU["cta"],
        }
    )


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {"jobs": pipeline.all_jobs(), "counts": pipeline.counts()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = pipeline.get(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    return job


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    job = pipeline.get(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    if job["status"] == "done":
        raise HTTPException(400, "Job sudah selesai")
    image_path = None
    if job.get("image_urls"):
        image_path = config.UPLOAD_DIR / Path(job["image_urls"][0]).name
    pipeline.requeue(job_id)
    jobs.submit(
        job_id=job_id,
        prompt=job["prompt"],
        duration=job["duration_clamped"],
        ratio=job["ratio"],
        filename=job.get("filename") or f"{config.BBU['file_prefix']}{job_id}.mp4",
        mode=job["mode"],
        image_path=image_path,
        image_urls=job.get("image_urls"),
        use_free_ai=job.get("use_free_ai", True),
        ai_model=job.get("ai_model"),
    )
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/pipeline")
def get_pipeline() -> dict:
    return pipeline.summary()


@app.get("/media/{filename}")
def media(filename: str):
    path = (config.OUTPUT_DIR / Path(filename).name)
    if not path.exists():
        raise HTTPException(404, "Video tidak ditemukan")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/uploads/{filename}")
def uploads(filename: str):
    path = (config.UPLOAD_DIR / Path(filename).name)
    if not path.exists():
        raise HTTPException(404, "File tidak ditemukan")
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
