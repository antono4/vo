"""Pipeline bookkeeping — port of the skill's `scripts/pipeline_tracker.py`.

Tracks every generation job through its lifecycle so a continuous batch
never drops a pending job. State is persisted to data/pipeline.json and an
append-only log is written to data/pipeline.log.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from . import config

VALID_STATUSES = {
    "pending", "dispatched", "done", "failed", "retry",
    "extension_pending", "extension_dispatched",
}

_lock = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    try:
        with config.PIPELINE_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load() -> dict:
    with _lock:
        if config.PIPELINE_STATE_FILE.exists():
            try:
                return json.loads(config.PIPELINE_STATE_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}


def save(state: dict) -> None:
    with _lock:
        config.PIPELINE_STATE_FILE.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def init_state() -> dict:
    state = {
        "pipeline_id": f"pipe_{int(datetime.now(timezone.utc).timestamp())}",
        "model": config.FORCED_MODEL_VERSION,
        "skill": "seedane-2.5-30s",
        "true_duration_range": [config.DURATION_MIN, config.DURATION_MAX],
        "legacy_tool_cap_note": (
            f"{config.LEGACY_TOOL_CAP}s — informational only, NOT enforced"
        ),
        "started_at": now_iso(),
        "jobs": {},
    }
    save(state)
    _log(
        f"initialized model={config.FORCED_MODEL_VERSION} "
        f"true_duration=[{config.DURATION_MIN},{config.DURATION_MAX}]s"
    )
    return state


def ensure_state() -> dict:
    state = load()
    if not state:
        state = init_state()
    return state


def enqueue(*, mode: str, prompt: str, ratio: str, duration_requested: int,
            duration_clamped: int, image_urls: list[str] | None = None,
            title: str = "", filename: str = "") -> str:
    state = ensure_state()
    with _lock:
        jid = f"job_{len(state['jobs']) + 1:03d}"
        state["jobs"][jid] = {
            "id": jid,
            "status": "pending",
            "attempts": 0,
            "url": None,
            "error": None,
            "mode": mode,
            "prompt": prompt,
            "title": title,
            "filename": filename,
            "ratio": ratio,
            "duration_requested": int(duration_requested),
            "duration_clamped": int(duration_clamped),
            "model_version": config.FORCED_MODEL_VERSION,
            "enqueued_at": now_iso(),
            "updated_at": now_iso(),
        }
        if image_urls:
            state["jobs"][jid]["image_urls"] = image_urls
        save(state)
    _log(
        f"enqueued {jid} ({mode}) requested={duration_requested}s "
        f"clamped={duration_clamped}s ratio={ratio}"
    )
    return jid


def update(job_id: str, **fields) -> dict | None:
    state = ensure_state()
    with _lock:
        job = state["jobs"].get(job_id)
        if job is None:
            return None
        job.update(fields)
        job["updated_at"] = now_iso()
        save(state)
        return dict(job)


def get(job_id: str) -> dict | None:
    return ensure_state()["jobs"].get(job_id)


def interrupted_jobs() -> list[dict]:
    """Jobs left mid-flight by a previous process (crash/restart)."""
    return [
        j for j in ensure_state()["jobs"].values()
        if j.get("status") in ("pending", "dispatched", "retry", "extension_dispatched")
    ]


def requeue(job_id: str) -> None:
    update(job_id, status="pending", url=None, error=None)


def all_jobs() -> list[dict]:
    jobs = ensure_state()["jobs"]
    return sorted(jobs.values(), key=lambda j: j["id"], reverse=True)


def counts() -> dict:
    result = {s: 0 for s in VALID_STATUSES}
    for job in ensure_state()["jobs"].values():
        result[job.get("status", "pending")] = result.get(job.get("status", "pending"), 0) + 1
    return {k: v for k, v in result.items() if v}


def summary() -> dict:
    state = ensure_state()
    jobs = state.get("jobs", {})
    return {
        "pipeline_id": state.get("pipeline_id"),
        "skill": state.get("skill"),
        "model": state.get("model"),
        "true_duration_range": state.get("true_duration_range"),
        "total": len(jobs),
        "counts": counts(),
        "delivered": [
            {"id": j["id"], "url": j["url"], "duration": j.get("duration_clamped")}
            for j in jobs.values() if j.get("url")
        ],
    }
