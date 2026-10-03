"""Scene planning for the multi-shot "Omni-lite" engine.

A single still frame zoomed for 30 seconds does not read as AI video. Real
video models (Gemini Omni, Seedance) cut between shots. This module splits a
user idea into a small storyboard — one prompt per shot — so the renderer can
generate a distinct AI image per scene and crossfade between them.

The free LLM is asked to produce the storyboard; when it is unreachable the
heuristic splitter below still yields a usable multi-scene plan, so the engine
never depends on the network.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import config, free_ai

# Camera moves are assigned deterministically so a scene's look is stable
# across runs (same prompt -> same plan) and every shot moves differently.
_CAMERA_MOVES = [
    "slow push-in",
    "gentle pan left to right",
    "crane rise",
    "slow pull-back reveal",
    "handheld drift",
    "orbit around the subject",
]

# Ordered fallback beats; each is appended to the base idea when the LLM is
# unavailable. Kept generic enough to fit nature, city, product and people.
_BEATS = [
    "wide establishing shot, natural ambient light",
    "medium shot of the main subject, shallow depth of field",
    "macro close-up detail, soft bokeh background",
    "over-the-shoulder angle, wider context, golden hour light",
    "atmospheric wide shot, volumetric light and haze",
    "final beauty shot, warm rim light, high dynamic range",
]


@dataclass
class Scene:
    prompt: str
    camera: str
    index: int = 0


@dataclass
class ScenePlan:
    scenes: list[Scene] = field(default_factory=list)
    source: str = "heuristic"  # 'llm' | 'heuristic'

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.scenes)


def scene_count(duration: int) -> int:
    """Number of shots for a duration, clamped to the configured maximum."""
    count = round(duration / max(1, config.OMNI_SCENE_SECONDS))
    return max(1, min(config.OMNI_MAX_SCENES, count))


def _split_idea(text: str) -> list[str]:
    """Break a prompt into distinct ideas on sentence/clause boundaries."""
    parts = re.split(r"(?<=[.!?;])\s+|\s*\|\s*|\s+-\s+", text or "")
    parts = [p.strip(" .!?;,") for p in parts]
    return [p for p in parts if len(p) > 2]


def heuristic_plan(prompt: str, count: int) -> ScenePlan:
    """Deterministic storyboard built from the prompt itself."""
    ideas = _split_idea(prompt) or [prompt.strip() or "a cinematic scene"]
    scenes: list[Scene] = []
    for i in range(count):
        base = ideas[i % len(ideas)]
        beat = _BEATS[i % len(_BEATS)]
        camera = _CAMERA_MOVES[i % len(_CAMERA_MOVES)]
        scenes.append(
            Scene(
                prompt=f"{base}, {beat}",
                camera=camera,
                index=i,
            )
        )
    return ScenePlan(scenes=scenes, source="heuristic")


def _coerce_scenes(raw, count: int) -> list[Scene] | None:
    """Normalise whatever the LLM returned into `count` scenes."""
    if isinstance(raw, dict):
        raw = raw.get("scenes") or raw.get("shots") or []
    if not isinstance(raw, list) or not raw:
        return None
    scenes: list[Scene] = []
    for item in raw:
        if isinstance(item, str):
            text, camera = item, ""
        elif isinstance(item, dict):
            text = item.get("prompt") or item.get("description") or item.get("shot") or ""
            camera = item.get("camera") or item.get("camera_move") or ""
        else:
            continue
        text = str(text).strip()
        if not text:
            continue
        scenes.append(Scene(prompt=text, camera=str(camera).strip()))
    if not scenes:
        return None
    # Pad by cycling so the requested shot count is always honoured.
    while len(scenes) < count:
        scenes.append(scenes[len(scenes) % len(scenes)])
    return scenes[:count]


def plan_scenes(prompt: str, duration: int, *, model: str | None = None,
                use_llm: bool = True) -> ScenePlan:
    """Build a storyboard for `prompt`, preferring the free LLM."""
    count = scene_count(duration)
    if count <= 1 or not (use_llm and free_ai.enabled()):
        return heuristic_plan(prompt, count)

    system = (
        "You are a storyboard artist for a photorealistic AI video model. Split the "
        f"user's idea into exactly {count} consecutive shots that together read as one "
        "coherent, live-action video. Return ONLY JSON: "
        "{\"scenes\":[{\"prompt\":\"...\",\"camera\":\"...\"}]}. "
        "Each prompt must be one vivid English sentence naming the subject, action, "
        "setting, lighting and lens feel (e.g. 'shallow depth of field', 'golden hour', "
        "'35mm film'), as if describing real camera footage rather than an illustration. "
        "Each camera must name a camera move. No markdown, no preamble."
    )
    try:
        result = free_ai.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=model,
        )
        text = result.text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group(0) if match else text)
        scenes = _coerce_scenes(data, count)
        if scenes:
            for i, scene in enumerate(scenes):
                scene.index = i
                if not scene.camera:
                    scene.camera = _CAMERA_MOVES[i % len(_CAMERA_MOVES)]
            return ScenePlan(scenes=scenes, source="llm")
    except Exception:  # noqa: BLE001 - any LLM/parse failure falls back
        pass
    return heuristic_plan(prompt, count)


def scene_aspect(ratio: str) -> str:
    """Map an output ratio to the aspect hint sent to the image provider."""
    return ratio if ratio in config.SUPPORTED_RATIOS else "16:9"
