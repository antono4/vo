"""Tests for the multi-shot Omni-lite engine and the Gemini Omni provider.

Network access is stubbed, so these exercise the real planning, geometry and
ffmpeg command construction offline. One integration test renders a real MP4
from synthetic frames and is skipped when ffmpeg is unavailable.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, gemini_omni, jobs, omni, scenes  # noqa: E402


# --- scene planning -------------------------------------------------------

def test_scene_count_scales_with_duration():
    assert scenes.scene_count(config.DURATION_MIN) == 1
    assert scenes.scene_count(10) == 2
    assert scenes.scene_count(30) == min(config.OMNI_MAX_SCENES, 6)


def test_scene_count_never_exceeds_max(monkeypatch):
    monkeypatch.setattr(config, "OMNI_MAX_SCENES", 3)
    monkeypatch.setattr(config, "OMNI_SCENE_SECONDS", 1)
    assert scenes.scene_count(30) == 3


def test_heuristic_plan_assigns_distinct_camera_moves():
    plan = scenes.heuristic_plan("kucing di pantai", 6)
    assert plan.source == "heuristic"
    assert len(plan.scenes) == 6
    assert len({s.camera for s in plan.scenes}) == 6
    assert all(s.prompt for s in plan.scenes)


def test_heuristic_plan_splits_multi_sentence_idea():
    plan = scenes.heuristic_plan("pantai saat senja. kota neon di malam hari.", 2)
    assert "pantai" in plan.scenes[0].prompt
    assert "kota" in plan.scenes[1].prompt


def test_plan_uses_llm_json(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_ENABLED", True)
    monkeypatch.setattr(
        scenes.free_ai,
        "chat",
        lambda *a, **k: type("R", (), {"text": json.dumps(
            {"scenes": [{"prompt": "shot one", "camera": "dolly"},
                        {"prompt": "shot two", "camera": "pan"}]}
        )})(),
    )
    plan = scenes.plan_scenes("apa saja", 10)
    assert plan.source == "llm"
    assert [s.prompt for s in plan.scenes] == ["shot one", "shot two"]
    assert plan.scenes[1].camera == "pan"


def test_plan_pads_llm_result_to_requested_count(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_ENABLED", True)
    monkeypatch.setattr(
        scenes.free_ai, "chat",
        lambda *a, **k: type("R", (), {"text": '{"scenes":[{"prompt":"only one"}]}'})(),
    )
    plan = scenes.plan_scenes("apa saja", 30)
    assert len(plan.scenes) == scenes.scene_count(30)


def test_plan_falls_back_when_llm_garbage(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_ENABLED", True)

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(scenes.free_ai, "chat", boom)
    plan = scenes.plan_scenes("kucing di pantai", 10)
    assert plan.source == "heuristic"
    assert len(plan.scenes) == 2


# --- segment geometry -----------------------------------------------------

@pytest.mark.parametrize("duration,n", [(30, 6), (10, 2), (12, 3), (4, 1), (30, 1)])
def test_segments_sum_to_requested_duration(duration, n):
    seg, xfade, _ = omni.segment_geometry(duration, n)
    total = n * seg - (n - 1) * xfade
    assert abs(total - duration) < 1e-6


def test_single_shot_has_no_xfade():
    seg, xfade, cut = omni.segment_geometry(8, 1)
    assert xfade == 0.0
    assert seg == 8.0
    assert cut == 8.0


# --- ffmpeg graph ---------------------------------------------------------

def test_camera_filter_uses_frame_counter_not_t():
    # zoompan has no `t`; clip time must come from `on/fps`.
    out = omni.camera_filter("slow push-in", 1920, 1080, 24)
    assert "zoompan" in out
    assert "on/24" in out
    assert "t/" not in out


def test_camera_filter_moves_differ_between_cameras():
    push = omni.camera_filter("slow push-in", 1280, 720, 24)
    pan = omni.camera_filter("gentle pan left to right", 1280, 720, 24)
    crane = omni.camera_filter("crane rise", 1280, 720, 24)
    assert push != pan != crane


def test_camera_filter_eases_the_move():
    # A linear ramp reads as a mechanical zoom; the move must be eased.
    out = omni.camera_filter("slow push-in", 1280, 720, 24)
    assert "3-2*" in out  # smoothstep polynomial
    assert "0.10*" in out  # ~10% travel


def test_grade_filters_are_photographic():
    grade = omni._grade_filters()
    joined = ",".join(grade)
    assert "curves=preset=medium_contrast" in joined
    assert "unsharp=" in joined
    assert "noise=" in joined
    assert "vignette=" in joined


def test_grade_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "REALISM", False)
    joined = ",".join(omni._grade_filters())
    assert "curves" not in joined and "noise" not in joined


def test_xfade_transition_varies_and_stays_soft():
    seen = {omni._xfade_transition(i) for i in range(5)}
    assert len(seen) == 5
    # Hard wipes (e.g. wipeup) would look like a slideshow.
    assert all(t in {"fade", "dissolve", "smoothleft", "fadeblack", "smoothright"} for t in seen)


def test_ambient_audio_gates_beats_with_gt():
    # `(t>b)` is rejected by ffmpeg's eval parser; `gt(t,b)` is required.
    out = omni.ambient_audio_filter(10, 4.6)
    assert "gt(t," in out
    assert "(t>" not in out
    assert out.endswith("afade=t=out:st=8.50:d=1.5")


def test_ambient_audio_emits_exactly_one_output_label():
    out = omni.ambient_audio_filter(10, 4.6)
    assert out.count("[a]") == 0  # caller appends the label


def test_build_command_has_one_xfade_per_join():
    inputs = [
        omni.OmniSceneInput(image=Path(f"/tmp/s{i}.png"), camera=c, index=i)
        for i, c in enumerate(["push-in", "pan", "crane"])
    ]
    cmd = omni.build_command(
        scene_inputs=inputs, duration=10, ratio="16:9",
        output=Path("/tmp/o.mp4"), text_dir=Path("/tmp"),
        title="T", subtitle="S", watermark="W",
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("xfade=transition=") == 2  # 3 shots -> 2 joins
    assert "xfade=transition=dissolve" in graph  # first join is a soft dissolve
    assert "-map" in cmd and "[aout]" in cmd


def test_build_command_requires_scenes():
    with pytest.raises(omni.OmniRenderError):
        omni.build_command(
            scene_inputs=[], duration=5, ratio="16:9", output=Path("/tmp/o.mp4"),
            text_dir=Path("/tmp"), title="t", subtitle="s", watermark="w",
        )


# --- engine resolution ----------------------------------------------------

def test_resolve_engine_maps_auto(monkeypatch):
    monkeypatch.setattr(gemini_omni, "available", lambda: False)
    assert jobs.resolve_engine("auto") == "omni"
    monkeypatch.setattr(gemini_omni, "available", lambda: True)
    assert jobs.resolve_engine("auto") == "gemini"


def test_resolve_engine_keeps_explicit_and_defaults(monkeypatch):
    monkeypatch.setattr(gemini_omni, "available", lambda: False)
    assert jobs.resolve_engine("classic") == "classic"
    assert jobs.resolve_engine("omni") == "omni"
    # gemini without a key must report the engine that really runs
    assert jobs.resolve_engine("gemini") == "omni"
    assert jobs.resolve_engine(None) == config.DEFAULT_ENGINE
    assert jobs.resolve_engine("nonsense") == config.DEFAULT_ENGINE
    monkeypatch.setattr(gemini_omni, "available", lambda: True)
    assert jobs.resolve_engine("gemini") == "gemini"
    assert jobs.resolve_engine("classic") == "classic"


# --- Gemini Omni provider -------------------------------------------------

def test_gemini_available_follows_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    assert gemini_omni.available() is False
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    assert gemini_omni.available() is True


def test_gemini_requires_key(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    with pytest.raises(gemini_omni.GeminiOmniError):
        gemini_omni.generate(prompt="x", duration=8, ratio="16:9")


def test_gemini_writes_inline_base64_video(monkeypatch, tmp_path):
    import base64

    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    payload = {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"mimeType": "video/mp4",
                                                   "data": base64.b64encode(b"MP4DATA").decode()}}]}}
        ]
    }
    monkeypatch.setattr(gemini_omni, "_request", lambda *a, **k: payload)
    path = gemini_omni.generate(prompt="a cat", duration=8, ratio="16:9")
    assert path.read_bytes() == b"MP4DATA"
    assert path.suffix == ".mp4"


def test_gemini_raises_on_unrecognised_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(gemini_omni, "_request", lambda *a, **k: {"weird": True})
    with pytest.raises(gemini_omni.GeminiOmniError):
        gemini_omni.generate(prompt="x", duration=8, ratio="16:9")


def test_gemini_parts_encode_reference_image(monkeypatch, tmp_path):
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    parts = gemini_omni._parts("prompt", [img])
    assert parts[0]["text"] == "prompt"
    assert parts[1]["inline_data"]["mime_type"] == "image/png"


# --- real render (integration) -------------------------------------------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_omni_render_produces_real_video(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    frames = []
    for i, src in enumerate(["testsrc=s=640x360:d=1", "testsrc2=s=640x360:d=1"]):
        frame = tmp_path / f"frame{i}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src,
             "-frames:v", "1", str(frame)],
            check=True,
        )
        frames.append((frame, "slow push-in" if i == 0 else "crane rise"))

    result = omni.render(
        prompt="tes render omni", duration=6, ratio="16:9",
        filename="omni_test_out.mp4", scene_images=frames,
        scene_prompts=["a", "b"], plan_source="heuristic",
    )
    assert result.path.exists()
    assert result.scene_count == 2

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(result.path)],
        capture_output=True, text=True,
    )
    assert abs(float(probe.stdout.strip()) - 6.0) < 0.5
