"""Tests for the parameter policies and the rendering pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, pipeline, prompt_utils, renderer  # noqa: E402


# ---------------- Duration policy ----------------
@pytest.mark.parametrize("requested,expected", [
    (30, 30), (20, 20), (4, 4), (60, 30), (2, 4), (1, 4), (100, 30),
])
def test_clamp_duration(requested, expected):
    value, _ = prompt_utils.clamp_duration(requested)
    assert value == expected
    assert config.DURATION_MIN <= value <= config.DURATION_MAX


def test_clamp_notes():
    _, note_low = prompt_utils.clamp_duration(2)
    _, note_high = prompt_utils.clamp_duration(60)
    _, note_ok = prompt_utils.clamp_duration(20)
    assert note_low and "minimum" in note_low
    assert note_high and "maksimum" in note_high
    assert note_ok is None


def test_legacy_cap_not_enforced():
    assert config.LEGACY_TOOL_CAP == 15
    assert prompt_utils.clamp_duration(15)[0] == 15
    assert prompt_utils.clamp_duration(30)[0] == 30


# ---------------- Ratio inference ----------------
@pytest.mark.parametrize("text,expected", [
    ("video tiktok kucing", "9:16"),
    ("reels dance", "9:16"),
    ("video youtube sinematik", "16:9"),
    ("postingan instagram", "1:1"),
    ("tidak ada kata kunci", "16:9"),
])
def test_infer_ratio(text, expected):
    assert prompt_utils.infer_ratio(text) == expected


def test_explicit_ratio_wins():
    assert prompt_utils.infer_ratio("video tiktok", "16:9") == "16:9"
    assert prompt_utils.infer_ratio("apa saja", "21:9") == "21:9"
    assert prompt_utils.infer_ratio("apa saja", "invalid") == "16:9"


# ---------------- Filenames / branding ----------------
def test_branded_filename_format():
    name = prompt_utils.branded_filename("kucing berlari di pantai", 2)
    assert name.startswith("bbuchannel_")
    assert name.endswith("_02.mp4")
    assert len(name) <= 70


def test_slugify_strips_stopwords():
    slug = prompt_utils.slugify("buatkan video kucing berlari")
    assert "buatkan" not in slug
    assert "kucing" in slug


# ---------------- Prompt translation ----------------
def test_translate_keeps_proper_nouns_and_maps_motion():
    out = prompt_utils.translate_prompt("kucing berlari di pantai")
    assert "cat" in out and "running" in out and "beach" in out


def test_translate_english_passthrough():
    out = prompt_utils.translate_prompt("a cyberpunk city at night")
    assert "cyberpunk" in out


# ---------------- Pipeline ----------------
def test_pipeline_enqueue_and_update():
    pipeline.init_state()
    jid = pipeline.enqueue(
        mode="text_to_video", prompt="test scene", ratio="16:9",
        duration_requested=60, duration_clamped=30, filename="bbuchannel_test_01.mp4",
    )
    job = pipeline.get(jid)
    assert job["status"] == "pending"
    assert job["duration_clamped"] == 30
    assert job["model_version"] == config.FORCED_MODEL_VERSION

    pipeline.update(jid, status="done", url="/media/x.mp4")
    assert pipeline.get(jid)["status"] == "done"
    assert pipeline.get(jid)["url"] == "/media/x.mp4"


# ---------------- Real render ----------------
def test_render_produces_playable_mp4():
    result = renderer.render(
        prompt="cat running on a beach at sunset",
        duration=4,
        ratio="16:9",
        filename="bbuchannel_test_render_01.mp4",
    )
    assert result.path.exists()
    info = renderer.probe(result.path)
    assert float(info["duration"]) == pytest.approx(4.0, abs=0.3)
    result.path.unlink(missing_ok=True)
