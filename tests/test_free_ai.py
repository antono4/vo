"""Tests for the free AI capabilities ported from MarbelAIv2.1.

All network access is stubbed by monkeypatching ``free_ai._read`` so the
tests exercise the real failover / parsing logic offline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, free_ai, jobs  # noqa: E402


def _completion(text: str, model: str = "m") -> bytes:
    return json.dumps(
        {"model": model, "choices": [{"message": {"content": text}}]}
    ).encode("utf-8")


def test_config_models_and_default():
    assert config.FREE_AI_MODELS == ["qwen3.8-27b", "gpt-oss-20b", "qwen3-8b"]
    assert config.FREE_AI_CHAT_MODEL in config.FREE_AI_MODELS


def test_chat_returns_content(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_CHAT_UPSTREAMS", ["https://a.test"])
    monkeypatch.setattr(free_ai, "_read", lambda *a, **k: _completion("hello"))
    result = free_ai.chat([{"role": "user", "content": "hi"}])
    assert result.text == "hello"
    assert result.provider == "https://a.test"


def test_chat_failover_to_second_provider(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_CHAT_UPSTREAMS", ["https://bad.test", "https://good.test"])

    def fake_read(url, **kwargs):
        if "bad.test" in url:
            raise OSError("boom")
        return _completion("ok")

    monkeypatch.setattr(free_ai, "_read", fake_read)
    result = free_ai.chat([{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert result.provider == "https://good.test"


def test_chat_all_providers_fail(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_CHAT_UPSTREAMS", ["https://bad.test"])
    monkeypatch.setattr(free_ai, "_read", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(free_ai.FreeAIError):
        free_ai.chat([{"role": "user", "content": "hi"}])


def test_chat_maps_model_name(monkeypatch):
    seen = {}

    def fake_read(url, *, data=None, **kwargs):
        seen["payload"] = json.loads(data.decode("utf-8"))
        return _completion("ok")

    monkeypatch.setattr(config, "FREE_AI_CHAT_UPSTREAMS", ["https://text.pollinations.ai"])
    monkeypatch.setattr(free_ai, "_read", fake_read)
    free_ai.chat([{"role": "user", "content": "hi"}], model="qwen3.8-27b")
    assert seen["payload"]["model"] == "openai"  # pollinations alias


def test_enhance_prompt_uses_model(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_ENABLED", True)
    monkeypatch.setattr(config, "FREE_AI_CHAT_UPSTREAMS", ["https://a.test"])
    monkeypatch.setattr(free_ai, "_read", lambda *a, **k: _completion("a vivid cat scene"))
    assert free_ai.enhance_prompt("kucing") == "a vivid cat scene"


def test_enhance_prompt_falls_back_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_ENABLED", False)
    assert free_ai.enhance_prompt("kucing berlari") == "kucing berlari"


def test_enhance_prompt_falls_back_on_error(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_ENABLED", True)
    monkeypatch.setattr(config, "FREE_AI_CHAT_UPSTREAMS", ["https://bad.test"])
    monkeypatch.setattr(free_ai, "_read", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert free_ai.enhance_prompt("kucing") == "kucing"


def test_generate_image_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(free_ai, "_read", lambda *a, **k: b"FAKEIMAGEDATA")
    dest = tmp_path / "img.png"
    result = free_ai.generate_image("a cat", str(dest))
    assert dest.read_bytes() == b"FAKEIMAGEDATA"
    assert result.provider


def test_generate_image_failover(monkeypatch, tmp_path):
    def fake_read(url, **kwargs):
        if "a0.dev" in url:
            raise OSError("a0 down")
        return b"POLLINATIONS"

    monkeypatch.setattr(free_ai, "_read", fake_read)
    dest = tmp_path / "img.png"
    result = free_ai.generate_image("a cat", str(dest))
    assert dest.read_bytes() == b"POLLINATIONS"
    assert "pollinations" in result.provider


def test_sniff_ext_from_magic_bytes():
    assert free_ai._sniff_ext(b"\x89PNG\r\n\x1a\nrest") == ".png"
    assert free_ai._sniff_ext(b"\xff\xd8\xff\xe0rest") == ".jpg"
    assert free_ai._sniff_ext(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == ".webp"
    assert free_ai._sniff_ext(b"GIF89a") == ".gif"


def test_generate_image_uses_real_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(free_ai, "_read", lambda *a, **k: b"RIFF\x00\x00\x00\x00WEBPVP8 x")
    result = free_ai.generate_image("a cat", str(tmp_path / "img.png"))
    assert result.path.endswith(".webp")
    assert Path(result.path).read_bytes().startswith(b"RIFF")


def test_image_size_respects_free_tier_cap():
    # The keyless tier rejects long edges above 768.
    for aspect in ("16:9", "9:16", "1:1", "21:9", "4:3"):
        w, h = free_ai.image_size(aspect)
        assert max(w, h) <= config.FREE_AI_IMAGE_MAX_EDGE
        assert w % 16 == 0 and h % 16 == 0


def test_image_size_keeps_aspect_within_rounding():
    w, h = free_ai.image_size("16:9", max_edge=768)
    assert abs((w / h) - (16 / 9)) < 0.08  # ~16px rounding tolerance


def test_realism_prompt_adds_photographic_tags():
    out = free_ai.realism_prompt("a cat on a beach")
    assert out.startswith("a cat on a beach")
    assert "photorealistic" in out and "35mm film" in out


def test_realism_prompt_does_not_duplicate_tags():
    once = free_ai.realism_prompt("a cat")
    twice = free_ai.realism_prompt(once)
    assert twice.lower().count("photorealistic") == 1


def test_generate_image_sends_clamped_size(monkeypatch, tmp_path):
    seen = {}

    def fake_read(url, **kwargs):
        seen["url"] = url
        return b"IMG"

    # Force the pollinations endpoint (a0.dev takes a different URL shape).
    monkeypatch.setattr(free_ai, "_IMAGE_ENDPOINTS", ["https://image.pollinations.ai/prompt"])
    monkeypatch.setattr(free_ai, "_fetch_image_with_retry", fake_read)
    free_ai.generate_image("a cat", str(tmp_path / "i.png"), aspect="1920x1080")
    # unknown aspect -> default 16:9, clamped to the free tier's cap
    w, h = free_ai.image_size("16:9")
    assert f"width={w}&height={h}" in seen["url"]
    assert max(w, h) <= config.FREE_AI_IMAGE_MAX_EDGE


def test_image_fetch_retries_on_402(monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def flaky(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(url, 402, "Payment Required", {}, None)
        return b"IMG"

    monkeypatch.setattr(free_ai, "_read", flaky)
    monkeypatch.setattr(free_ai.time, "sleep", lambda *a, **k: None)
    assert free_ai._fetch_image_with_retry("https://x.test/img") == b"IMG"
    assert calls["n"] == 2


def test_image_fetch_gives_up_after_retries(monkeypatch):
    import urllib.error

    monkeypatch.setattr(free_ai, "_read", lambda url, **k: (_ for _ in ()).throw(
        urllib.error.HTTPError(url, 402, "Payment Required", {}, None)))
    monkeypatch.setattr(free_ai.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(config, "FREE_AI_IMAGE_RETRIES", 1)
    with pytest.raises(urllib.error.HTTPError):
        free_ai._fetch_image_with_retry("https://x.test/img")


def test_402_puts_endpoint_in_cooldown_and_fails_over(monkeypatch, tmp_path):
    import urllib.error

    poll = "https://image.pollinations.ai/prompt"
    a0 = "https://api.a0.dev/assets/image"
    monkeypatch.setattr(free_ai, "_IMAGE_ENDPOINTS", [poll, a0])
    monkeypatch.setattr(free_ai, "_fetch_image_with_retry",
                        lambda url, **k: (_ for _ in ()).throw(
                            urllib.error.HTTPError(url, 402, "Payment Required", {}, None))
                        if poll in url else b"A0IMAGE")
    monkeypatch.setattr(free_ai.time, "sleep", lambda *a, **k: None)

    result = free_ai.generate_image("a cat", str(tmp_path / "i.png"))
    assert result.provider == a0
    assert free_ai._endpoint_available(poll) is False  # now in cooldown


def test_cooldown_endpoint_is_skipped(monkeypatch, tmp_path):
    poll = "https://image.pollinations.ai/prompt"
    a0 = "https://api.a0.dev/assets/image"
    monkeypatch.setattr(free_ai, "_IMAGE_ENDPOINTS", [poll, a0])
    free_ai._throttle_endpoint(poll)
    seen = []

    def fake_read(url, **kwargs):
        seen.append(url)
        return b"IMG"

    monkeypatch.setattr(free_ai, "_fetch_image_with_retry", fake_read)
    result = free_ai.generate_image("a cat", str(tmp_path / "i.png"))
    assert result.provider == a0
    assert all(poll not in u for u in seen)  # throttled endpoint never called


def test_prepare_prompt_enhances_and_fetches_image(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FREE_AI_ENABLED", True)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(free_ai, "enhance_prompt", lambda prompt, model=None: "enhanced english prompt")

    def fake_gen(prompt, dest, **kwargs):
        Path(dest).write_bytes(b"img")
        return free_ai.ImageResult(path=dest, provider="fake")

    monkeypatch.setattr(free_ai, "generate_image", fake_gen)

    prompt, image_path = jobs._prepare_prompt("job_999", "kucing", "text_to_video", True, None)
    assert prompt == "enhanced english prompt"
    assert image_path and Path(image_path).read_bytes() == b"img"


def test_prepare_prompt_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "FREE_AI_ENABLED", False)
    prompt, image_path = jobs._prepare_prompt("job_999", "kucing", "text_to_video", True, None)
    assert prompt == "kucing"
    assert image_path is None
