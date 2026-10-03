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
