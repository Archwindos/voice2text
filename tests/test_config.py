"""配置模块单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxscribe import secure
from voxscribe.config import AppSettings


def test_defaults() -> None:
    settings = AppSettings()
    assert settings.asr_model == "medium"
    assert settings.api_model == "deepseek-chat"
    assert settings.api_base_url.startswith("https://")
    assert 1 <= settings.effective_threads() <= 4
    assert settings.idle_unload_seconds() == 0  # 默认常驻，保证口述响应速度
    assert settings.close_to_tray is True
    assert settings.preload_model is True
    assert settings.effective_device() == "cuda"
    assert settings.effective_compute_type() == "float16"


def test_device_and_compute_type() -> None:
    settings = AppSettings(asr_device="cpu", compute_type="auto")
    assert settings.effective_device() == "cpu"
    assert settings.effective_compute_type() == "int8"

    settings.asr_device = "cuda"
    settings.compute_type = "auto"
    assert settings.effective_compute_type() == "float16"

    settings.compute_type = "int8_float16"
    assert settings.effective_compute_type() == "int8_float16"

    settings.asr_device = "gpu"
    assert settings.effective_device() == "cuda"
    settings.asr_device = "cpu-fallback"
    assert settings.effective_device() == "cpu"


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    settings = AppSettings(
        asr_model="base",
        cpu_threads=2,
        export_formats=["txt", "docx"],
        glossary="VoxScribe、DeepSeek",
        output_dir=str(tmp_path / "out"),
    )
    saved = settings.save(path)
    assert saved == path and path.exists()

    loaded = AppSettings.load(path)
    assert loaded.asr_model == "base"
    assert loaded.cpu_threads == 2
    assert loaded.export_formats == ["txt", "docx"]
    assert loaded.glossary == "VoxScribe、DeepSeek"


def test_unknown_fields_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"asr_model": "tiny", "future_field": 123}), encoding="utf-8"
    )
    assert AppSettings.load(path).asr_model == "tiny"


def test_corrupt_config_is_backed_up(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{ this is not json", encoding="utf-8")

    settings = AppSettings.load(path)
    assert settings.asr_model == "medium"
    backup = path.with_suffix(path.suffix + ".broken")
    assert backup.exists()


def test_missing_config_returns_defaults(tmp_path: Path) -> None:
    settings = AppSettings.load(tmp_path / "nope.json")
    assert settings.asr_model == "medium"


def test_api_key_roundtrip_and_storage() -> None:
    settings = AppSettings()
    assert settings.has_api_key is False

    encrypted = settings.set_api_key("sk-test-1234567890")
    assert settings.get_api_key() == "sk-test-1234567890"
    assert settings.has_api_key is True
    if secure.is_available():
        assert encrypted is True
        assert settings.api_key_enc.startswith("dpapi:")
        assert "sk-test" not in settings.api_key_enc
    else:  # 非 Windows / DPAPI 不可用时降级明文
        assert encrypted is False
        assert settings.api_key_enc == "plain:sk-test-1234567890"

    settings.set_api_key("")
    assert settings.get_api_key() == ""
    assert settings.api_key_enc == ""


def test_api_key_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXSCRIBE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = AppSettings()
    monkeypatch.setenv("VOXSCRIBE_API_KEY", "sk-from-env")
    assert settings.get_api_key() == "sk-from-env"


def test_paths_and_threads(tmp_path: Path) -> None:
    settings = AppSettings(
        model_dir=str(tmp_path / "models"), output_dir=str(tmp_path / "outputs")
    )
    assert settings.resolve_model_dir() == tmp_path / "models"
    out = settings.resolve_output_dir()
    assert out == tmp_path / "outputs" and out.exists()

    settings.cpu_threads = 2
    assert settings.effective_threads() == 2
    settings.cpu_threads = 0
    assert settings.effective_threads() <= 4

    settings.idle_unload_minutes = 0
    assert settings.idle_unload_seconds() == 0
    settings.idle_unload_minutes = 7
    assert settings.idle_unload_seconds() == 420
