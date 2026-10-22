"""中文简繁统一（script_normalize / pipeline / prompt）测试。"""

from __future__ import annotations

from pathlib import Path

from voxscribe import prompts
from voxscribe.config import AppSettings
from voxscribe.models import JobOptions, RefineOptions, RefineResult, Segment, Transcript
from voxscribe.pipeline import Pipeline
from voxscribe.script_normalize import (
    count_script_changes,
    convert_script,
    is_chinese_language,
    normalize_script_name,
    script_label,
    script_rule,
)

TRADITIONAL = "我們這個項目的進度問題，繁體字測試。"
SIMPLIFIED = "我们这个项目的进度问题，繁体字测试。"


def test_convert_script_simplified_and_traditional() -> None:
    assert convert_script(TRADITIONAL, "simplified") == SIMPLIFIED
    assert convert_script(SIMPLIFIED, "traditional") == TRADITIONAL
    assert convert_script(TRADITIONAL, "preserve") == TRADITIONAL
    assert convert_script(TRADITIONAL, "bogus") == SIMPLIFIED  # 非法值回退简体


def test_convert_script_keeps_english_and_numbers() -> None:
    text = "Checklist 2026 和 70% 的数字不变，我们只改繁體字。"
    converted = convert_script(text, "simplified")
    assert "Checklist 2026 和 70%" in converted
    assert "繁體" not in converted


def test_script_helpers() -> None:
    assert is_chinese_language("zh") is True
    assert is_chinese_language("zh-CN") is True
    assert is_chinese_language("yue") is True
    assert is_chinese_language("en") is False
    assert normalize_script_name("TRADITIONAL") == "traditional"
    assert normalize_script_name("") == "simplified"
    assert script_label("traditional") == "繁体中文"
    assert count_script_changes("我們兩個", "simplified") == 3
    assert count_script_changes("我们两个", "simplified") == 0


def test_script_rule_only_for_chinese() -> None:
    rule = script_rule("zh", "simplified")
    assert "简体中文" in rule and "不得输出繁体字" in rule
    assert script_rule("en", "simplified") == ""
    assert script_rule("zh", "preserve") == ""
    assert "繁体中文" in script_rule("zh", "traditional")


def test_prompts_include_script_rule() -> None:
    simplified_prompt = prompts.system_prompt("direct", "zh", "simplified")
    assert "不得输出繁体字" in simplified_prompt
    traditional_prompt = prompts.system_prompt("direct", "zh", "traditional")
    assert "不得输出简体字" in traditional_prompt
    english_prompt = prompts.system_prompt("direct", "en", "simplified")
    assert "不得输出繁体字" not in english_prompt
    assert prompts.language_name("zh", "traditional") == "繁体中文"
    assert prompts.language_name("zh", "preserve") == "中文（保持原文简繁）"


class _FakeASR:
    def __init__(self, text: str, language: str = "zh") -> None:
        self.text = text
        self.language = language
        self.loaded = True

    def is_model_available(self) -> bool:
        return True

    def model_size_hint(self) -> str:
        return "1MB"

    def ensure_loaded(self, on_progress=None, cancel=None) -> None:
        return None

    def transcribe(self, audio_path, *, language="zh", on_progress=None, cancel=None):
        return Transcript(
            segments=[Segment(0.0, 1.0, self.text)],
            language=self.language,
            duration=1.0,
        )

    def unload(self) -> None:
        self.loaded = False


class _FakeRefiner:
    def __init__(self, text: str) -> None:
        self.text = text

    def refine(self, transcript, options, *, on_progress=None, cancel=None):
        return RefineResult(text=self.text, chunks=1)


def _make_pipeline(tmp_path: Path, asr: _FakeASR, refiner: _FakeRefiner) -> Pipeline:
    settings = AppSettings(chinese_script="simplified", output_dir=str(tmp_path / "out"))
    settings.set_api_key("sk-test")
    return Pipeline(
        settings,
        asr_factory=lambda _settings: asr,
        refiner_factory=lambda _settings, _key: refiner,
    )


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 64)
    return path


def test_pipeline_normalizes_raw_transcript_before_display(tmp_path: Path) -> None:
    asr = _FakeASR(TRADITIONAL)
    refiner = _FakeRefiner(SIMPLIFIED)
    pipe = _make_pipeline(tmp_path, asr, refiner)
    seen = []
    result = pipe.transcribe_and_refine(
        _audio(tmp_path),
        JobOptions(refine=True, style="direct", language="auto", reasoning_effort="none"),
        lambda event: None,
        on_transcript=lambda transcript: seen.append(transcript.text),
    )
    assert result.transcript.text == SIMPLIFIED
    assert seen == [SIMPLIFIED]
    assert result.final_text == SIMPLIFIED


def test_pipeline_normalizes_refined_text(tmp_path: Path) -> None:
    asr = _FakeASR(SIMPLIFIED)
    refiner = _FakeRefiner(TRADITIONAL)
    pipe = _make_pipeline(tmp_path, asr, refiner)
    result = pipe.transcribe_and_refine(
        _audio(tmp_path),
        JobOptions(refine=True, style="direct", language="zh"),
        lambda event: None,
    )
    assert result.transcript.text == SIMPLIFIED
    assert result.refined is not None
    assert result.refined.text == SIMPLIFIED


def test_pipeline_respects_traditional_setting(tmp_path: Path) -> None:
    asr = _FakeASR(SIMPLIFIED)
    refiner = _FakeRefiner(SIMPLIFIED)
    settings = AppSettings(chinese_script="traditional", output_dir=str(tmp_path / "out"))
    settings.set_api_key("sk-test")
    pipe = Pipeline(
        settings,
        asr_factory=lambda _settings: asr,
        refiner_factory=lambda _settings, _key: refiner,
    )
    result = pipe.transcribe_and_refine(
        _audio(tmp_path), JobOptions(refine=True, style="direct", language="zh"), lambda event: None
    )
    assert result.transcript.text == TRADITIONAL
    assert result.refined is not None
    assert result.refined.text == TRADITIONAL


def test_pipeline_preserve_setting_leaves_text(tmp_path: Path) -> None:
    asr = _FakeASR(TRADITIONAL)
    refiner = _FakeRefiner(TRADITIONAL)
    settings = AppSettings(chinese_script="preserve", output_dir=str(tmp_path / "out"))
    settings.set_api_key("sk-test")
    pipe = Pipeline(
        settings,
        asr_factory=lambda _settings: asr,
        refiner_factory=lambda _settings, _key: refiner,
    )
    result = pipe.transcribe_and_refine(
        _audio(tmp_path), JobOptions(refine=True, style="direct", language="zh"), lambda event: None
    )
    assert result.transcript.text == TRADITIONAL
    assert result.refined is not None
    assert result.refined.text == TRADITIONAL
