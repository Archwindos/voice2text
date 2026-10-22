"""词语表在 pipeline 中的本地替换与 prompt 合并测试。"""

from __future__ import annotations

from pathlib import Path

from voxscribe import glossary
from voxscribe.config import AppSettings
from voxscribe.models import JobOptions, RefineResult, Segment, Transcript
from voxscribe.pipeline import Pipeline


class _FakeASR:
    loaded = True

    def __init__(self, text: str) -> None:
        self.text = text

    def is_model_available(self) -> bool:
        return True

    def model_size_hint(self) -> str:
        return "1MB"

    def ensure_loaded(self, on_progress=None, cancel=None) -> None:
        return None

    def transcribe(self, audio_path, *, language="zh", on_progress=None, cancel=None):
        if on_progress:
            on_progress(1.0, "完成")
        return Transcript(
            segments=[Segment(0.0, 1.0, self.text)], language="zh", duration=1.0
        )

    def unload(self) -> None:
        self.loaded = False


class _FakeRefiner:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.last_options = None

    def refine(self, transcript, options, *, on_progress=None, cancel=None):
        self.last_options = options
        if on_progress:
            on_progress(1.0, "完成")
        return RefineResult(text=self.text or transcript.text, chunks=1)


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 64)
    return path


def _make_pipeline(tmp_path: Path, asr_text: str, refiner_text: str = "", **kwargs):
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    settings.set_api_key("sk-test")
    refiner = _FakeRefiner(refiner_text)
    pipe = Pipeline(
        settings,
        asr_factory=lambda _settings: _FakeASR(asr_text),
        refiner_factory=lambda _settings, _key: refiner,
        **kwargs,
    )
    return pipe, refiner


def test_learned_glossary_applied_locally_and_in_prompt(tmp_path: Path) -> None:
    entry = glossary.GlossaryEntry(
        wrong="阴轨", correct="音轨", kind="replace", source="learned", enabled=True
    )
    pipe, refiner = _make_pipeline(
        tmp_path, "我们打开阴轨看看", "我们打开阴轨看看", glossary_entries=[entry]
    )
    seen = []
    result = pipe.transcribe_and_refine(
        _audio(tmp_path),
        JobOptions(refine=True, style="direct", language="zh"),
        lambda event: None,
        on_transcript=lambda transcript: seen.append(transcript.text),
    )
    assert result.transcript.text == "我们打开音轨看看"
    assert seen == ["我们打开音轨看看"]
    assert result.refined is not None and result.refined.text == "我们打开音轨看看"
    assert refiner.last_options is not None
    assert "音轨：阴轨" in refiner.last_options.glossary


def test_manual_glossary_applied_and_prompt_keeps_raw_text(tmp_path: Path) -> None:
    settings_text = "音轨：阴轨\nVoxScribe、DeepSeek"
    pipe, refiner = _make_pipeline(tmp_path, "我们打开阴轨看看", "我们打开阴轨看看")
    pipe.settings.glossary = settings_text
    result = pipe.transcribe_and_refine(
        _audio(tmp_path),
        JobOptions(refine=True, style="direct", language="zh"),
        lambda event: None,
    )
    assert result.transcript.text == "我们打开音轨看看"
    assert result.refined is not None and result.refined.text == "我们打开音轨看看"
    assert refiner.last_options is not None
    assert "音轨：阴轨" in refiner.last_options.glossary
    # 手写术语表原文保留（即使不是「正确：错误」格式）
    assert "VoxScribe、DeepSeek" in refiner.last_options.glossary


def test_disabled_glossary_entry_not_applied(tmp_path: Path) -> None:
    entry = glossary.GlossaryEntry(
        wrong="阴轨", correct="音轨", source="learned", enabled=False
    )
    pipe, refiner = _make_pipeline(
        tmp_path, "我们打开阴轨看看", "我们打开阴轨看看", glossary_entries=[entry]
    )
    result = pipe.transcribe_and_refine(
        _audio(tmp_path),
        JobOptions(refine=True, style="direct", language="zh"),
        lambda event: None,
    )
    assert result.transcript.text == "我们打开阴轨看看"
    assert result.refined is not None and result.refined.text == "我们打开阴轨看看"
    assert refiner.last_options is not None
    assert "音轨：阴轨" not in refiner.last_options.glossary


def test_set_glossary_entries_updates_pipeline(tmp_path: Path) -> None:
    pipe, refiner = _make_pipeline(tmp_path, "我们打开阴轨看看", "我们打开阴轨看看")
    pipe.set_glossary_entries(
        [glossary.GlossaryEntry(wrong="阴轨", correct="音轨", source="learned")]
    )
    result = pipe.transcribe_and_refine(
        _audio(tmp_path),
        JobOptions(refine=True, style="direct", language="zh"),
        lambda event: None,
    )
    assert result.transcript.text == "我们打开音轨看看"
