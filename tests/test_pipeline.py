"""Pipeline 流水线测试（用假 ASR/Refiner 隔离外部依赖）。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from voxscribe import errors, pipeline as pipeline_module
from voxscribe.config import AppSettings
from voxscribe.models import (
    JobEvent,
    JobOptions,
    RefineResult,
    RefineUsage,
    Segment,
    Transcript,
)
from voxscribe.pipeline import Pipeline


class FakeASR:
    created = 0

    def __init__(self, *, fail: Optional[Exception] = None) -> None:
        FakeASR.created += 1
        self.fail = fail
        self.unloaded = False
        self.last_language: Optional[str] = None
        self.ensured_loaded = 0

    @property
    def loaded(self) -> bool:
        return not self.unloaded

    def is_model_available(self) -> bool:
        return True

    def model_size_hint(self) -> str:
        return "1MB"

    def ensure_loaded(self, on_progress=None, cancel=None) -> None:
        self.ensured_loaded += 1
        if self.fail:
            raise self.fail

    def transcribe(self, audio_path, *, language="zh", on_progress=None, cancel=None):
        if on_progress:
            on_progress(0.4, "转写中…")
            on_progress(1.0, "转写完成")
        if self.fail:
            raise self.fail
        self.last_language = language
        return Transcript(
            segments=[Segment(0.0, 2.0, "嗯，那个，我们讨论一下项目进度。")],
            language="zh",
            duration=2.0,
        )

    def unload(self) -> None:
        self.unloaded = True


class FakeRefiner:
    created = 0

    def __init__(self, fail: Optional[Exception] = None) -> None:
        FakeRefiner.created += 1
        self.fail = fail
        self.last_options = None

    def refine(self, transcript, options, *, on_progress=None, cancel=None) -> RefineResult:
        self.last_options = options
        if on_progress:
            on_progress(1.0, "整理完成")
        if self.fail:
            raise self.fail
        return RefineResult(
            text="我们讨论一下项目进度。",
            chunks=1,
            usage=RefineUsage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
            elapsed=0.1,
        )


def _settings(tmp_path: Path, *, with_key: bool = True) -> AppSettings:
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    if with_key:
        settings.set_api_key("sk-test")
    return settings


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "录音.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 64)
    return path


@pytest.fixture
def fake_export(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def _export(result, *, out_dir, base_name, formats, title=None, include_raw=True):
        calls.append(
            dict(out_dir=Path(out_dir), base_name=base_name, formats=list(formats))
        )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{base_name}.txt"
        target.write_text(result.final_text, encoding="utf-8")
        return [target]

    monkeypatch.setattr(pipeline_module, "export_result", _export)
    return calls


def _make_pipeline(
    settings: AppSettings, asr: FakeASR, refiner: Optional[FakeRefiner] = None
) -> Pipeline:
    return Pipeline(
        settings,
        asr_factory=lambda _settings: asr,
        refiner_factory=lambda _settings, _key: refiner or FakeRefiner(),
    )


def _collect() -> tuple[List[JobEvent], callable]:
    events: List[JobEvent] = []

    def emit(event: JobEvent) -> None:
        events.append(event)

    return events, emit


def test_run_without_refine(tmp_path: Path, fake_export) -> None:
    settings = _settings(tmp_path, with_key=False)
    asr = FakeASR()
    pipe = _make_pipeline(settings, asr)
    job = JobOptions(refine=False, export_formats=["txt"], output_dir=str(tmp_path / "out"))
    events, emit = _collect()

    result = pipe.run(_audio(tmp_path), job, emit)

    stages = [event.stage for event in events]
    assert stages[0] == "prepare"
    assert "asr" in stages and "export" in stages
    assert stages[-1] == "done"
    assert result.refined is None
    assert result.final_text.startswith("嗯，那个")
    assert result.output_files and Path(result.output_files[0]).exists()
    assert fake_export and fake_export[0]["base_name"] == "录音"
    assert pipe.busy is False


def test_run_with_refine(tmp_path: Path, fake_export) -> None:
    settings = _settings(tmp_path)
    asr = FakeASR()
    refiner = FakeRefiner()
    pipe = _make_pipeline(settings, asr, refiner)
    job = JobOptions(refine=True, style="meeting", export_formats=["txt"])
    events, emit = _collect()

    result = pipe.run(_audio(tmp_path), job, emit)

    stages = [event.stage for event in events]
    assert stages.index("asr") < stages.index("refine") < stages.index("export")
    assert result.final_text == "我们讨论一下项目进度。"
    assert result.refined is not None
    assert result.refined.usage.total_tokens == 18
    assert refiner.last_options.style == "meeting"


def test_transcribe_and_refine_without_export(tmp_path: Path, monkeypatch) -> None:
    """快速语音输入入口：只转写+整理，不写文件；原始转写可先回调。"""
    settings = _settings(tmp_path)
    asr = FakeASR()
    refiner = FakeRefiner()
    pipe = _make_pipeline(settings, asr, refiner)
    export_calls = []
    monkeypatch.setattr(
        pipeline_module,
        "export_result",
        lambda *args, **kwargs: export_calls.append(1) or [],
    )
    events, emit = _collect()
    callback_texts = []

    result = pipe.transcribe_and_refine(
        _audio(tmp_path),
        JobOptions(refine=True, style="proofread"),
        emit,
        on_transcript=lambda transcript: callback_texts.append(transcript.text),
    )

    assert export_calls == []
    assert result.output_files == []
    assert result.final_text == "我们讨论一下项目进度。"
    assert result.elapsed >= 0
    assert events[-1].stage == "done"
    assert "转换完成" in events[-1].message
    assert callback_texts == ["嗯，那个，我们讨论一下项目进度。"]
    assert pipe.busy is False


def test_on_transcript_callback_errors_are_ignored(tmp_path: Path) -> None:
    """界面的显示回调出错不应中断转换。"""
    settings = _settings(tmp_path)
    pipe = _make_pipeline(settings, FakeASR(), FakeRefiner())
    events, emit = _collect()

    def broken(_transcript):
        raise RuntimeError("ui callback boom")

    result = pipe.transcribe_and_refine(
        _audio(tmp_path),
        JobOptions(refine=True, style="proofread"),
        emit,
        on_transcript=broken,
    )
    assert result.final_text == "我们讨论一下项目进度。"


def test_missing_api_key_raises(tmp_path: Path, fake_export, monkeypatch) -> None:
    monkeypatch.delenv("VOXSCRIBE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = _settings(tmp_path, with_key=False)
    pipe = _make_pipeline(settings, FakeASR())
    events, emit = _collect()

    with pytest.raises(errors.MissingAPIKeyError) as excinfo:
        pipe.run(_audio(tmp_path), JobOptions(refine=True), emit)

    assert "API Key" in str(excinfo.value)
    assert excinfo.value.user_hint


def test_missing_audio_raises(tmp_path: Path, fake_export) -> None:
    pipe = _make_pipeline(_settings(tmp_path), FakeASR())
    events, emit = _collect()
    with pytest.raises(errors.AudioError):
        pipe.run(tmp_path / "不存在.mp3", JobOptions(refine=False), emit)


def test_empty_audio_raises(tmp_path: Path, fake_export) -> None:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    pipe = _make_pipeline(_settings(tmp_path), FakeASR())
    events, emit = _collect()
    with pytest.raises(errors.AudioError):
        pipe.run(empty, JobOptions(refine=False), emit)


def test_cancel_before_start(tmp_path: Path, fake_export) -> None:
    pipe = _make_pipeline(_settings(tmp_path), FakeASR())
    events, emit = _collect()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(errors.CancelledError):
        pipe.run(_audio(tmp_path), JobOptions(refine=False), emit, cancel)
    assert pipe.busy is False


def test_asr_error_propagates(tmp_path: Path, fake_export) -> None:
    pipe = _make_pipeline(_settings(tmp_path), FakeASR(fail=errors.ASRError("模型炸了")))
    events, emit = _collect()
    with pytest.raises(errors.ASRError):
        pipe.run(_audio(tmp_path), JobOptions(refine=False), emit)
    assert pipe.busy is False


def test_empty_transcript_raises(tmp_path: Path, fake_export) -> None:
    class EmptyASR(FakeASR):
        def transcribe(self, audio_path, *, language="zh", on_progress=None, cancel=None):
            return Transcript(segments=[], language="zh", duration=1.0)

    pipe = _make_pipeline(_settings(tmp_path), EmptyASR())
    events, emit = _collect()
    with pytest.raises(errors.ASRError) as excinfo:
        pipe.run(_audio(tmp_path), JobOptions(refine=False), emit)
    assert "没有识别到有效语音" in str(excinfo.value)


def test_idle_unload(tmp_path: Path, fake_export) -> None:
    settings = _settings(tmp_path)
    settings.idle_unload_minutes = 5
    asr = FakeASR()
    pipe = _make_pipeline(settings, asr)
    events, emit = _collect()
    pipe.run(_audio(tmp_path), JobOptions(refine=False), emit)
    assert asr.unloaded is False

    pipe._last_activity = time.monotonic() - 3600  # 模拟空闲一小时
    assert pipe.maybe_unload_idle() is True
    assert asr.unloaded is True
    assert pipe.maybe_unload_idle() is False


def test_warm_up_loads_model_and_reports_failure(tmp_path: Path) -> None:
    """启动预热：成功返回 True；失败只返回 False，不抛异常。"""
    settings = _settings(tmp_path)
    asr = FakeASR()
    pipe = _make_pipeline(settings, asr)
    assert pipe.warm_up() is True
    assert asr.ensured_loaded == 1

    broken = FakeASR(fail=errors.ASRError("GPU 不可用"))
    pipe2 = _make_pipeline(settings, broken)
    assert pipe2.warm_up() is False


def test_apply_settings_reloads_asr(tmp_path: Path, fake_export) -> None:
    settings = _settings(tmp_path)
    asr = FakeASR()
    factory_calls: List[FakeASR] = []

    def asr_factory(_settings):
        factory_calls.append(asr)
        return asr

    pipe = Pipeline(
        settings,
        asr_factory=asr_factory,
        refiner_factory=lambda _settings, _key: FakeRefiner(),
    )
    events, emit = _collect()
    pipe.run(_audio(tmp_path), JobOptions(refine=False), emit)
    assert len(factory_calls) == 1

    new_settings = _settings(tmp_path)
    new_settings.asr_model = "base"
    pipe.apply_settings(new_settings)
    assert asr.unloaded is True
    assert pipe._asr is None

    pipe.run(_audio(tmp_path), JobOptions(refine=False), emit)
    assert len(factory_calls) == 2


def test_quick_mode_reasoning_effort_passed_to_refiner(tmp_path: Path) -> None:
    """语音输入三模式：style 与 reasoning_effort 原样传给 refiner。"""
    settings = _settings(tmp_path)
    asr = FakeASR()
    refiner = FakeRefiner()
    pipe = _make_pipeline(settings, asr, refiner)
    events, emit = _collect()
    cases = (
        ("direct", "none"),
        ("person", "low"),
        ("ai", "high"),
    )
    for style, effort in cases:
        refiner.last_options = None
        pipe.transcribe_and_refine(
            _audio(tmp_path),
            JobOptions(refine=True, style=style, reasoning_effort=effort),
            emit,
        )
        assert refiner.last_options is not None
        assert refiner.last_options.style == style
        assert refiner.last_options.reasoning_effort == effort
