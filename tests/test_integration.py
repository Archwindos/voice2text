"""端到端集成测试。

* 常规用例：真实 ``DeepSeekRefiner`` HTTP 调用（本地 mock 服务）+ 真实导出；
* 门控用例：真实 faster-whisper tiny 模型转写中文 TTS 音频 + mock DeepSeek + 导出。
  需要 ``VOXSCRIBE_RUN_ASR=1`` 且本机已缓存/可下载 tiny 模型。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.mock_api import MockAPIServer, fixed_handler
from voxscribe.config import AppSettings
from voxscribe.models import JobOptions, Segment, Transcript
from voxscribe.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "tts_zh.wav"
RUN_ASR = os.environ.get("VOXSCRIBE_RUN_ASR") == "1"


class _FakeASR:
    """只返回固定中文转写的假 ASR（用于不依赖模型的集成测试）。"""

    def __init__(self, text: str = "嗯，那个，我们今天讨论一下项目进度。") -> None:
        self.text = text
        self.unloaded = False

    @property
    def loaded(self) -> bool:
        return not self.unloaded

    def is_model_available(self) -> bool:
        return True

    def model_size_hint(self) -> str:
        return "0MB"

    def transcribe(self, audio_path, *, language="zh", on_progress=None, cancel=None):
        if on_progress:
            on_progress(0.5, "转写中…")
            on_progress(1.0, "转写完成")
        return Transcript(
            segments=[Segment(0.0, 2.0, self.text)], language="zh", duration=2.0
        )

    def unload(self) -> None:
        self.unloaded = True


def _settings(tmp_path: Path) -> AppSettings:
    settings = AppSettings(output_dir=str(tmp_path / "out"))
    settings.model_dir = str(ROOT / "data" / "models")
    return settings


def test_pipeline_with_real_refiner_and_mock_api(tmp_path: Path) -> None:
    """pipeline -> DeepSeekRefiner(真实 HTTP) -> export 全链路（ASR 用假数据）。"""
    with MockAPIServer(fixed_handler("整理后的书面文字。")) as server:
        settings = _settings(tmp_path)
        settings.set_api_key("sk-integration-test")
        settings.api_base_url = server.url
        pipeline = Pipeline(settings, asr_factory=lambda _settings: _FakeASR())
        events = []

        result = pipeline.run(
            FIXTURE,
            JobOptions(
                refine=True,
                style="oral",
                export_formats=["txt", "md"],
                output_dir=str(tmp_path / "out"),
            ),
            events.append,
        )

        assert result.final_text == "整理后的书面文字。"
        assert result.refined is not None
        assert result.refined.usage.total_tokens > 0
        assert server.request_count == 1
        request = server.last_request
        assert request is not None
        assert request.headers.get("authorization", "").lower() == (
            "bearer sk-integration-test"
        )
        assert request.path.endswith("/chat/completions")
        # 正文确实传给了模型
        assert "项目进度" in request.body

        suffixes = {Path(path).suffix for path in result.output_files}
        assert {".txt", ".md"} <= suffixes
        assert all(Path(path).stat().st_size > 0 for path in result.output_files)
        assert any("原始转写" in path for path in result.output_files)
        assert [event.stage for event in events][-1] == "done"


def test_pipeline_missing_api_key_is_friendly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VOXSCRIBE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = _settings(tmp_path)
    pipeline = Pipeline(settings, asr_factory=lambda _settings: _FakeASR())
    from voxscribe import errors

    with pytest.raises(errors.MissingAPIKeyError) as excinfo:
        pipeline.run(FIXTURE, JobOptions(refine=True), lambda event: None)
    assert "API Key" in str(excinfo.value)
    assert excinfo.value.user_hint


@pytest.mark.skipif(
    not RUN_ASR, reason="需要 VOXSCRIBE_RUN_ASR=1 且本机可使用 tiny 模型"
)
@pytest.mark.skipif(not FIXTURE.exists(), reason="缺少中文 TTS 测试音频")
def test_full_e2e_real_asr_tiny(tmp_path: Path) -> None:
    """真实 tiny 模型转写 + mock DeepSeek 整理 + 四格式导出。"""
    with MockAPIServer() as server:  # 默认 echo：把正文原样返回
        settings = _settings(tmp_path)
        settings.set_api_key("sk-e2e-test")
        settings.api_base_url = server.url
        settings.asr_model = "tiny"
        settings.cpu_threads = 4
        settings.idle_unload_minutes = 0
        pipeline = Pipeline(settings)
        events = []
        started = time.monotonic()

        result = pipeline.run(
            FIXTURE,
            JobOptions(
                refine=True,
                style="oral",
                export_formats=["txt", "md", "docx", "srt"],
                output_dir=str(tmp_path / "out"),
            ),
            events.append,
        )
        elapsed = time.monotonic() - started

        raw_text = result.transcript.text
        assert any(
            keyword in raw_text for keyword in ("项目", "进度", "前端", "后端", "两周")
        ), f"真实转写结果异常：{raw_text!r}"
        assert result.refined is not None and result.refined.text.strip()
        suffixes = {Path(path).suffix for path in result.output_files}
        assert {".txt", ".md", ".docx", ".srt"} <= suffixes
        assert any("原始转写" in path for path in result.output_files)
        assert all(Path(path).stat().st_size > 0 for path in result.output_files)

        srt_path = next(
            Path(path) for path in result.output_files if path.endswith(".srt")
        )
        srt_text = srt_path.read_text(encoding="utf-8")
        assert "-->" in srt_text and "00:00:" in srt_text

        docx_path = next(
            Path(path) for path in result.output_files if path.endswith(".docx")
        )
        from docx import Document

        document = Document(str(docx_path))
        docx_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        assert "项目" in docx_text or "进度" in docx_text

        stages = [event.stage for event in events]
        assert "asr" in stages and "refine" in stages and stages[-1] == "done"
        assert result.transcript.duration > 5
        assert elapsed < 300, f"端到端耗时异常：{elapsed:.1f}s"
        pipeline.unload()


@pytest.mark.skipif(
    not RUN_ASR, reason="需要 VOXSCRIBE_RUN_ASR=1 且本机可使用 tiny 模型"
)
@pytest.mark.skipif(not FIXTURE.exists(), reason="缺少中文 TTS 测试音频")
def test_cli_headless_real_asr(tmp_path: Path) -> None:
    """命令行模式：真实转写 + 真实导出（不启用整理，无需 API Key）。"""
    config = {
        "asr_model": "tiny",
        "model_dir": str(ROOT / "data" / "models"),
        "output_dir": str(tmp_path / "cli_out"),
        "export_formats": ["txt"],
        "language": "zh",
        "idle_unload_minutes": 0,
        "cpu_threads": 4,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "VOXSCRIBE_CONFIG": str(config_path),
            "VOXSCRIBE_DATA_DIR": str(tmp_path / "data"),
            "PYTHONIOENCODING": "utf-8",
        }
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "voxscribe",
            "-i",
            str(FIXTURE),
            "--model",
            "tiny",
            "--no-refine",
            "--formats",
            "txt",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    outputs = list((tmp_path / "cli_out").glob("*.txt"))
    assert outputs, f"没有生成输出文件：{proc.stdout}\n{proc.stderr}"
    content = "\n".join(path.read_text(encoding="utf-8") for path in outputs)
    assert any(keyword in content for keyword in ("项目", "进度", "前端", "后端"))
