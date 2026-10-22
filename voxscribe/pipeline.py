"""任务流水线：语音转写 -> DeepSeek 整理 ->（可选）导出文件。

设计要点：
* 所有耗时操作都在调用方（GUI 工作线程）中同步执行，本模块不创建线程；
* 通过 ``emit(JobEvent)`` 回报进度，由界面线程换算成进度条；
* 取消使用 ``threading.Event``，在阶段边界与模块内部检查；
* ASR 模型懒加载并复用；空闲时可调用 :meth:`maybe_unload_idle`
  主动卸载，保证「轻量」；
* :meth:`Pipeline.run` 用于文件转写（转写 + 整理 + 导出）；
  :meth:`Pipeline.transcribe_and_refine` 用于快速语音输入（不写文件）。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import errors
from .asr import WhisperASR
from .config import AppSettings
from .export import export_result
from .models import JobEvent, JobOptions, JobResult, RefineOptions, RefineResult, Transcript
from .refine import DeepSeekRefiner
from .util import human_duration, sanitize_filename

log = logging.getLogger(__name__)

Emit = Callable[[JobEvent], None]


def _default_asr_factory(settings: AppSettings) -> WhisperASR:
    return WhisperASR(
        model_size=settings.asr_model,
        device=settings.effective_device(),
        compute_type=settings.effective_compute_type(),
        cpu_threads=settings.effective_threads(),
        model_dir=str(settings.resolve_model_dir()),
        beam_size=settings.beam_size,
        vad_filter=settings.vad_filter,
    )


def _default_refiner_factory(settings: AppSettings, api_key: str) -> DeepSeekRefiner:
    return DeepSeekRefiner(
        api_key=api_key,
        base_url=settings.api_base_url,
        model=settings.api_model,
        timeout=settings.api_timeout,
        max_retries=settings.api_max_retries,
        reasoning_effort=settings.api_reasoning_effort,
    )


class Pipeline:
    """串联各模块的任务执行器（一个实例可复用多次任务）。"""

    def __init__(
        self,
        settings: AppSettings,
        *,
        asr_factory: Optional[Callable[[AppSettings], WhisperASR]] = None,
        refiner_factory: Optional[Callable[[AppSettings, str], DeepSeekRefiner]] = None,
    ) -> None:
        self.settings = settings
        self._asr_factory = asr_factory or _default_asr_factory
        self._refiner_factory = refiner_factory or _default_refiner_factory
        self._asr: Optional[WhisperASR] = None
        self._asr_signature: Optional[tuple] = None
        self._refiner: Optional[DeepSeekRefiner] = None
        self._refiner_signature: Optional[tuple] = None
        self._last_activity: float = time.monotonic()
        self._busy = False
        self._guard = threading.RLock()

    # ------------------------------------------------------------------
    # 设置
    # ------------------------------------------------------------------
    def apply_settings(self, settings: AppSettings) -> None:
        """界面「保存设置」后调用；会按需丢弃旧的模型/客户端缓存。"""
        with self._guard:
            old_signature = self._asr_signature
            self.settings = settings
            if self._asr is not None and old_signature != self._asr_signature_for(settings):
                self._release_asr()
            self._refiner = None
            self._refiner_signature = None

    def _asr_signature_for(self, settings: AppSettings) -> tuple:
        return (
            settings.asr_model,
            settings.compute_type,
            settings.effective_threads(),
            settings.beam_size,
            settings.vad_filter,
            str(settings.resolve_model_dir()),
        )

    def _get_asr(self) -> WhisperASR:
        signature = self._asr_signature_for(self.settings)
        with self._guard:
            if self._asr is None or self._asr_signature != signature:
                self._release_asr()
                self._asr = self._asr_factory(self.settings)
                self._asr_signature = signature
            return self._asr

    def _get_refiner(self) -> DeepSeekRefiner:
        api_key = self.settings.get_api_key()
        if not api_key:
            raise errors.MissingAPIKeyError(
                "尚未配置 DeepSeek API Key，无法进行文字整理",
                hint="请在「设置」中填写 API Key（platform.deepseek.com 获取），"
                "或关闭「AI 整理」只做语音转写。",
            )
        signature = (
            api_key,
            self.settings.api_base_url,
            self.settings.api_model,
            self.settings.api_timeout,
            self.settings.api_max_retries,
        )
        with self._guard:
            if self._refiner is None or self._refiner_signature != signature:
                self._refiner = self._refiner_factory(self.settings, api_key)
                self._refiner_signature = signature
            return self._refiner

    def _release_asr(self) -> None:
        if self._asr is not None:
            try:
                self._asr.unload()
            except Exception:  # 卸载失败不影响主流程
                log.warning("卸载语音模型失败", exc_info=True)
        self._asr = None
        self._asr_signature = None

    # ------------------------------------------------------------------
    # 校验与阶段执行
    # ------------------------------------------------------------------
    def _prepare_audio(
        self, audio_path: str | Path, emit: Emit, cancel: threading.Event
    ) -> Path:
        audio = Path(audio_path)
        emit(
            JobEvent(
                stage="prepare",
                progress=0.0,
                message=f"检查文件：{audio.name}",
                audio_path=str(audio),
            )
        )
        if not audio.exists():
            raise errors.AudioError(f"找不到音频文件：{audio}")
        if audio.is_dir():
            raise errors.AudioError(f"这是一个文件夹，不是音频文件：{audio}")
        try:
            if audio.stat().st_size <= 0:
                raise errors.AudioError(f"音频文件是空的：{audio.name}")
        except OSError as exc:
            raise errors.AudioError(f"无法读取音频文件：{audio.name}（{exc}）") from exc
        if cancel.is_set():
            raise errors.CancelledError("任务已取消")
        return audio

    def _transcribe_and_refine(
        self,
        audio: Path,
        job: JobOptions,
        emit: Emit,
        cancel: threading.Event,
        on_transcript: Optional[Callable[[Transcript], None]] = None,
    ) -> JobResult:
        """阶段 1+2：本地转写 + DeepSeek 整理，返回未导出的结果。"""
        # ---------- 1. 语音转写 ----------
        asr = self._get_asr()
        emit(
            JobEvent(
                stage="asr",
                progress=0.0,
                message=f"语音转写中（模型 {self.settings.asr_model}，"
                f"{self.settings.effective_threads()} 线程）…",
                audio_path=str(audio),
            )
        )

        def on_asr_progress(ratio: float, message: str) -> None:
            if cancel.is_set():
                raise errors.CancelledError("任务已取消")
            emit(
                JobEvent(
                    stage="asr",
                    progress=max(0.0, min(1.0, float(ratio))),
                    message=message or "语音转写中…",
                    audio_path=str(audio),
                )
            )

        transcript = asr.transcribe(
            audio,
            language=None if job.language in (None, "", "auto") else job.language,
            on_progress=on_asr_progress,
            cancel=cancel,
        )
        if transcript.is_empty:
            raise errors.ASRError(
                "没有识别到有效语音内容",
                hint="请确认音频中有清晰的人声，或换用更大的识别模型（设置 → 识别模型）。",
            )
        if cancel.is_set():
            raise errors.CancelledError("任务已取消")

        # 先把「原始转写」交给界面，让用户更早看到文字（随后再用整理稿替换）
        if on_transcript is not None:
            try:
                on_transcript(transcript)
            except Exception:  # 回调异常不应中断主流程
                log.warning("on_transcript 回调执行失败", exc_info=True)

        # ---------- 2. DeepSeek 整理 ----------
        refined: Optional[RefineResult] = None
        if job.refine:
            emit(
                JobEvent(
                    stage="refine",
                    progress=0.0,
                    message="DeepSeek 正在整理文字…",
                    audio_path=str(audio),
                )
            )
            options = RefineOptions(
                enabled=True,
                style=job.style or self.settings.default_style,
                language=(
                    job.language
                    if job.language not in (None, "", "auto")
                    else (transcript.language or "zh")
                ),
                glossary=self.settings.glossary,
                reasoning_effort=job.reasoning_effort or "",
            )

            def on_refine_progress(ratio: float, message: str) -> None:
                if cancel.is_set():
                    raise errors.CancelledError("任务已取消")
                emit(
                    JobEvent(
                        stage="refine",
                        progress=max(0.0, min(1.0, float(ratio))),
                        message=message or "DeepSeek 正在整理文字…",
                        audio_path=str(audio),
                    )
                )

            refined = self._get_refiner().refine(
                transcript, options, on_progress=on_refine_progress, cancel=cancel
            )
            if not (refined.text or "").strip():
                raise errors.RefineError(
                    "DeepSeek 返回了空结果",
                    hint="可以关闭「AI 整理」后重试，或稍后再试。",
                )
            if cancel.is_set():
                raise errors.CancelledError("任务已取消")

        return JobResult(audio_path=str(audio), transcript=transcript, refined=refined)

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def run(
        self,
        audio_path: str | Path,
        job: JobOptions,
        emit: Emit,
        cancel: Optional[threading.Event] = None,
    ) -> JobResult:
        """文件转写：转写 + 整理 + 导出文件；异常交由调用方捕获。"""
        cancel = cancel or threading.Event()
        audio = self._prepare_audio(audio_path, emit, cancel)
        self._busy = True
        started = time.monotonic()
        try:
            result = self._transcribe_and_refine(audio, job, emit, cancel)

            emit(
                JobEvent(
                    stage="export",
                    progress=0.0,
                    message="写出结果文件…",
                    audio_path=str(audio),
                )
            )
            out_dir = (
                Path(job.output_dir) if job.output_dir else self.settings.resolve_output_dir()
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            base_name = sanitize_filename(job.title or audio.stem)
            files = export_result(
                result,
                out_dir=out_dir,
                base_name=base_name,
                formats=list(job.export_formats or ["txt"]),
                title=job.title or audio.stem,
                include_raw=True,
            )
            result.output_files = [str(path) for path in files]
            result.elapsed = time.monotonic() - started
            emit(
                JobEvent(
                    stage="done",
                    progress=1.0,
                    message=f"完成，用时 {human_duration(result.elapsed)}",
                    audio_path=str(audio),
                    result=result,
                )
            )
            return result
        finally:
            self._last_activity = time.monotonic()
            self._busy = False

    def transcribe_and_refine(
        self,
        audio_path: str | Path,
        job: JobOptions,
        emit: Emit,
        cancel: Optional[threading.Event] = None,
        on_transcript: Optional[Callable[[Transcript], None]] = None,
    ) -> JobResult:
        """快速语音输入：转写 + 整理，不写任何文件（结果直接交给界面）。

        :param on_transcript: 可选回调；ASR 完成、DeepSeek 开始前调用，
            用于界面先把原始转写显示出来（降低等待感）。
        """
        cancel = cancel or threading.Event()
        audio = self._prepare_audio(audio_path, emit, cancel)
        self._busy = True
        started = time.monotonic()
        try:
            result = self._transcribe_and_refine(
                audio, job, emit, cancel, on_transcript=on_transcript
            )
            result.elapsed = time.monotonic() - started
            emit(
                JobEvent(
                    stage="done",
                    progress=1.0,
                    message=f"转换完成，用时 {human_duration(result.elapsed)}",
                    audio_path=str(audio),
                    result=result,
                )
            )
            return result
        finally:
            self._last_activity = time.monotonic()
            self._busy = False

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------
    def warm_up(self, cancel: Optional[threading.Event] = None) -> bool:
        """在后台线程预加载识别模型（不转写），降低第一次口述的等待。

        失败只记录日志并返回 ``False``（例如 GPU 环境异常），不抛出异常，
        真正使用时仍会再次尝试加载并给出中文错误提示。
        """
        try:
            asr = self._get_asr()
            loader = getattr(asr, "ensure_loaded", None)
            if loader is not None:
                loader(None, cancel or threading.Event())
            self._last_activity = time.monotonic()
            return True
        except Exception:
            log.warning("识别模型预热失败，将在首次使用时重试", exc_info=True)
            return False

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def last_activity(self) -> float:
        return self._last_activity

    def unload(self) -> None:
        """立即卸载模型（关闭程序或手动释放内存时调用）。"""
        with self._guard:
            self._release_asr()
            self._refiner = None
            self._refiner_signature = None

    def maybe_unload_idle(self) -> bool:
        """空闲超过设置时长后卸载 ASR 模型；返回是否执行了卸载。"""
        idle_seconds = self.settings.idle_unload_seconds()
        if idle_seconds <= 0 or self._busy or self._asr is None:
            return False
        if time.monotonic() - self._last_activity < idle_seconds:
            return False
        log.info("空闲 %.0f 秒，自动卸载语音模型释放内存", idle_seconds)
        self.unload()
        return True


__all__ = ["Pipeline"]
