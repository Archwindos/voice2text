"""语音识别模块（本地 faster-whisper）。

负责人：asr-dev。接口契约（pipeline / GUI 依赖，改动需同步）：

    asr = WhisperASR(model_size="small", compute_type="int8",
                     cpu_threads=4, model_dir=".../data/models",
                     beam_size=5, vad_filter=True)
    asr.is_model_available() -> bool        # 本地缓存中是否已有模型
    asr.model_size_hint() -> str            # 例如 "484MB"
    asr.transcribe(path, language="zh",
                   on_progress=callable(ratio: float, message: str),
                   cancel=threading.Event()) -> Transcript
    asr.unload() -> None                    # 释放模型内存
    asr.loaded -> bool

要求：
* 懒加载：首次 transcribe 时才加载模型；同一实例复用。
* 进度回调：0.0~1.0，按已转写时长 / 音频总时长估算。
* 取消：每个 segment 之间检查 cancel.is_set()，取消时抛出
  voxscribe.errors.CancelledError。
* 任何底层异常包装为 voxscribe.errors.ASRError（给出中文提示）。
* 默认 CPU int8；不引入 torch；解码交给 faster-whisper 自带的 PyAV。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

from .models import Transcript

ProgressCallback = Callable[[float, str], None]


class WhisperASR:
    """faster-whisper 的懒加载封装。"""

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        model_dir: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        raise NotImplementedError("ASR 模块尚未实现（asr-dev）")

    @property
    def loaded(self) -> bool:
        raise NotImplementedError

    def is_model_available(self) -> bool:
        raise NotImplementedError

    def model_size_hint(self) -> str:
        raise NotImplementedError

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = "zh",
        on_progress: Optional[ProgressCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> Transcript:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError
