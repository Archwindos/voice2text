"""麦克风录音模块（可选功能，依赖 sounddevice）。

负责人：asr-dev。接口契约（GUI 依赖，改动需同步）：

    recorder = Recorder(recordings_dir=".../data/recordings")
    Recorder.available() -> bool                  # 是否有可用输入设备
    Recorder.list_input_devices() -> list[tuple[int, str]]  # (设备索引, 描述)
    recorder.start(device=None) -> None
    recorder.stop() -> Path                       # 保存为 16kHz 单声道 wav，返回路径
    recorder.cancel() -> None                     # 停止并删除未完成的录音
    recorder.is_recording -> bool
    recorder.elapsed_seconds -> float

要求：
* 16kHz / 单声道 / PCM16，直接写标准 wav，无需额外依赖；
* 录音数据在后台回调线程收集，主线程只读状态；
* 无麦克风、设备被占用、缺少 sounddevice 时，`available()` 返回 False，
  start() 抛出 voxscribe.errors.AudioError（中文提示）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from . import errors


class Recorder:
    """基于 sounddevice 的极简录音器。"""

    def __init__(self, recordings_dir: str | Path, samplerate: int = 16000) -> None:
        raise NotImplementedError("录音模块尚未实现（asr-dev）")

    @staticmethod
    def available() -> bool:
        raise NotImplementedError

    @staticmethod
    def list_input_devices() -> List[Tuple[int, str]]:
        raise NotImplementedError

    def start(self, device: Optional[int] = None) -> None:
        raise NotImplementedError

    def stop(self) -> Path:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError

    @property
    def is_recording(self) -> bool:
        raise NotImplementedError

    @property
    def elapsed_seconds(self) -> float:
        raise NotImplementedError
