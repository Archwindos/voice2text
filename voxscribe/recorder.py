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
    recorder.level -> float                       # v1.1 实时音量 0.0~1.0（未录音时 0.0）
    recorder.recent_levels(count=90) -> list[float]  # v1.1 最近 count 个音量（旧→新）
    pcm16_level(frames) -> float                  # v1.1 模块级：PCM16 字节 -> 归一化音量

实现要点：
* 16kHz / 单声道 / PCM16，用标准库 ``wave`` 直接写文件，无需额外依赖；
* sounddevice 懒 import：没有麦克风或没有 sounddevice 时，``available()`` 返回
  False，``start()`` 抛出 ``voxscribe.errors.AudioError``（中文提示）；
* 录音数据在 PortAudio 回调线程中收集进线程安全队列，主线程只读状态；
  停止时一次性拼接并写 wav（最长 2 小时，防止内存无限增长）；
* 实时音量：每次音频回调用纯 Python ``array('h')`` 计算 RMS 并软压缩到 0.0~1.0，
  写入线程安全的 ``collections.deque``（最长 240 个），供 GUI 每 50ms 轮询绘图；
  不引入 numpy 等新依赖（512 采样/帧的纯 Python 计算开销可忽略）；
* ``write_pcm16_wav()`` / ``pcm16_level()`` 抽成模块级函数，方便无麦克风机器单测。
"""

from __future__ import annotations

import logging
import math
import queue
import sys
import threading
import time
import wave
from array import array
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple

from . import errors

log = logging.getLogger(__name__)

#: 录音统一格式
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # PCM16，2 字节/采样
#: 最长录音时长保护（秒）
MAX_RECORD_SECONDS = 2 * 60 * 60
#: 实时音量历史最多保留的回调次数（16kHz/512 采样约 16 秒；GUI 只取最近 90 个）
LEVEL_HISTORY_MAX = 240
#: 软压缩参考点：RMS 达到 0.25 满幅即视为 1.0（正常说话约 0.02~0.08）
_LEVEL_FULL_SCALE = 0.25
#: PCM16 满幅值
_PCM16_FULL_SCALE = 32768.0
#: 写出 wav 时要求的最短有效数据（短于一帧视为没有录到声音）
_MIN_PCM_BYTES = SAMPLE_WIDTH * CHANNELS


def pcm16_level(frames: bytes) -> float:
    """把一段 PCM16（小端）音频换算成归一化音量。

    计算方式：先求 RMS，再软压缩到 ``0.0~1.0``：

    ``level = min(1.0, rms / 0.25) ** 0.5``

    这样正常说话（RMS 约 0.02~0.08）会落在 0.3~0.6，波形起伏明显；
    静音返回 0.0，满幅正弦接近 1.0。纯 Python ``array('h')`` 实现，
    512 采样/帧的开销远小于音频回调周期，不需要 numpy。

    :param frames: PCM16 小端字节（sounddevice 默认 int16 / 小端）。
    :return: ``0.0~1.0`` 的音量；空数据或静音返回 0.0。
    """
    if not frames:
        return 0.0
    usable = len(frames) - (len(frames) % SAMPLE_WIDTH)
    if usable <= 0:
        return 0.0
    samples: array[int] = array("h")
    samples.frombytes(frames[:usable])
    if sys.byteorder == "big":  # Windows/Python 默认小端；保持跨平台正确
        samples.byteswap()
    total = 0
    for sample in samples:
        total += sample * sample
    rms = math.sqrt(total / len(samples)) / _PCM16_FULL_SCALE
    if rms <= 0.0:
        return 0.0
    return min(1.0, rms / _LEVEL_FULL_SCALE) ** 0.5



def write_pcm16_wav(
    path: str | Path,
    pcm_data: bytes,
    *,
    samplerate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> Path:
    """把 PCM16（小端）裸数据写成标准 wav 文件。

    :param path: 目标文件路径；父目录不存在时自动创建。
    :param pcm_data: 交错排列的 PCM16 字节，长度必须是 ``2 * channels`` 的整数倍。
    :param samplerate: 采样率，默认 16000。
    :param channels: 声道数，默认 1。
    :raises ValueError: 采样率/声道数非法，或数据长度不是完整帧。
    :return: 写入后的文件路径。
    """
    if samplerate <= 0:
        raise ValueError(f"采样率必须为正数：{samplerate}")
    if channels <= 0:
        raise ValueError(f"声道数必须为正数：{channels}")
    frame_bytes = SAMPLE_WIDTH * channels
    if len(pcm_data) % frame_bytes != 0:
        raise ValueError(
            f"PCM 数据长度 {len(pcm_data)} 不是完整采样帧（帧大小 {frame_bytes} 字节）"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(samplerate)
        handle.writeframes(bytes(pcm_data))
    return target


class Recorder:
    """基于 sounddevice 的极简录音器。"""

    def __init__(self, recordings_dir: str | Path, samplerate: int = SAMPLE_RATE) -> None:
        samplerate = int(samplerate or 0)
        if samplerate <= 0:
            raise ValueError(f"采样率必须为正数：{samplerate}")
        self.recordings_dir = Path(recordings_dir)
        self.samplerate = samplerate
        self._max_frames = int(MAX_RECORD_SECONDS * samplerate)

        self._lock = threading.RLock()
        self._queue: Optional[queue.Queue] = None
        self._stream: Optional[Any] = None
        self._recording = False
        self._frames = 0
        self._elapsed = 0.0
        self._started_at = 0.0
        self._overflow = False
        # 实时音量：回调线程写、GUI 线程读，统一由 _lock 保护
        self._levels: Deque[float] = deque(maxlen=LEVEL_HISTORY_MAX)
        self._level = 0.0

    # ------------------------------------------------------------------
    # 设备查询（不抛异常）
    # ------------------------------------------------------------------
    @staticmethod
    def _import_sounddevice() -> Optional[Any]:
        """懒 import sounddevice；不可用时返回 None 并记录日志。"""
        try:
            import sounddevice  # noqa: PLC0415  懒加载：没有麦克风也能启动主程序

            return sounddevice
        except Exception as exc:  # 缺依赖 / 缺 PortAudio 动态库
            log.info("sounddevice 不可用：%s", exc)
            return None

    @staticmethod
    def available() -> bool:
        """当前机器是否有可用输入设备；任何异常都返回 False。"""
        sounddevice = Recorder._import_sounddevice()
        if sounddevice is None:
            return False
        try:
            devices = sounddevice.query_devices()
            return any(
                int(device.get("max_input_channels") or 0) > 0 for device in devices
            )
        except Exception:
            return False

    @staticmethod
    def list_input_devices() -> List[Tuple[int, str]]:
        """列出输入设备 ``[(设备索引, 描述), ...]``；失败时返回空列表。"""
        sounddevice = Recorder._import_sounddevice()
        if sounddevice is None:
            return []
        devices: List[Tuple[int, str]] = []
        try:
            for index, device in enumerate(sounddevice.query_devices()):
                try:
                    if int(device.get("max_input_channels") or 0) <= 0:
                        continue
                    name = str(device.get("name") or f"输入设备 {index}")
                    devices.append((int(index), name))
                except Exception:
                    continue
        except Exception:
            log.debug("枚举输入设备失败", exc_info=True)
            return []
        return devices

    # ------------------------------------------------------------------
    # 录音控制
    # ------------------------------------------------------------------
    def start(self, device: Optional[int] = None) -> None:
        """开始录音；设备不可用 / 被占用时抛 :class:`voxscribe.errors.AudioError`。"""
        with self._lock:
            if self._recording:
                raise errors.AudioError(
                    "已经在录音中，请先停止当前录音",
                    hint="如需重新录制，请先停止或取消当前录音。",
                )

        sounddevice = self._import_sounddevice()
        if sounddevice is None:
            raise errors.AudioError(
                "缺少录音组件（sounddevice），无法使用麦克风",
                hint="请重新运行安装脚本安装依赖后重启程序；"
                "也可以直接选择已有的音频文件进行转写。",
            )

        device_name = self._describe_device(sounddevice, device)
        try:
            # 先校验参数，能在打开流之前给出更明确的中文错误
            sounddevice.check_input_settings(
                device=device,
                channels=CHANNELS,
                dtype="int16",
                samplerate=self.samplerate,
            )
        except Exception as exc:
            raise errors.AudioError(
                f"无法使用麦克风{('：' + device_name) if device_name else ''}",
                hint="请确认麦克风已连接、没有被其他程序独占，"
                "并在系统「设置 → 隐私 → 麦克风」中允许应用访问麦克风。",
            ) from exc

        pcm_queue: queue.Queue = queue.Queue()
        with self._lock:
            self._queue = pcm_queue
            self._frames = 0
            self._elapsed = 0.0
            self._overflow = False
            self._started_at = time.monotonic()
            self._recording = True
        self._reset_level_history()  # 新录音从空白波形开始（上一轮 stop 保留的历史作废）

        def _callback(indata, frame_count: int, time_info, status) -> None:  # noqa: ANN001
            current_queue = self._queue
            if current_queue is None:  # 理论上不会发生：停止后不会再有回调
                return
            if status:
                log.debug("录音状态提示：%s", status)
            payload = indata.tobytes()
            level = pcm16_level(payload)  # 512 采样纯 Python RMS，开销可忽略
            with self._lock:
                if self._queue is not current_queue:
                    return
                self._frames += int(frame_count)
                self._elapsed = self._frames / float(self.samplerate)
                overflow = self._frames >= self._max_frames
            self._push_level(level)
            current_queue.put_nowait(payload)
            if overflow:
                self._overflow = True
                log.warning("录音达到最长时长 %.0f 秒，已自动停止", MAX_RECORD_SECONDS)
                raise sounddevice.CallbackStop

        stream = None
        try:
            stream = sounddevice.InputStream(
                samplerate=self.samplerate,
                channels=CHANNELS,
                dtype="int16",
                device=device,
                callback=_callback,
            )
            stream.start()
        except Exception as exc:
            if stream is not None:
                self._close_stream(stream)
            with self._lock:
                self._recording = False
                self._queue = None
                self._stream = None
                self._frames = 0
                self._elapsed = 0.0
            self._reset_level_history()
            raise errors.AudioError(
                f"无法开始录音{('：' + device_name) if device_name else ''}",
                hint="麦克风可能被其他程序占用。请关闭占用麦克风的软件后重试，"
                "或换一个输入设备（设置 → 录音设备）。",
            ) from exc

        with self._lock:
            self._stream = stream
        log.info("开始录音：设备 %s，%d Hz 单声道", device_name or "默认输入设备", self.samplerate)

    def stop(self) -> Path:
        """停止录音并保存为 wav，返回文件路径。"""
        with self._lock:
            if not self._recording or self._stream is None:
                raise errors.AudioError(
                    "当前没有正在进行的录音",
                    hint="请先点击「开始录音」再停止。",
                )
            stream = self._stream
            self._stream = None
            self._recording = False
            if self._started_at:
                self._elapsed = max(self._elapsed, time.monotonic() - self._started_at)

        # 先停流（回调线程可能仍在写队列），再统一取数据
        self._close_stream(stream)
        with self._lock:
            pcm_queue = self._queue
            self._queue = None
            overflow = self._overflow
            duration = self._elapsed

        pcm_data = self._drain_queue(pcm_queue)
        if overflow:
            log.warning(
                "录音达到最长时长 %.0f 秒并自动停止，仅保存已录到的部分",
                MAX_RECORD_SECONDS,
            )
        if len(pcm_data) < _MIN_PCM_BYTES:
            raise errors.AudioError(
                "没有录到任何声音，录音文件未保存",
                hint="请确认麦克风是否静音、是否选错设备，"
                "并在系统设置中检查应用是否有麦克风权限。",
            )
        target = self._new_recording_path()
        written = write_pcm16_wav(
            target,
            pcm_data,
            samplerate=self.samplerate,
            channels=CHANNELS,
        )
        log.info(
            "录音已保存：%s（%.1f 秒，%.0fKB）",
            written,
            duration,
            len(pcm_data) / 1024,
        )
        return written

    def cancel(self) -> None:
        """停止录音并丢弃数据（未保存文件）；没有录音时安全返回。"""
        with self._lock:
            stream = self._stream
            pcm_queue = self._queue
            self._stream = None
            self._queue = None
            was_recording = self._recording
            self._recording = False
            self._frames = 0
            self._elapsed = 0.0
            self._started_at = 0.0
            # 取消即数据作废：实时音量历史一并清空（stop() 才保留用于定格）
            self._levels.clear()
            self._level = 0.0
        if stream is not None:
            self._close_stream(stream)
        self._drain_queue(pcm_queue)
        if was_recording:
            log.info("录音已取消，未保存文件")

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def elapsed_seconds(self) -> float:
        """已录制时长（秒，按已收集到的采样帧数计算）。"""
        with self._lock:
            return max(0.0, self._elapsed)

    @property
    def level(self) -> float:
        """最近一次音频回调的归一化音量（0.0~1.0）；未录音/停止后为 0.0。

        GUI 可每 50ms 读取一次做麦克风电平指示；正常说话通常在 0.2~0.8 之间变化。
        """
        with self._lock:
            return self._level if self._recording else 0.0

    def recent_levels(self, count: int = 90) -> List[float]:
        """返回最近 ``count`` 个回调音量（旧→新），用于绘制滚动波形。

        * 录音中每次音频回调追加一个值，最多保留 :data:`LEVEL_HISTORY_MAX` 个；
        * ``start()`` 开始新录音时清空；``stop()`` 停止后保留最后一批供界面定格；
        * 未录音且无历史数据时返回 ``[]``；``count <= 0`` 返回 ``[]``；
          历史不足 ``count`` 个时返回全部已有值；
        * 线程安全：回调线程写入、GUI 线程读取互不阻塞（持锁时间极短）。
        """
        try:
            wanted = int(count)
        except (TypeError, ValueError):
            return []
        if wanted <= 0:
            return []
        with self._lock:
            if not self._levels:
                return []
            values = list(self._levels)
        return values[-wanted:]

    def _push_level(self, raw_level: float) -> None:
        """记录一次回调音量（PortAudio 回调线程调用；测试可注入假数据）。"""
        try:
            value = float(raw_level)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):  # NaN/Inf 等异常值直接忽略
            return
        value = max(0.0, min(1.0, value))
        with self._lock:
            self._level = value
            self._levels.append(value)

    def _reset_level_history(self) -> None:
        """清空实时音量历史与当前值（``start()`` 新录音时调用）。"""
        with self._lock:
            self._levels.clear()
            self._level = 0.0

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _describe_device(sounddevice: Any, device: Optional[int]) -> str:
        """取设备名用于中文提示；查询失败返回空字符串。"""
        try:
            info = sounddevice.query_devices(device if device is not None else None)
            if isinstance(info, dict):
                return str(info.get("name") or "")
        except Exception:
            pass
        return ""

    @staticmethod
    def _close_stream(stream: Any) -> None:
        """停止并关闭音频流；重复或异常调用都不抛。"""
        try:
            stream.stop()
        except Exception:
            log.debug("停止音频流失败", exc_info=True)
        try:
            stream.close()
        except Exception:
            log.debug("关闭音频流失败", exc_info=True)

    @staticmethod
    def _drain_queue(pcm_queue: Optional[queue.Queue]) -> bytes:
        """取出队列里所有已录制的 PCM 数据。"""
        if pcm_queue is None:
            return b""
        chunks: List[bytes] = []
        while True:
            try:
                chunks.append(pcm_queue.get_nowait())
            except queue.Empty:
                break
        return b"".join(chunks)

    def _new_recording_path(self) -> Path:
        """生成不重复的录音文件路径：``录音_20250101_120000.wav``。"""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        target = self.recordings_dir / f"录音_{stamp}.wav"
        counter = 1
        while target.exists():
            target = self.recordings_dir / f"录音_{stamp}_{counter}.wav"
            counter += 1
        return target


__all__ = [
    "Recorder",
    "write_pcm16_wav",
    "pcm16_level",
    "SAMPLE_RATE",
    "CHANNELS",
    "SAMPLE_WIDTH",
    "MAX_RECORD_SECONDS",
    "LEVEL_HISTORY_MAX",
]
