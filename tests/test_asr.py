"""语音识别（WhisperASR）与麦克风录音（Recorder）测试。

分两层：

* 单元测试：不下载模型、不依赖麦克风，默认全跑；
  覆盖 v1.1 的实时音量波形 API（``level`` / ``recent_levels()`` / ``pcm16_level()``），
  并用假 sounddevice 验证回调→音量历史→wav 的完整链路；
* 端到端测试：用 Windows 中文 TTS 生成的 ``tests/fixtures/tts_zh.wav``，
  以真实 faster-whisper ``tiny`` 模型转写（模型缓存在项目 ``data/models``）。
  默认跳过，需要显式开启：

      $env:VOXSCRIBE_RUN_ASR = '1'
      & .\\.venv\\Scripts\\python.exe -m pytest tests/test_asr.py -q

  首次运行会自动下载 tiny 模型（约 75MB）。若直连 HuggingFace 失败
  （国内网络常见 DNS/连接问题），可设置镜像后再运行：

      $env:HF_ENDPOINT = 'https://hf-mirror.com'
      $env:HF_HUB_DISABLE_XET = '1'

  模型下载到 ``data/models`` 后，后续运行（以及 App 本身）完全离线可用。

  GPU 相关用例默认跳过，需要显式开启（要求本机有可用 NVIDIA GPU）：

      $env:VOXSCRIBE_RUN_GPU = '1'
      & .\\.venv\\Scripts\\python.exe -m pytest tests/test_asr.py -q
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import List, Tuple

import pytest

from voxscribe import asr as asr_module
from voxscribe import errors
from voxscribe.asr import (
    MODEL_SIZE_HINTS,
    ZH_INITIAL_PROMPT,
    WhisperASR,
    ensure_cuda_dll_directories,
    find_cuda_dll_directories,
    get_cuda_compute_types,
    get_cuda_device_count,
    is_cuda_device,
    resolve_compute_type,
)
from voxscribe.config import ASR_MODELS
from voxscribe.recorder import (
    CHANNELS,
    LEVEL_HISTORY_MAX,
    MAX_RECORD_SECONDS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    Recorder,
    pcm16_level,
    write_pcm16_wav,
)
from voxscribe.util import process_memory_mb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_AUDIO = PROJECT_ROOT / "tests" / "fixtures" / "tts_zh.wav"
MAKE_AUDIO_SCRIPT = PROJECT_ROOT / "tools" / "make_test_audio.ps1"
E2E_MODEL_DIR = PROJECT_ROOT / "data" / "models"
RUN_ASR = os.environ.get("VOXSCRIBE_RUN_ASR") == "1"
#: GPU 重载/真实转写用例默认跳过，避免在无 GPU 环境失败
RUN_GPU = os.environ.get("VOXSCRIBE_RUN_GPU") == "1"
#: 端到端断言用的关键词：TTS 文本「项目 / 进度 / 前端 / 后端 / 百分之七十 / 两周」。
#: 同时收纳繁体写法，避免不同模型版本输出繁体时误判失败。
E2E_KEYWORDS = (
    "项目", "項目", "进度", "進度", "前端", "后端", "後端",
    "百分之七十", "两周", "兩週", "会议", "會議", "讨论", "討論",
)


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def _make_fake_hf_cache(
    model_dir: Path, model_size: str = "small", *, with_model_bin: bool = True
) -> Path:
    """伪造 HuggingFace 缓存目录：models--Systran--faster-whisper-<size>。"""
    repo = f"Systran/faster-whisper-{model_size}"
    snapshot = (
        model_dir
        / ("models--" + repo.replace("/", "--"))
        / "snapshots"
        / "0123456789abcdef"
    )
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    if with_model_bin:
        (snapshot / "model.bin").write_bytes(b"\x00" * 128)
    return snapshot


class _FakeRawSegment:
    """模拟 faster_whisper.transcribe.Segment。"""

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    """模拟 faster_whisper.transcribe.TranscriptionInfo。"""

    def __init__(self, duration: float = 10.0, language: str = "zh") -> None:
        self.duration = duration
        self.language = language


class _FakeWhisperModel:
    """不加载真实模型，只验证 WhisperASR 的进度/取消/组装逻辑。"""

    def __init__(self, segments: List[_FakeRawSegment], **info_kwargs) -> None:
        self._segments = segments
        self._info = _FakeInfo(**info_kwargs)
        self.kwargs: dict = {}

    def transcribe(self, audio_path, **kwargs) -> Tuple[object, _FakeInfo]:
        self.kwargs = kwargs
        return iter(self._segments), self._info


def _asr_with_fake_model(
    tmp_path: Path, segments: List[_FakeRawSegment], **info_kwargs
) -> WhisperASR:
    asr = WhisperASR(model_size="tiny", model_dir=str(tmp_path / "models"))
    asr._model = _FakeWhisperModel(segments, **info_kwargs)  # noqa: SLF001 白盒注入
    return asr


def _ensure_tts_audio() -> Path:
    """确保测试音频存在；缺失时调用 tools/make_test_audio.ps1 生成。"""
    if FIXTURE_AUDIO.is_file() and FIXTURE_AUDIO.stat().st_size > 44:
        return FIXTURE_AUDIO
    if os.name != "nt" or not MAKE_AUDIO_SCRIPT.is_file():
        pytest.skip("缺少中文 TTS 测试音频，且当前环境无法生成")
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        pytest.skip("找不到 PowerShell，无法生成中文 TTS 测试音频")
    try:
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(MAKE_AUDIO_SCRIPT),
            ],
            capture_output=True,
            timeout=180,
        )
    except Exception as exc:  # pragma: no cover - 环境问题
        pytest.skip(f"生成中文 TTS 测试音频失败：{exc}")
    if result.returncode != 0 or not FIXTURE_AUDIO.is_file():
        stderr = (result.stderr or b"").decode("utf-8", "replace").strip()
        pytest.skip(f"生成中文 TTS 测试音频失败：{stderr[:300]}")
    return FIXTURE_AUDIO


def _memory_mb() -> float:
    """当前进程内存（MB）。

    优先调用 ``voxscribe.util.process_memory_mb()``；该函数在本机某些 Python
    环境下会因 ``GetCurrentProcess()`` 伪句柄被 ctypes 默认按 int 截断而返回
    0.0（见交付报告），这里用正确的 ctypes 类型兜底，保证能记录真实数字。
    """
    value = process_memory_mb()
    if value > 0 or os.name != "nt":
        return value
    try:
        import ctypes
        from ctypes import wintypes

        class _MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_MemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:  # pragma: no cover - 兜底实现失败时不阻塞测试
        pass
    return 0.0


def _cuda_available() -> bool:
    """当前机器是否有可用的 CUDA GPU（ctranslate2 视角）。"""
    try:
        return get_cuda_device_count() > 0 and "float16" in get_cuda_compute_types()
    except Exception:  # pragma: no cover - 环境异常
        return False


def _pcm16_sine(
    frames: int = 512,
    *,
    amplitude: int = 32767,
    frequency: float = 440.0,
    samplerate: int = SAMPLE_RATE,
) -> bytes:
    """生成一段正弦波 PCM16 小端字节（用于音量计算测试）。"""
    samples = array(
        "h",
        [
            int(amplitude * math.sin(2 * math.pi * frequency * index / samplerate))
            for index in range(frames)
        ],
    )
    return samples.tobytes()


class _FakeInputStream:
    """假 sounddevice.InputStream：start() 时同步喂一帧正弦波数据。

    用它在没有麦克风/不打扰用户的情况下验证
    「回调 → level/recent_levels → 队列 → wav」完整链路。
    """

    instances: List["_FakeInputStream"] = []

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.callback = kwargs["callback"]
        self.started = False
        self.closed = False
        _FakeInputStream.instances.append(self)

    def start(self) -> None:
        self.started = True
        samples = array(
            "h",
            [
                int(12000 * math.sin(2 * math.pi * 440 * index / SAMPLE_RATE))
                for index in range(512)
            ],
        )
        self.callback(samples, 512, None, None)

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class _FakeSoundDevice:
    """假 sounddevice 模块（只实现 Recorder 用到的接口）。"""

    CallbackStop = RuntimeError

    def __init__(self) -> None:
        _FakeInputStream.instances = []

    @staticmethod
    def check_input_settings(**kwargs) -> None:  # noqa: ANN003
        return None

    @staticmethod
    def query_devices(device=None):  # noqa: ANN001
        return {"name": "Fake Mic", "max_input_channels": 1}

    @staticmethod
    def InputStream(**kwargs) -> _FakeInputStream:  # noqa: ANN003
        return _FakeInputStream(**kwargs)


# ===========================================================================
# 单元测试：模型信息与本地缓存检测
# ===========================================================================
@pytest.mark.parametrize(
    ("model_size", "expected"),
    [
        ("tiny", "75MB"),
        ("base", "145MB"),
        ("small", "484MB"),
        ("medium", "1.5GB"),
        ("large-v3", "3.1GB"),
        ("TINY", "75MB"),
        ("small.en", "484MB"),
        ("Systran/faster-whisper-small", "484MB"),
        ("large", "3.1GB"),
    ],
)
def test_model_size_hint(model_size: str, expected: str, tmp_path: Path) -> None:
    asr = WhisperASR(model_size=model_size, model_dir=str(tmp_path / "models"))
    assert asr.model_size_hint() == expected


def test_model_size_hint_covers_config_models() -> None:
    for model_size in ASR_MODELS:
        assert WhisperASR(model_size=model_size).model_size_hint() != "未知大小"


def test_model_size_hint_unknown(tmp_path: Path) -> None:
    asr = WhisperASR(model_size="不存在的规格", model_dir=str(tmp_path / "models"))
    assert asr.model_size_hint() == "未知大小"


def test_model_size_hint_local_model_dir(tmp_path: Path) -> None:
    local_model = tmp_path / "本地模型"
    local_model.mkdir()
    (local_model / "model.bin").write_bytes(b"\x00" * 16)
    asr = WhisperASR(model_size=str(local_model))
    assert asr.model_size_hint() == "本地模型"
    assert asr.is_model_available() is True


def test_is_model_available_detects_hf_cache(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    asr_small = WhisperASR(model_size="small", model_dir=str(model_dir))
    asr_tiny = WhisperASR(model_size="tiny", model_dir=str(model_dir))
    assert asr_small.is_model_available() is False

    _make_fake_hf_cache(model_dir, "small")
    assert asr_small.is_model_available() is True
    assert asr_tiny.is_model_available() is False


def test_is_model_available_ignores_incomplete_cache(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    _make_fake_hf_cache(model_dir, "small", with_model_bin=False)
    asr = WhisperASR(model_size="small", model_dir=str(model_dir))
    assert asr.is_model_available() is False


def test_is_model_available_without_model_dir(tmp_path: Path) -> None:
    asr = WhisperASR(model_size="small", model_dir=str(tmp_path / "不存在" / "models"))
    assert asr.is_model_available() is False


# ===========================================================================
# 单元测试：加载状态与卸载
# ===========================================================================
def test_loaded_and_unload_idempotent(tmp_path: Path) -> None:
    asr = WhisperASR(model_size="tiny", model_dir=str(tmp_path / "models"))
    assert asr.loaded is False
    # 未加载时卸载多次是安全的
    asr.unload()
    asr.unload()
    assert asr.loaded is False

    asr._model = object()  # noqa: SLF001 模拟已加载
    assert asr.loaded is True
    asr.unload()
    assert asr.loaded is False
    asr.unload()
    assert asr.loaded is False


def test_unload_releases_reference(tmp_path: Path) -> None:
    asr = WhisperASR(model_size="tiny", model_dir=str(tmp_path / "models"))
    fake_model = _FakeWhisperModel([])
    asr._model = fake_model  # noqa: SLF001
    asr.unload()
    assert asr._model is None  # noqa: SLF001


# ===========================================================================
# 单元测试：转写主流程（用假模型，不下载）
# ===========================================================================
def test_transcribe_builds_transcript_and_progress(tmp_path: Path) -> None:
    segments = [
        _FakeRawSegment(0.0, 2.0, "  今天开会  "),
        _FakeRawSegment(2.0, 4.0, "   "),  # 纯空白：丢弃
        _FakeRawSegment(4.0, 6.0, "讨论项目进度"),
    ]
    asr = _asr_with_fake_model(tmp_path, segments)
    events: List[Tuple[float, str]] = []
    transcript = asr.transcribe(
        tmp_path / "a.wav",
        language="zh",
        on_progress=lambda ratio, message: events.append((ratio, message)),
    )

    assert [segment.text for segment in transcript.segments] == ["今天开会", "讨论项目进度"]
    assert transcript.segments[0].start == 0.0
    assert transcript.duration == 10.0
    assert transcript.language == "zh"
    assert transcript.text == "今天开会\n讨论项目进度"

    ratios = [ratio for ratio, _ in events]
    assert ratios[0] == 0.0
    assert ratios[-1] == 1.0
    assert all(0.0 <= ratio <= 1.0 for ratio in ratios)
    assert any("正在转写" in message for _, message in events)
    assert all(message for _, message in events)

    # 传给 faster-whisper 的关键参数保持默认省资源策略
    kwargs = asr._model.kwargs  # noqa: SLF001
    assert kwargs["language"] == "zh"
    assert kwargs["beam_size"] == 5
    assert kwargs["vad_filter"] is True
    assert kwargs["word_timestamps"] is False
    assert kwargs["task"] == "transcribe"
    # 中文默认带 initial_prompt（实测能避免繁体输出、提升常用词准确率）
    assert kwargs["initial_prompt"] == ZH_INITIAL_PROMPT


def test_initial_prompt_not_used_for_other_languages(tmp_path: Path) -> None:
    asr = _asr_with_fake_model(tmp_path, [_FakeRawSegment(0.0, 1.0, "hello")])
    asr.transcribe(tmp_path / "a.wav", language="en")
    assert asr._model.kwargs["initial_prompt"] is None  # noqa: SLF001


def test_initial_prompt_can_be_overridden(tmp_path: Path) -> None:
    asr = WhisperASR(
        model_size="tiny",
        model_dir=str(tmp_path / "models"),
        initial_prompt="自定义提示词。",
    )
    asr._model = _FakeWhisperModel([_FakeRawSegment(0.0, 1.0, "你好")])  # noqa: SLF001
    asr.transcribe(tmp_path / "a.wav", language="en")
    assert asr._model.kwargs["initial_prompt"] == "自定义提示词。"  # noqa: SLF001


def test_download_monitor_reports_progress(tmp_path: Path) -> None:
    """模型下载期间的后台监控线程应持续更新「已下载」消息。"""
    asr = WhisperASR(model_size="small", model_dir=str(tmp_path / "models"))
    target = asr._download_target_dir()  # noqa: SLF001
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.bin.part").write_bytes(b"\x00" * 512 * 1024)

    messages: List[str] = []
    monitor = asr._start_download_monitor(  # noqa: SLF001
        lambda ratio, message: messages.append(message)
    )
    try:
        time.sleep(1.2)
    finally:
        asr._stop_monitor(monitor)  # noqa: SLF001
    assert messages, "下载监控线程没有回调任何消息"
    assert all("正在下载识别模型" in message for message in messages)
    assert any("已下载" in message for message in messages)



def test_transcribe_auto_language_uses_detected(tmp_path: Path) -> None:
    asr = _asr_with_fake_model(
        tmp_path, [_FakeRawSegment(0.0, 1.0, "hello")], language="en"
    )
    transcript = asr.transcribe(tmp_path / "a.wav", language=None)
    assert transcript.language == "en"
    assert asr._model.kwargs["language"] is None  # noqa: SLF001


def test_transcribe_cancel_before_start_raises(tmp_path: Path) -> None:
    asr = WhisperASR(model_size="tiny", model_dir=str(tmp_path / "models"))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(errors.CancelledError, match="任务已取消"):
        asr.transcribe(tmp_path / "不存在.wav", cancel=cancel)
    # 取消优先于模型加载：不应该开始下载/加载
    assert asr.loaded is False


def test_transcribe_cancel_between_segments_raises(tmp_path: Path) -> None:
    segments = [_FakeRawSegment(i, i + 1, f"第{i}段") for i in range(10)]
    asr = _asr_with_fake_model(tmp_path, segments)
    cancel = threading.Event()
    ratios: List[float] = []

    def on_progress(ratio: float, message: str) -> None:
        ratios.append(ratio)
        if len(ratios) >= 2:  # 第一段转写完成后取消
            cancel.set()

    with pytest.raises(errors.CancelledError, match="任务已取消"):
        asr.transcribe(tmp_path / "a.wav", on_progress=on_progress, cancel=cancel)
    assert cancel.is_set()


def test_progress_callback_cancelled_error_propagates(tmp_path: Path) -> None:
    asr = _asr_with_fake_model(tmp_path, [_FakeRawSegment(0.0, 1.0, "你好")])

    def on_progress(ratio: float, message: str) -> None:
        raise errors.CancelledError("任务已取消")

    with pytest.raises(errors.CancelledError):
        asr.transcribe(tmp_path / "a.wav", on_progress=on_progress)


def test_transcribe_wraps_decode_error(tmp_path: Path) -> None:
    class _BrokenModel:
        def transcribe(self, audio_path, **kwargs):
            raise PermissionError("拒绝访问")

    asr = _asr_with_fake_model(tmp_path, [])
    asr._model = _BrokenModel()  # noqa: SLF001
    with pytest.raises(errors.ASRError) as excinfo:
        asr.transcribe(tmp_path / "a.wav")
    assert "无法读取音频文件" in str(excinfo.value)
    assert excinfo.value.user_hint


def test_transcribe_wraps_invalid_audio_error(tmp_path: Path) -> None:
    class _BrokenModel:
        def transcribe(self, audio_path, **kwargs):
            raise ValueError("Invalid data found when processing input")

    asr = _asr_with_fake_model(tmp_path, [])
    asr._model = _BrokenModel()  # noqa: SLF001
    with pytest.raises(errors.ASRError) as excinfo:
        asr.transcribe(tmp_path / "损坏.wav")
    assert "无法解码音频文件" in str(excinfo.value)
    assert excinfo.value.user_hint


def test_invalid_model_size_wrapped_as_asr_error(tmp_path: Path) -> None:
    asr = WhisperASR(model_size="不存在的模型", model_dir=str(tmp_path / "models"))
    with pytest.raises(errors.ASRError) as excinfo:
        asr.transcribe(tmp_path / "a.wav")
    assert "模型" in str(excinfo.value)
    assert excinfo.value.user_hint
    assert asr.loaded is False


# ===========================================================================
# 单元测试：录音器与 wav 写出
# ===========================================================================
def test_recorder_available_never_raises() -> None:
    assert isinstance(Recorder.available(), bool)


def test_list_input_devices_format() -> None:
    devices = Recorder.list_input_devices()
    assert isinstance(devices, list)
    for item in devices:
        assert isinstance(item, tuple) and len(item) == 2
        index, name = item
        assert isinstance(index, int) and index >= 0
        assert isinstance(name, str) and name


def test_recorder_initial_state(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "recordings")
    assert recorder.is_recording is False
    assert recorder.elapsed_seconds == 0.0
    assert recorder.samplerate == SAMPLE_RATE
    # 没有开始录音时取消是安全的，且可以重复调用
    recorder.cancel()
    recorder.cancel()
    assert recorder.is_recording is False


def test_recorder_stop_without_start_raises(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "recordings")
    with pytest.raises(errors.AudioError) as excinfo:
        recorder.stop()
    assert "没有正在进行的录音" in str(excinfo.value)
    assert excinfo.value.user_hint


def test_write_pcm16_wav_header_and_data(tmp_path: Path) -> None:
    frames = 1600  # 0.1 秒
    pcm = b"".join(int(index % 1000 * 32).to_bytes(2, "little", signed=True) for index in range(frames))
    target = tmp_path / "nested" / "录音.wav"

    written = write_pcm16_wav(target, pcm)

    assert written == target and target.is_file()
    with wave.open(str(target), "rb") as handle:
        assert handle.getnchannels() == CHANNELS == 1
        assert handle.getsampwidth() == SAMPLE_WIDTH == 2
        assert handle.getframerate() == SAMPLE_RATE == 16000
        assert handle.getnframes() == frames
        assert handle.readframes(frames) == pcm


def test_write_pcm16_wav_empty(tmp_path: Path) -> None:
    target = write_pcm16_wav(tmp_path / "empty.wav", b"")
    with wave.open(str(target), "rb") as handle:
        assert handle.getnframes() == 0


def test_write_pcm16_wav_rejects_partial_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="完整采样帧"):
        write_pcm16_wav(tmp_path / "bad.wav", b"\x00")


def test_record_roundtrip_when_microphone_available(tmp_path: Path) -> None:
    if not Recorder.available():
        pytest.skip("当前机器没有可用输入设备")
    recorder = Recorder(tmp_path / "recordings")
    try:
        recorder.start()
    except errors.AudioError as exc:  # 设备被占用等环境问题
        pytest.skip(f"麦克风不可用：{exc}")
    try:
        time.sleep(0.4)
        assert recorder.is_recording is True
        assert recorder.elapsed_seconds > 0.0
        recorded = recorder.stop()
    except errors.AudioError as exc:  # 没有采到数据（无声卡回环等）
        recorder.cancel()
        pytest.skip(f"麦克风未采集到数据：{exc}")

    assert recorded.parent == recorder.recordings_dir
    assert recorded.suffix == ".wav"
    with wave.open(str(recorded), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == recorder.samplerate
        assert handle.getnframes() > 0
    assert recorder.is_recording is False
    assert recorder.elapsed_seconds > 0.0


def test_cancel_discards_recording(tmp_path: Path) -> None:
    if not Recorder.available():
        pytest.skip("当前机器没有可用输入设备")
    recorder = Recorder(tmp_path / "recordings")
    try:
        recorder.start()
    except errors.AudioError as exc:
        pytest.skip(f"麦克风不可用：{exc}")
    time.sleep(0.2)
    recorder.cancel()
    assert recorder.is_recording is False
    assert recorder.elapsed_seconds == 0.0
    assert list(recorder.recordings_dir.glob("*.wav")) == []
    recorder.cancel()  # 幂等


def test_recorder_limits() -> None:
    assert MAX_RECORD_SECONDS == 2 * 60 * 60


# ===========================================================================
# v1.1 实时音量波形：pcm16_level / level / recent_levels
# ===========================================================================
def test_pcm16_level_empty_and_silence() -> None:
    assert pcm16_level(b"") == 0.0
    assert pcm16_level(b"\x00" * 1024) == 0.0
    assert pcm16_level(b"\x00") == 0.0  # 不足一个采样：忽略
    assert pcm16_level(b"\x00\x00\x00") == 0.0  # 尾部不完整字节：忽略


def test_pcm16_level_full_scale_sine_is_loud() -> None:
    level = pcm16_level(_pcm16_sine(512))
    assert level > 0.5  # 满幅正弦 RMS≈0.707，软压缩后接近 1.0
    assert level <= 1.0
    assert level == pytest.approx(1.0, abs=0.02)


def test_pcm16_level_tracks_amplitude() -> None:
    quiet = pcm16_level(_pcm16_sine(512, amplitude=327))  # 约 1% 幅度
    medium = pcm16_level(_pcm16_sine(512, amplitude=4000))
    loud = pcm16_level(_pcm16_sine(512))
    assert 0.0 < quiet < medium < loud <= 1.0
    assert quiet < 0.2


def test_pcm16_level_random_range() -> None:
    # 任意 PCM16 数据都应落在 [0, 1]
    pcm = array("h", [((index * 7919) % 65536) - 32768 for index in range(256)]).tobytes()
    level = pcm16_level(pcm)
    assert 0.0 <= level <= 1.0


def test_level_and_recent_levels_initial_state(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "recordings")
    assert recorder.level == 0.0
    assert recorder.recent_levels() == []
    assert recorder.recent_levels(0) == []
    assert recorder.recent_levels(-5) == []


def test_recent_levels_order_and_count(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "recordings")
    values = [0.0, 0.1, 0.25, 0.5, 1.0]
    for value in values:
        recorder._push_level(value)  # noqa: SLF001 模拟回调线程写入
    # 旧→新，且 count 只截取最近 N 个
    assert recorder.recent_levels() == values
    assert recorder.recent_levels(3) == [0.25, 0.5, 1.0]
    assert recorder.recent_levels(100) == values
    # 未录音时 level 归零，但历史保留（stop 后 GUI 定格语义）
    assert recorder.level == 0.0


def test_recent_levels_clamps_and_ignores_bad_values(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "recordings")
    recorder._push_level(2.0)  # noqa: SLF001 越界值截断到 1.0
    recorder._push_level(-1.0)  # noqa: SLF001 负值截断到 0.0
    recorder._push_level(float("nan"))  # noqa: SLF001 非有限值直接忽略
    recorder._push_level(float("inf"))  # noqa: SLF001
    levels = recorder.recent_levels()
    assert levels == [1.0, 0.0]
    assert all(0.0 <= value <= 1.0 for value in levels)


def test_recent_levels_maxlen_and_reset(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path / "recordings")
    for index in range(LEVEL_HISTORY_MAX + 60):
        recorder._push_level((index % 100) / 100.0)  # noqa: SLF001
    levels = recorder.recent_levels(10_000)
    assert len(levels) == LEVEL_HISTORY_MAX
    assert levels[-1] == pytest.approx(((LEVEL_HISTORY_MAX + 59) % 100) / 100.0)

    recorder._reset_level_history()  # noqa: SLF001 start() 的内部清空逻辑
    assert recorder.recent_levels() == []
    assert recorder.level == 0.0


def test_start_callback_populates_levels_and_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用假 sounddevice 验证「start 清空历史 → 回调写音量 → stop 保留历史 → 写 wav」。"""
    recorder = Recorder(tmp_path / "recordings")
    for _ in range(5):  # 上一轮录音残留的历史，start 时应被清空
        recorder._push_level(0.9)  # noqa: SLF001
    assert len(recorder.recent_levels()) == 5

    fake_sd = _FakeSoundDevice()
    monkeypatch.setattr(Recorder, "_import_sounddevice", staticmethod(lambda: fake_sd))
    recorder.start()
    try:
        assert recorder.is_recording is True
        # start 清空旧历史；假回调同步喂了一帧正弦波，所以只剩 1 个新值
        levels = recorder.recent_levels()
        assert len(levels) == 1
        assert 0.0 < levels[0] <= 1.0
        assert recorder.level == levels[0]
        assert recorder.elapsed_seconds > 0.0
        saved = recorder.stop()
    finally:
        if recorder.is_recording:  # 断言失败等异常路径下别把流晾着
            recorder.cancel()

    assert saved.is_file()
    with wave.open(str(saved), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == recorder.samplerate
        assert handle.getnframes() == 512
    # stop 后保留最后一批音量供界面定格，但 level 归零
    assert recorder.recent_levels() == levels
    assert recorder.level == 0.0
    assert recorder.is_recording is False


def test_cancel_clears_level_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder(tmp_path / "recordings")
    fake_sd = _FakeSoundDevice()
    monkeypatch.setattr(Recorder, "_import_sounddevice", staticmethod(lambda: fake_sd))
    recorder.start()
    assert recorder.recent_levels()  # 已有一帧音量
    recorder.cancel()
    assert recorder.recent_levels() == []
    assert recorder.level == 0.0


def test_recent_levels_with_real_microphone(tmp_path: Path) -> None:
    """真实麦克风短录 0.3s：level 与 recent_levels 应有真实波形数据。"""
    if not Recorder.available():
        pytest.skip("当前机器没有可用输入设备")
    recorder = Recorder(tmp_path / "recordings")
    try:
        recorder.start()
    except errors.AudioError as exc:
        pytest.skip(f"麦克风不可用：{exc}")
    try:
        time.sleep(0.3)
        assert recorder.is_recording is True
        live_level = recorder.level
        live_levels = list(recorder.recent_levels())
    finally:
        try:
            recorder.stop()
        except errors.AudioError as exc:  # 没有采到数据（静音/独占等环境问题）
            recorder.cancel()
            pytest.skip(f"麦克风未采集到数据：{exc}")

    final_levels = recorder.recent_levels()
    assert final_levels, "真实录音后应保留音量历史"
    assert len(final_levels) <= LEVEL_HISTORY_MAX
    assert all(0.0 <= value <= 1.0 for value in final_levels)
    assert live_levels, "录音中应能读到实时音量历史"
    # 真实麦克风应能采到非零幅度；整段全为数字静音时无法验证，按环境问题跳过
    if max(final_levels) <= 0.0:
        pytest.skip("麦克风采集到的是纯数字静音，无法验证非零波形")
    assert live_level >= 0.0
    assert max(live_levels) > 0.0
    assert recorder.level == 0.0  # 停止后电平归零
    print(
        f"\n[mic] 音量历史 {len(final_levels)} 个，"
        f"min={min(final_levels):.4f} max={max(final_levels):.4f} "
        f"mean={sum(final_levels)/len(final_levels):.4f}"
    )
    print("[mic] 最近 12 个音量值（旧→新）："
          + ", ".join(f"{value:.3f}" for value in final_levels[-12:]))


# ===========================================================================
# v1.2 GPU 支持：compute_type 解析 / CUDA DLL 目录 / 中文错误提示
# ===========================================================================
def test_is_cuda_device() -> None:
    assert is_cuda_device("cuda") is True
    assert is_cuda_device("CUDA") is True
    assert is_cuda_device("cuda:0") is True
    assert is_cuda_device(" cuda ") is True
    assert is_cuda_device("cpu") is False
    assert is_cuda_device(None) is False
    assert is_cuda_device("") is False


@pytest.mark.parametrize(
    ("device", "requested", "expected"),
    [
        ("cuda", "auto", "float16"),
        ("cuda", "", "float16"),
        ("cuda", None, "float16"),
        ("cuda", "AUTO", "float16"),
        ("cuda:0", "auto", "float16"),
        ("cpu", "auto", "int8"),
        ("cpu", "", "int8"),
        ("cpu", None, "int8"),
        ("cuda", "int8_float16", "int8_float16"),  # 显式值原样保留
        ("cuda", "int8", "int8"),
        ("cpu", "float32", "float32"),
        ("cpu", "int8", "int8"),
    ],
)
def test_resolve_compute_type(device: str, requested, expected: str) -> None:
    assert resolve_compute_type(device, requested) == expected


def test_whisper_asr_compute_type_auto_by_device() -> None:
    assert WhisperASR(device="cpu").compute_type == "int8"
    assert WhisperASR(device="cuda").compute_type == "float16"
    assert WhisperASR(device="cuda", compute_type="auto").compute_type == "float16"
    assert (
        WhisperASR(device="cuda", compute_type="int8_float16").compute_type
        == "int8_float16"
    )


def test_find_cuda_dll_directories_from_fake_site(tmp_path: Path) -> None:
    site_root = tmp_path / "site-packages"
    for component in ("cublas", "cudnn", "cuda_runtime"):
        (site_root / "nvidia" / component / "bin").mkdir(parents=True)
    (site_root / "nvidia" / "cublas" / "lib").mkdir()  # Linux 布局也应识别
    (site_root / "not_nvidia").mkdir()

    directories = find_cuda_dll_directories([site_root])

    relative = [f"{path.parent.name}/{path.name}" for path in directories]
    assert relative == [
        "cublas/bin",
        "cublas/lib",
        "cuda_runtime/bin",
        "cudnn/bin",
    ]
    for path in directories:
        assert path.is_dir()


def test_find_cuda_dll_directories_without_nvidia(tmp_path: Path) -> None:
    site_root = tmp_path / "site-packages"
    (site_root / "faster_whisper").mkdir(parents=True)
    assert find_cuda_dll_directories([site_root]) == []
    assert find_cuda_dll_directories([tmp_path / "不存在"]) == []


def test_ensure_cuda_dll_directories_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform != "win32":
        pytest.skip("os.add_dll_directory / PATH 注入仅在 Windows 生效")
    site_root = tmp_path / "site-packages"
    for component in ("cublas", "cudnn"):
        (site_root / "nvidia" / component / "bin").mkdir(parents=True)

    # 隔离模块级全局状态，并确保 PATH 在测试结束后恢复
    monkeypatch.setattr(asr_module, "_CONFIGURED_DLL_DIRS", set())
    monkeypatch.setattr(asr_module, "_DLL_DIR_HANDLES", [])
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    first = ensure_cuda_dll_directories([site_root])
    assert len(first) == 2
    handles_after_first = len(asr_module._DLL_DIR_HANDLES)
    assert handles_after_first == len(first)
    path_after_first = os.environ["PATH"]
    for directory in first:
        assert path_after_first.split(os.pathsep).count(str(directory)) == 1

    # 幂等：重复调用不重复注册 DLL 目录、不再改 PATH
    second = ensure_cuda_dll_directories([site_root])
    assert second == first
    assert len(asr_module._DLL_DIR_HANDLES) == handles_after_first
    assert os.environ["PATH"] == path_after_first


def test_prepare_cuda_environment_without_gpu_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = WhisperASR(device="cuda")
    monkeypatch.setattr(asr_module, "ensure_cuda_dll_directories", lambda *args, **kwargs: [])
    monkeypatch.setattr(asr_module, "get_cuda_device_count", lambda: 0)
    with pytest.raises(errors.ASRError) as excinfo:
        asr._prepare_cuda_environment()
    assert "GPU" in str(excinfo.value)
    assert "asr_device=cpu" in excinfo.value.user_hint


def test_prepare_cuda_environment_rejects_unsupported_compute_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr = WhisperASR(device="cuda", compute_type="int8")
    monkeypatch.setattr(asr_module, "ensure_cuda_dll_directories", lambda *args, **kwargs: [])
    monkeypatch.setattr(asr_module, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(asr_module, "get_cuda_compute_types", lambda: {"float16"})
    monkeypatch.setattr(asr_module, "get_gpu_name", lambda: "Fake GPU")
    with pytest.raises(errors.ASRError) as excinfo:
        asr._prepare_cuda_environment()
    assert "compute_type" in str(excinfo.value)
    assert "asr_device=cpu" in excinfo.value.user_hint


def test_cuda_error_messages_are_chinese_and_actionable() -> None:
    asr = WhisperASR(device="cuda")
    message = "Library cublas64_12.dll is not found or cannot be loaded"

    load_error = asr._wrap_load_error(RuntimeError(message), needs_download=False)  # noqa: SLF001
    assert isinstance(load_error, errors.ASRError)
    assert "GPU" in str(load_error)
    assert "asr_device=cpu" in load_error.user_hint

    transcribe_error = asr._wrap_transcribe_error(  # noqa: SLF001
        RuntimeError(message), Path("a.wav")
    )
    assert isinstance(transcribe_error, errors.ASRError)
    assert "GPU" in str(transcribe_error)
    assert "asr_device=cpu" in transcribe_error.user_hint

    # CPU 设备的普通模型错误不应被误判成 GPU 错误
    cpu_asr = WhisperASR(device="cpu")
    cpu_error = cpu_asr._wrap_load_error(  # noqa: SLF001
        RuntimeError("some unknown failure"), needs_download=False
    )
    assert "GPU" not in str(cpu_error)


# ===========================================================================
# 端到端测试：真实 faster-whisper tiny 模型转写中文 TTS 音频
# ===========================================================================
@pytest.mark.skipif(not RUN_ASR, reason="端到端转写需下载 tiny 模型，设置 VOXSCRIBE_RUN_ASR=1 启用")
def test_e2e_transcribe_tts_audio() -> None:
    audio = _ensure_tts_audio()
    E2E_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    asr = WhisperASR(
        model_size="tiny",
        compute_type="int8",
        cpu_threads=4,
        model_dir=str(E2E_MODEL_DIR),
        beam_size=5,
        vad_filter=True,
    )
    progress: List[Tuple[float, str]] = []
    memory_before = _memory_mb()
    try:
        transcript = asr.transcribe(
            audio,
            language="zh",
            on_progress=lambda ratio, message: progress.append((ratio, message)),
        )
        memory_after = _memory_mb()
    finally:
        asr.unload()
        memory_unloaded = _memory_mb()

    matched = [keyword for keyword in E2E_KEYWORDS if keyword in transcript.text]
    print(
        f"\n[e2e] 进程内存：加载前 {memory_before:.1f}MB / "
        f"加载并转写后 {memory_after:.1f}MB / 卸载后 {memory_unloaded:.1f}MB"
    )
    print(f"[e2e] 语言={transcript.language} 音频时长={transcript.duration:.2f}s "
          f"片段数={len(transcript.segments)} 命中关键词={matched}")
    print("[e2e] 转写文本：\n" + transcript.text)

    assert transcript.segments, "tiny 模型应识别出至少一个片段"
    assert transcript.duration > 5.0, f"音频时长异常：{transcript.duration}"
    assert transcript.language.startswith("zh"), f"语言检测异常：{transcript.language}"
    assert transcript.text.strip()
    assert len(matched) >= 2, (
        f"转写结果命中的关键词不足（{matched}），完整文本：{transcript.text!r}"
    )
    # 所有片段文本都已 strip，且纯空白段被丢弃
    for segment in transcript.segments:
        assert segment.text == segment.text.strip() and segment.text
        assert segment.end >= segment.start >= 0.0
    # 进度回调：0.0 起步、1.0 结束、不越界、消息非空
    ratios = [ratio for ratio, _ in progress]
    assert ratios[0] == 0.0 and ratios[-1] == 1.0
    assert all(0.0 <= ratio <= 1.0 for ratio in ratios)
    assert all(message for _, message in progress)
    # 卸载后状态准确，且模型已缓存到本地
    assert asr.loaded is False
    assert asr.is_model_available() is True

    print(f"[e2e] 内存净增（模型加载+转写）：{memory_after - memory_before:.1f}MB")


@pytest.mark.skipif(not RUN_ASR, reason="端到端转写需下载 tiny 模型，设置 VOXSCRIBE_RUN_ASR=1 启用")
def test_e2e_cached_model_reloads_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型已缓存时，第二次使用完全离线（不再访问 HuggingFace）。"""
    audio = _ensure_tts_audio()
    asr = WhisperASR(model_size="tiny", model_dir=str(E2E_MODEL_DIR))
    if not asr.is_model_available():
        pytest.skip("tiny 模型尚未缓存，跳过离线复用测试（先跑 test_e2e_transcribe_tts_audio）")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    try:
        transcript = asr.transcribe(audio, language="zh")
    finally:
        asr.unload()
    assert transcript.text.strip()
    assert any(keyword in transcript.text for keyword in E2E_KEYWORDS), (
        f"离线复用转写结果异常：{transcript.text!r}"
    )


@pytest.mark.skipif(
    not RUN_GPU, reason="GPU 端到端测试需本机 NVIDIA GPU，设置 VOXSCRIBE_RUN_GPU=1 启用"
)
def test_e2e_gpu_transcribe_and_model_reuse() -> None:
    """真实 GPU 转写：device=cuda / float16，同一实例第二次转写不重载模型。"""
    if not _cuda_available():
        pytest.skip("当前环境没有可用的 CUDA GPU（ctranslate2 未检测到）")
    audio = _ensure_tts_audio()

    asr = WhisperASR(
        model_size="tiny",
        device="cuda",
        compute_type="auto",
        model_dir=str(E2E_MODEL_DIR),
    )
    # 构造阶段就应解析出 float16，且不会静默回退 CPU
    assert asr.compute_type == "float16"
    assert asr.device == "cuda"
    assert asr.is_model_available(), "tiny 模型应已缓存到 data/models（先跑 ASR e2e）"

    try:
        started = time.monotonic()
        transcript = asr.transcribe(audio, language="zh")
        first_elapsed = time.monotonic() - started
        assert asr.loaded is True
        model_reference = asr._model  # noqa: SLF001

        assert transcript.text.strip()
        matched = [keyword for keyword in E2E_KEYWORDS if keyword in transcript.text]
        assert len(matched) >= 2, f"GPU 转写结果异常：{transcript.text!r}"

        started = time.monotonic()
        transcript_second = asr.transcribe(audio, language="zh")
        second_elapsed = time.monotonic() - started
        # 同一实例连续第二次转写：复用同一个模型对象，不重新加载
        assert asr._model is model_reference  # noqa: SLF001
        assert asr.loaded is True
        assert transcript_second.text.strip()
        assert second_elapsed < first_elapsed, (
            f"第二次转写应省去模型加载时间：first={first_elapsed:.2f}s "
            f"second={second_elapsed:.2f}s"
        )
        print(
            f"\n[GPU] tiny 首次（含加载）={first_elapsed:.2f}s，"
            f"第二次（纯转写）={second_elapsed:.2f}s，"
            f"compute_type={asr.compute_type}"
        )
        print("[GPU] 转写文本：\n" + transcript.text)
    finally:
        asr.unload()
    assert asr.loaded is False
