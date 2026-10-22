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

实现要点：
* 懒加载：首次 transcribe 时才 import faster_whisper 并创建 WhisperModel；
  同一实例复用。若本地缓存已有模型，则直接使用缓存快照目录，完全不联网。
* 进度回调：模型下载期间由后台线程监控缓存目录体积并更新消息；转写期间
  按 ``segment.end / info.duration`` 估算比例。
* 取消：每个 segment 之间（以及模型加载前后）检查 ``cancel.is_set()``，
  取消时抛出 ``voxscribe.errors.CancelledError``，绝不被包装成 ASRError。
* 任何底层异常包装为 ``voxscribe.errors.ASRError``（给出中文提示与 hint）。
* 默认 CPU int8；不引入 torch；解码交给 faster-whisper 自带的 PyAV。
* 中文场景默认加 ``initial_prompt``（见 :data:`ZH_INITIAL_PROMPT`）：
  实测 tiny 模型在中文 TTS 上不加提示会输出繁体（「那個」「項目」「禁讀」），
  加上提示后输出简体且识别更准（「项目」「进度」「两周」），速度不变。
* v1.2 GPU 支持：``device="cuda"`` 时加载前自动把 pip 安装的 NVIDIA CUDA
  运行库目录（``site-packages/nvidia/*/bin``）加入 DLL 搜索路径；``compute_type``
  支持 ``"auto"``（GPU→float16，CPU→int8）。GPU 加载失败会抛 :class:`ASRError`
  并提示如何回退 CPU，**绝不静默降级到 CPU**（产品要求必须用 GPU）。
* 不向 stdout 打印；日志走 ``logging``，用户可见提示走 ``on_progress``。

已知限制：模型首次下载/加载无法从外部打断（faster-whisper 自身不支持），
取消只在加载前后与 segment 之间生效。
"""

from __future__ import annotations

import gc
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import errors
from .config import models_dir
from .models import Segment, Transcript
from .util import human_duration, human_size

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

#: 展示用模型体积（与 CTranslate2 int8 权重实际下载量基本一致）
MODEL_SIZE_HINTS: Dict[str, str] = {
    "tiny": "75MB",
    "base": "145MB",
    "small": "484MB",
    "medium": "1.5GB",
    "large": "3.1GB",
    "large-v1": "3.1GB",
    "large-v2": "3.1GB",
    "large-v3": "3.1GB",
    "large-v3-turbo": "1.6GB",
    "turbo": "1.6GB",
}

#: 模型规格 -> HuggingFace 仓库（与 faster_whisper.utils._MODELS 保持一致）。
#: 这里只保留配置界面允许选择的规格，避免 import faster_whisper（懒加载）。
_REPO_IDS: Dict[str, str] = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

#: 中文场景的 initial_prompt。实测能显著改善 Whisper 的中文输出：
#: 一是把繁体输出拉回简体，二是减少「进度 / 项目」这类会议常用词的同音错字。
ZH_INITIAL_PROMPT = "以下是普通话的会议记录。"

#: 下载监控线程刷新间隔（秒）
_DOWNLOAD_POLL_SECONDS = 0.8
#: 转写进度最多报到 0.99，1.0 留给「转写完成」消息
_MAX_RUNNING_RATIO = 0.99
#: 异常信息在中文提示里最多保留的字符数，避免 GUI 提示过长
_MAX_ERROR_TEXT = 180


def _repo_id_for(model_size: str) -> Optional[str]:
    """把模型规格换算成 HuggingFace 仓库 ID；无法识别时返回 None。"""
    size = (model_size or "").strip()
    if not size:
        return None
    if "/" in size or "\\" in size:
        # 已经是仓库 ID（org/name）或本地路径：本地路径在别处处理
        return size.replace("\\", "/") if "/" in size else None
    return _REPO_IDS.get(size.lower())


def _cache_dir_name(repo_id: str) -> str:
    """HuggingFace 缓存目录名：``Systran/xxx`` -> ``models--Systran--xxx``。"""
    return "models--" + repo_id.replace("/", "--")


def _normalized_size(model_size: str) -> str:
    """``small.en`` -> ``small``；``Systran/faster-whisper-small`` -> ``small``。"""
    size = (model_size or "").strip().lower()
    if "/" in size:
        size = size.rsplit("/", 1)[-1]
        if size.startswith("faster-whisper-"):
            size = size[len("faster-whisper-"):]
        elif size.startswith("faster-distil-whisper-"):
            size = size[len("faster-distil-whisper-"):]
    if size.endswith(".en"):
        size = size[:-3]
    return size


def _short_error(exc: BaseException) -> str:
    """把异常压缩成一行短文本，用于中文提示。"""
    text = re.sub(r"\s+", " ", str(exc) or "").strip()
    if len(text) > _MAX_ERROR_TEXT:
        text = text[:_MAX_ERROR_TEXT] + "…"
    return text or exc.__class__.__name__


def _exception_chain_text(exc: BaseException, *, depth: int = 4) -> str:
    """拼出异常链上的名称与信息，便于用关键字分类网络/磁盘/内存错误。"""
    parts: List[str] = []
    current: Optional[BaseException] = exc
    seen = 0
    while current is not None and seen < depth:
        parts.append(current.__class__.__name__)
        parts.append(str(current))
        current = current.__cause__ or current.__context__
        seen += 1
    return " ".join(parts).lower()


def _is_network_error(exc: BaseException) -> bool:
    chain = _exception_chain_text(exc)
    keywords = (
        "connection",
        "connect",
        "timeout",
        "timed out",
        "network",
        "socket",
        "ssl",
        "proxy",
        "certificate",
        "name or service not known",
        "temporary failure",
        "max retries",
        "offline",
        "huggingface",
        "hf_hub",
        "download",
        "本地缓存",
        "无法获取",
    )
    return any(word in chain for word in keywords)


def _is_disk_full_error(exc: BaseException) -> bool:
    chain = _exception_chain_text(exc)
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:  # ENOSPC
        return True
    return "no space left" in chain or "disk full" in chain or "空间不足" in chain


def _is_memory_error(exc: BaseException) -> bool:
    chain = _exception_chain_text(exc)
    return "memoryerror" in chain or "out of memory" in chain or "bad_alloc" in chain


def _is_decode_error(exc: BaseException) -> bool:
    chain = _exception_chain_text(exc)
    keywords = (
        "invalid data",
        "invalid_data",
        "could not open",
        "could not decode",
        "unknown format",
        "unsupported",
        "failed to open",
        "decoding",
        "decode",
        "no audio",
        "audio stream",
        "eof",
    )
    return any(word in chain for word in keywords)


# ---------------------------------------------------------------------------
# GPU / CUDA 支持（v1.2）
# ---------------------------------------------------------------------------
#: os.add_dll_directory 返回的句柄必须保持引用，否则目录会被提前移除
_DLL_DIR_HANDLES: List[object] = []
#: 已经加入过 DLL 搜索路径的目录（小写绝对路径），保证幂等
_CONFIGURED_DLL_DIRS: set[str] = set()
#: CUDA 相关错误关键字，用于把底层异常翻译成中文提示
_CUDA_ERROR_KEYWORDS = (
    "cuda",
    "cublas",
    "cudnn",
    "cudart",
    "cusparse",
    "cufft",
    "nvidia",
    "gpu",
    "dll",
    "driver",
    "no kernel image",
    "out of memory",
    "compute capability",
)


def is_cuda_device(device: Optional[str]) -> bool:
    """``device`` 是否为 CUDA（支持 ``"cuda"`` / ``"cuda:0"`` 写法）。"""
    return str(device or "").strip().lower().startswith("cuda")


def resolve_compute_type(device: Optional[str], requested: Optional[str]) -> str:
    """解析实际使用的 ``compute_type``。

    * GPU（device="cuda"）且请求 auto / 空 → ``"float16"``（RTX 显卡速度/精度最佳）；
    * CPU 且请求 auto / 空 → ``"int8"``（轻量、内存占用低）；
    * 显式指定（如 ``"int8_float16"``）→ 原样返回。

    :param device: ``"cuda"`` / ``"cpu"``（也接受 ``"cuda:0"``）。
    :param requested: 用户/配置请求的精度，可为 ``None`` / ``""`` / ``"auto"``。
    """
    requested_clean = str(requested or "").strip()
    if requested_clean.lower() in ("", "auto", "default"):
        return "float16" if is_cuda_device(device) else "int8"
    return requested_clean


def _default_site_package_roots() -> List[Path]:
    """当前 Python 环境的 site-packages 根目录（去重，保持存在性检查）。"""
    candidates: List[str] = []
    try:
        import site as site_module  # 标准库

        candidates.extend(site_module.getsitepackages())
        try:
            candidates.append(site_module.getusersitepackages())
        except Exception:  # pragma: no cover - 极端环境
            pass
    except Exception:  # pragma: no cover - 极少数嵌入式环境
        log.debug("无法通过 site.getsitepackages() 定位依赖目录", exc_info=True)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates.extend(
        [
            str(Path(sys.prefix) / "Lib" / "site-packages"),  # Windows venv
            str(Path(sys.prefix) / "lib" / version / "site-packages"),  # Linux/macOS
        ]
    )
    # 桌面快捷方式用基础 pythonw 启动（sys.prefix 指向基础解释器）时，
    # 依赖目录由 VIRTUAL_ENV / PYTHONPATH 注入，这里显式补上。
    venv_dir = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv_dir:
        candidates.extend(
            [
                str(Path(venv_dir) / "Lib" / "site-packages"),
                str(Path(venv_dir) / "lib" / version / "site-packages"),
            ]
        )
    for entry in sys.path:
        try:
            path = Path(entry)
            if path.name.lower() == "site-packages":
                candidates.append(str(path))
        except (OSError, TypeError):
            continue
    roots: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            path = Path(candidate)
            key = str(path.resolve())
            if key in seen or not path.is_dir():
                continue
        except OSError:
            continue
        seen.add(key)
        roots.append(path)
    return roots


def find_cuda_dll_directories(
    site_roots: Optional[List[Path]] = None,
) -> List[Path]:
    """发现 pip 安装的 NVIDIA CUDA 运行库 DLL 目录。

    新版 CUDA 相关 pip 包（``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` 等）
    会把 DLL 放在 ``site-packages/nvidia/<组件>/bin``（Windows）或
    ``site-packages/nvidia/<组件>/lib``（Linux）下，这些目录默认不在 DLL
    搜索路径里，需要手动 ``os.add_dll_directory`` / 加入 PATH。

    :param site_roots: 用于测试的 site-packages 根目录列表；``None`` 时自动探测。
    :return: 找到的 DLL 目录（排序、去重，路径不存在时返回空列表）。
    """
    roots = [Path(root) for root in site_roots] if site_roots is not None else _default_site_package_roots()
    directories: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        nvidia_root = root / "nvidia"
        try:
            if not nvidia_root.is_dir():
                continue
            components = list(nvidia_root.iterdir())
        except OSError:
            continue
        for component in components:
            for sub_name in ("bin", "lib"):
                dll_dir = component / sub_name
                try:
                    if not dll_dir.is_dir():
                        continue
                    key = str(dll_dir.resolve())
                except OSError:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                directories.append(dll_dir)
    directories.sort(key=lambda path: str(path).lower())
    return directories


def ensure_cuda_dll_directories(
    site_roots: Optional[List[Path]] = None,
) -> List[Path]:
    """把 NVIDIA CUDA DLL 目录加入 Windows DLL 搜索路径（幂等）。

    对每个目录调用一次 ``os.add_dll_directory``（句柄保存在模块变量里防止被回收），
    并把目录加入 ``PATH``，让 ctranslate2 加载 ``cublas64_12.dll`` /
    ``cudnn64_9.dll`` 等运行库时能够找到。非 Windows 平台只返回目录不做处理。

    :param site_roots: 用于测试的 site-packages 根目录列表；``None`` 时自动探测。
    :return: 发现的 DLL 目录列表（即使已经配置过也会返回，方便日志展示）。
    """
    directories = find_cuda_dll_directories(site_roots)
    if sys.platform != "win32" or not directories:
        return directories

    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    normalized_path = {os.path.normcase(part) for part in path_parts}
    changed = False
    for directory in directories:
        key = str(directory).lower()
        if key not in _CONFIGURED_DLL_DIRS:
            try:
                handle = os.add_dll_directory(str(directory))
                _DLL_DIR_HANDLES.append(handle)
            except (AttributeError, OSError):
                log.debug("加入 DLL 搜索目录失败：%s", directory, exc_info=True)
            _CONFIGURED_DLL_DIRS.add(key)
        normalized_dir = os.path.normcase(str(directory))
        if normalized_dir not in normalized_path:
            path_parts.insert(0, str(directory))
            normalized_path.add(normalized_dir)
            changed = True
    if changed:
        os.environ["PATH"] = os.pathsep.join(path_parts)
    return directories


def get_cuda_device_count() -> int:
    """CUDA 可用设备数（不 import faster-whisper / torch；失败返回 0）。"""
    try:
        import ctranslate2  # noqa: PLC0415  只有用 GPU 时才 import
    except Exception:
        return 0
    try:
        return max(0, int(ctranslate2.get_cuda_device_count()))
    except Exception:
        return 0


def get_cuda_compute_types() -> set[str]:
    """当前 GPU 支持的 compute_type 集合（失败返回空集合）。"""
    try:
        import ctranslate2  # noqa: PLC0415

        return {str(item).lower() for item in ctranslate2.get_supported_compute_types("cuda")}
    except Exception:
        return set()


def get_gpu_name() -> str:
    """通过 nvidia-smi 查询 GPU 名称（仅用于日志；失败返回空串，不弹窗口）。"""
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=creation_flags,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    lines = [line.strip() for line in result.stdout.decode("utf-8", "replace").splitlines() if line.strip()]
    return lines[0] if lines else ""


class WhisperASR:
    """faster-whisper 的懒加载封装。"""

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "auto",
        cpu_threads: int = 4,
        model_dir: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: Optional[str] = None,
    ) -> None:
        self.model_size = (model_size or "small").strip() or "small"
        self.device = (device or "cpu").strip() or "cpu"
        #: 用户/配置请求的精度；"auto" 表示按设备自动选择（GPU→float16，CPU→int8）
        self.requested_compute_type = (compute_type or "auto").strip() or "auto"
        #: 实际传给 CTranslate2 的 compute_type（解析后）
        self.compute_type = resolve_compute_type(self.device, self.requested_compute_type)
        self.model_dir: Path = (
            Path(model_dir).expanduser() if model_dir else models_dir()
        )
        cpu_count = os.cpu_count() or 4
        raw_threads = int(cpu_threads) if cpu_threads else 0
        if raw_threads > 0:
            self.cpu_threads = max(1, min(raw_threads, cpu_count))
        else:
            # 与 config.AppSettings.effective_threads 一致：默认最多 4 线程
            self.cpu_threads = max(1, min(4, cpu_count - 1))
        self.beam_size = max(1, int(beam_size or 5))
        self.vad_filter = bool(vad_filter)
        #: 透传给 Whisper 的提示词；None 表示中文自动用内置提示、其他语言不用。
        self.initial_prompt = initial_prompt

        self._model: Optional[object] = None
        self._lock = threading.RLock()
        #: 最近一次检测到的 GPU 名称（仅用于日志展示）
        self._gpu_name = ""

    # ------------------------------------------------------------------
    # 状态与本地缓存
    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        """模型是否已经加载在内存中。"""
        return self._model is not None

    def is_model_available(self) -> bool:
        """本地 ``model_dir`` 里是否已缓存对应模型（只查目录，不联网）。"""
        return self._cached_model_path() is not None

    def model_size_hint(self) -> str:
        """展示用的模型体积，例如 ``"484MB"``；未知规格返回 ``"未知大小"``。"""
        size = _normalized_size(self.model_size)
        hint = MODEL_SIZE_HINTS.get(size)
        if hint:
            return hint
        if self._local_model_path() is not None:
            return "本地模型"
        return "未知大小"

    def _local_model_path(self) -> Optional[Path]:
        """``model_size`` 本身就是一个本地模型目录时返回该目录。"""
        candidate = Path(self.model_size).expanduser()
        try:
            if candidate.is_dir() and (candidate / "model.bin").is_file():
                return candidate
        except OSError:  # 非法路径字符等
            return None
        return None

    def _expected_repo_id(self) -> Optional[str]:
        return _repo_id_for(self.model_size)

    def _cache_candidates(self) -> List[Path]:
        """可能存放该模型的所有 HuggingFace 缓存目录。"""
        candidates: List[Path] = []
        repo_id = self._expected_repo_id()
        if repo_id:
            candidates.append(self.model_dir / _cache_dir_name(repo_id))
        # 兜底：扫描已有缓存目录，按 ``faster-whisper-<size>`` 后缀匹配，
        # 兼容自定义仓库 ID 或手动拷入的缓存。
        size = _normalized_size(self.model_size)
        if size and self.model_dir.is_dir():
            for child in self.model_dir.iterdir():
                try:
                    if not child.is_dir() or not child.name.startswith("models--"):
                        continue
                except OSError:
                    continue
                repo_name = child.name.split("--")[-1].lower()
                if repo_name.endswith("faster-whisper-" + size) or repo_name in (
                    "faster-distil-whisper-" + size,
                ):
                    if child not in candidates:
                        candidates.append(child)
        return candidates

    def _cached_model_path(self) -> Optional[Path]:
        """在本地缓存中找出完整可用（含 model.bin/config.json）的快照目录。"""
        local = self._local_model_path()
        if local is not None:
            return local
        newest: Optional[Tuple[float, Path]] = None
        for cache_dir in self._cache_candidates():
            snapshots = cache_dir / "snapshots"
            try:
                snapshot_dirs = list(snapshots.iterdir()) if snapshots.is_dir() else []
            except OSError:
                continue
            for snapshot in snapshot_dirs:
                try:
                    if not snapshot.is_dir():
                        continue
                    if not (snapshot / "model.bin").is_file():
                        continue
                    if not (snapshot / "config.json").is_file():
                        continue
                    mtime = snapshot.stat().st_mtime
                except OSError:
                    continue
                if newest is None or mtime > newest[0]:
                    newest = (mtime, snapshot)
        return newest[1] if newest else None

    def _download_target_dir(self) -> Path:
        """下载监控关注的目标目录（尽量精确到模型缓存目录）。"""
        repo_id = self._expected_repo_id()
        if repo_id:
            return self.model_dir / _cache_dir_name(repo_id)
        return self.model_dir

    # ------------------------------------------------------------------
    # 转写
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = "zh",
        on_progress: Optional[ProgressCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> Transcript:
        """把音频转写成 :class:`Transcript`。

        :param audio_path: 音频文件路径（wav/mp3/m4a/flac 等，PyAV 可解码即可）。
        :param language: ``"zh"`` 等语言代码；``None`` 表示自动检测。
        :param on_progress: ``callable(ratio, message)``，ratio 在 ``0.0~1.0``。
        :param cancel: 取消事件；每隔一个 segment 检查一次。
        :raises voxscribe.errors.CancelledError: 用户取消。
        :raises voxscribe.errors.ASRError: 模型下载/加载失败或音频解码失败。
        """
        cancel = cancel or threading.Event()
        if cancel.is_set():
            raise errors.CancelledError("任务已取消")

        model = self._ensure_model(on_progress, cancel)
        if cancel.is_set():
            raise errors.CancelledError("任务已取消")

        audio = Path(audio_path)
        prompt = self._resolve_initial_prompt(language)
        try:
            raw_segments, info = model.transcribe(  # type: ignore[attr-defined]
                str(audio),
                language=language or None,
                task="transcribe",
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=False,
                condition_on_previous_text=True,
                initial_prompt=prompt,
                log_progress=False,
            )
        except errors.VoxError:
            raise
        except Exception as exc:  # 解码 / VAD / 语言检测失败
            raise self._wrap_transcribe_error(exc, audio) from exc

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        detected_language = getattr(info, "language", None) or language or "zh"
        log.info(
            "开始转写 %s：时长 %.1f 秒，语言 %s，模型 %s",
            audio.name,
            duration,
            detected_language,
            self.model_size,
        )
        self._call_progress(
            on_progress,
            0.0,
            f"开始转写，音频时长 {human_duration(duration)}…" if duration else "开始转写…",
        )

        segments: List[Segment] = []
        try:
            for raw in raw_segments:
                if cancel.is_set():
                    raise errors.CancelledError("任务已取消")
                text = str(getattr(raw, "text", "") or "").strip()
                start = float(getattr(raw, "start", 0.0) or 0.0)
                end = float(getattr(raw, "end", start) or start)
                if end < start:
                    end = start
                if text:
                    segments.append(Segment(start=start, end=end, text=text))
                if duration > 0:
                    ratio = min(_MAX_RUNNING_RATIO, max(0.0, end / duration))
                    self._call_progress(
                        on_progress,
                        ratio,
                        f"正在转写… 已完成 {human_duration(end)} / 共 "
                        f"{human_duration(duration)}（{len(segments)} 段）",
                    )
        except errors.VoxError:
            raise
        except Exception as exc:  # 迭代解码过程中的底层错误
            raise self._wrap_transcribe_error(exc, audio) from exc

        transcript = Transcript(
            segments=segments,
            language=str(detected_language),
            duration=duration,
        )
        log.info(
            "转写完成 %s：%d 个片段，%d 个字符，用时语言 %s",
            audio.name,
            len(segments),
            len(transcript.text),
            detected_language,
        )
        self._call_progress(
            on_progress,
            1.0,
            f"转写完成，共 {len(segments)} 个片段",
        )
        return transcript

    # ------------------------------------------------------------------
    # 模型加载 / 卸载
    # ------------------------------------------------------------------
    def _ensure_model(
        self,
        on_progress: Optional[ProgressCallback],
        cancel: threading.Event,
    ) -> object:
        """返回已加载的 WhisperModel；首次调用时加载（必要时先下载）。"""
        with self._lock:
            if self._model is not None:
                return self._model
            if cancel.is_set():
                raise errors.CancelledError("任务已取消")

            # 1) GPU 设备：必须先把 pip 安装的 CUDA 运行库目录加入 DLL 搜索路径，
            #    然后再 import faster_whisper / ctranslate2，否则 Windows 上
            #    加载 cublas64_12.dll / cudnn64_9.dll 会失败（顺序很关键）。
            if is_cuda_device(self.device):
                self._call_progress(on_progress, 0.0, "正在准备 GPU 识别环境…")
                self._prepare_cuda_environment()

            # 2) 懒 import faster-whisper（启动时不 import，不拖慢冷启动）
            try:
                import faster_whisper  # noqa: PLC0415
            except Exception as exc:  # 缺依赖
                raise errors.ASRError(
                    "缺少语音识别组件（faster-whisper），无法转写",
                    hint="请重新运行安装脚本（setup.bat / pip install -e .）后再试。",
                ) from exc

            cached = self._cached_model_path()
            needs_download = cached is None
            monitor: Optional[Tuple[threading.Thread, threading.Event]] = None
            if needs_download:
                try:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise errors.ASRError(
                        f"无法创建模型目录：{self.model_dir}",
                        hint="请在「设置」里换一个可写的模型目录，或检查磁盘权限。",
                    ) from exc
                self._call_progress(
                    on_progress,
                    0.0,
                    f"首次使用，正在下载识别模型（约 {self.model_size_hint()}）…",
                )
                monitor = self._start_download_monitor(on_progress)
            else:
                device_label = "GPU" if is_cuda_device(self.device) else "CPU"
                self._call_progress(
                    on_progress,
                    0.0,
                    f"正在加载本地识别模型（{self.model_size}，{device_label}）…",
                )

            try:
                model = self._create_model(faster_whisper, cached)
            except errors.VoxError:
                raise
            except Exception as exc:
                raise self._wrap_load_error(exc, needs_download=needs_download) from exc
            finally:
                self._stop_monitor(monitor)

            if is_cuda_device(self.device):
                # CUDA 运行库是懒加载的：构造 WhisperModel 成功不代表 GPU 真能用。
                # 先做一次最小推理，把缺 cublas/cuDNN、显存不足等问题在「加载阶段」
                # 就暴露成 ASRError（而不是转写做到一半才失败）。
                try:
                    self._warmup_cuda_model(model)
                except errors.VoxError:
                    del model
                    gc.collect()
                    raise
                except Exception as exc:
                    del model
                    gc.collect()
                    raise self._wrap_load_error(exc, needs_download=False) from exc

            self._model = model
            if is_cuda_device(self.device):
                log.info(
                    "识别模型 %s 已加载到 GPU（device=%s，compute_type=%s，"
                    "GPU=%s，CUDA 设备数=%d）",
                    self.model_size,
                    self.device,
                    self.compute_type,
                    self._gpu_name or "未知",
                    get_cuda_device_count(),
                )
            else:
                log.info(
                    "识别模型 %s 已加载（device=%s，compute_type=%s，%d 线程）",
                    self.model_size,
                    self.device,
                    self.compute_type,
                    self.cpu_threads,
                )
            if cancel.is_set():
                raise errors.CancelledError("任务已取消")
            self._call_progress(on_progress, 0.0, "识别模型已就绪，开始转写…")
            return model

    def _warmup_cuda_model(self, model: object) -> None:
        """GPU 加载后做一次最小推理，提前暴露 CUDA 运行库 / 显存问题。"""
        try:
            import numpy as np  # faster-whisper 依赖，无需新装
        except Exception:  # pragma: no cover - numpy 缺失时跳过预热
            log.debug("numpy 不可用，跳过 GPU 预热", exc_info=True)
            return
        n_mels = 80
        try:
            n_mels = int(getattr(getattr(model, "model", None), "n_mels", 80) or 80)
        except Exception:
            pass
        features = np.zeros((1, n_mels, 3000), dtype=np.float32)
        started = time.monotonic()
        model.encode(features)  # type: ignore[attr-defined]  触发一次真实的 GPU 推理
        log.info("GPU 预热完成（%.0f ms）", (time.monotonic() - started) * 1000)

    def _prepare_cuda_environment(self) -> None:
        """GPU 加载前置检查：注入 CUDA DLL 目录、确认 GPU 与 compute_type 可用。

        只做检测与准备，不修改用户设置；任何问题都抛 :class:`ASRError`
        （中文提示如何安装 CUDA 依赖或回退 CPU），不会静默降级到 CPU。
        """
        directories = ensure_cuda_dll_directories()
        if directories:
            log.info(
                "已加入 %d 个 NVIDIA CUDA DLL 目录：%s",
                len(directories),
                "; ".join(str(path) for path in directories),
            )
        else:
            log.info("未发现 pip 安装的 NVIDIA CUDA 运行库目录，尝试使用系统 CUDA 库")

        try:
            import ctranslate2  # noqa: PLC0415
        except Exception as exc:
            raise errors.ASRError(
                "缺少 GPU 识别所需组件（ctranslate2 / CUDA）",
                hint="请重新安装依赖（setup.bat）；如需先用 CPU，可在设置里把"
                "识别设备改为 CPU（asr_device=cpu）。",
            ) from exc

        device_count = get_cuda_device_count()
        if device_count <= 0:
            raise errors.ASRError(
                "未检测到可用的 NVIDIA GPU，无法使用 GPU 转写",
                hint="请确认 NVIDIA 显卡驱动已安装且显卡未被禁用；"
                "如需改用 CPU，请在设置里把识别设备改为 CPU（asr_device=cpu）。",
            )

        self._gpu_name = get_gpu_name()
        supported = get_cuda_compute_types()
        requested = self.compute_type.lower()
        if supported and requested not in supported:
            raise errors.ASRError(
                f"当前 GPU 不支持 compute_type={self.compute_type}",
                hint="可把计算精度设置为 auto（GPU 自动使用 float16），"
                "或选择 int8_float16；也可以在设置里把识别设备改为 CPU（asr_device=cpu）。",
            )
        log.info(
            "GPU 环境检查通过：device=%s，compute_type=%s，GPU=%s（%d 张），支持精度=%s",
            self.device,
            self.compute_type,
            self._gpu_name or "未知",
            device_count,
            ", ".join(sorted(supported)) or "未知",
        )

    def _create_model(self, faster_whisper_module, cached: Optional[Path]) -> object:
        """真正创建 WhisperModel；cached 为本地快照目录时完全不联网。"""
        common_kwargs = dict(
            device=self.device,
            device_index=0,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=str(self.model_dir),
        )
        if cached is not None:
            return faster_whisper_module.WhisperModel(str(cached), **common_kwargs)
        if self._local_model_path() is not None:
            return faster_whisper_module.WhisperModel(
                str(self._local_model_path()), **common_kwargs
            )
        return faster_whisper_module.WhisperModel(self.model_size, **common_kwargs)

    def _start_download_monitor(
        self, on_progress: Optional[ProgressCallback]
    ) -> Tuple[threading.Thread, threading.Event]:
        """下载期间用后台线程观察缓存目录体积，更新「已下载」提示。"""
        stop = threading.Event()
        target = self._download_target_dir()
        expected_hint = self.model_size_hint()

        def _loop() -> None:
            last_size = -1
            while not stop.wait(_DOWNLOAD_POLL_SECONDS):
                try:
                    size = self._dir_size(target)
                    if size <= 0 or size - last_size < 256 * 1024:
                        continue
                    last_size = size
                    self._call_progress(
                        on_progress,
                        0.0,
                        f"正在下载识别模型（约 {expected_hint}），"
                        f"已下载 {human_size(size)}…",
                    )
                except errors.VoxError:
                    # on_progress 可能因取消而抛 CancelledError：结束监控即可
                    return
                except Exception:  # 监控失败绝不能影响下载主流程
                    log.debug("下载进度监控异常", exc_info=True)
            return

        thread = threading.Thread(
            target=_loop, name="voxscribe-download-monitor", daemon=True
        )
        thread.start()
        return thread, stop

    @staticmethod
    def _stop_monitor(monitor: Optional[Tuple[threading.Thread, threading.Event]]) -> None:
        if monitor is None:
            return
        thread, stop = monitor
        stop.set()
        thread.join(timeout=2.0)

    @staticmethod
    def _dir_size(root: Path) -> int:
        """递归统计目录体积（跳过符号链接、按 inode 去重硬链接）。"""
        total = 0
        seen = set()
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as iterator:
                    for entry in iterator:
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            stat = entry.stat(follow_symlinks=False)
                            key: object = (
                                (stat.st_dev, stat.st_ino)
                                if stat.st_ino
                                else entry.path
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            total += stat.st_size
                        except OSError:
                            continue
            except OSError:
                continue
        return total

    def ensure_loaded(
        self,
        on_progress: Optional[ProgressCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> None:
        """预先加载识别模型（不转写），用于程序启动时预热，降低首句口述等待。

        与 :meth:`transcribe` 共用同一实例；加载失败抛 :class:`ASRError`。
        """
        cancel = cancel or threading.Event()
        if cancel.is_set():
            raise errors.CancelledError("任务已取消")
        self._ensure_model(on_progress, cancel)

    def unload(self) -> None:
        """卸载模型并触发垃圾回收；重复调用安全。"""
        with self._lock:
            model, self._model = self._model, None
        if model is None:
            return
        del model
        gc.collect()
        log.info("识别模型已卸载，内存已释放")

    # ------------------------------------------------------------------
    # 进度与异常
    # ------------------------------------------------------------------
    def _resolve_initial_prompt(self, language: Optional[str]) -> Optional[str]:
        """决定本次转写使用的提示词。

        ``initial_prompt`` 显式设置时优先；否则中文自动启用
        :data:`ZH_INITIAL_PROMPT`，其他语言/自动检测不启用（避免干扰语言识别）。
        """
        if self.initial_prompt is not None:
            return self.initial_prompt or None
        code = str(language or "").strip().lower()
        if code == "zh" or code.startswith("zh-") or code.startswith("zh_"):
            return ZH_INITIAL_PROMPT
        return None

    @staticmethod
    def _call_progress(
        on_progress: Optional[ProgressCallback],
        ratio: float,
        message: str,
    ) -> None:
        """调用进度回调；回调自身的异常（如取消）原样抛给调用方。"""
        if on_progress is None:
            return
        on_progress(max(0.0, min(1.0, float(ratio))), message)

    def _wrap_load_error(self, exc: BaseException, *, needs_download: bool) -> errors.ASRError:
        """把模型加载/下载异常包装成带中文提示的 ASRError。"""
        log.warning(
            "识别模型加载失败（%s）：%r",
            "下载阶段" if needs_download else "加载阶段",
            exc,
        )
        model_dir = self.model_dir
        if _is_memory_error(exc):
            return errors.ASRError(
                f"内存不足，无法加载识别模型（{self.model_size}）",
                hint="请换用更小的模型（tiny/base），或关闭其他占用内存的程序后重试。",
            )
        if _is_disk_full_error(exc):
            return errors.ASRError(
                "下载识别模型失败：磁盘空间不足",
                hint=f"模型约需 {self.model_size_hint()}，"
                f"请清理 {model_dir} 所在磁盘后重试。",
            )
        if isinstance(exc, PermissionError):
            return errors.ASRError(
                f"无法写入模型目录：{model_dir}",
                hint="请在「设置」里换一个可写的模型目录，或检查文件夹权限。",
            )
        if _is_network_error(exc):
            return errors.ASRError(
                "下载识别模型失败，请检查网络连接",
                hint="可在能访问 HuggingFace 的网络下重试；"
                "公司网络受限时可设置环境变量 HF_ENDPOINT=https://hf-mirror.com 后重启程序。",
            )
        if is_cuda_device(self.device):
            chain = _exception_chain_text(exc)
            if any(word in chain for word in _CUDA_ERROR_KEYWORDS):
                return errors.ASRError(
                    f"GPU（CUDA）加载识别模型失败：{_short_error(exc)}",
                    hint="请确认 NVIDIA 显卡驱动与 CUDA 运行库（cublas / cuDNN）已正确安装；"
                    "如果显存不足，可换用更小的模型（tiny/base）或关闭占用显存的程序；"
                    "如需改用 CPU，请在设置里把识别设备改为 CPU（asr_device=cpu）后重试。",
                )
        if isinstance(exc, ValueError) and "invalid model size" in str(exc).lower():
            return errors.ASRError(
                f"无法识别的模型规格：{self.model_size}",
                hint="请在「设置 → 识别模型」中选择 tiny / base / small / medium / large-v3。",
            )
        detail = _short_error(exc)
        if needs_download:
            return errors.ASRError(
                f"下载或加载识别模型失败：{detail}",
                hint="请检查网络与磁盘空间后重试；"
                f"如反复失败，可删除缓存目录 {model_dir} 里的模型文件后重新下载。",
            )
        return errors.ASRError(
            f"加载本地识别模型失败：{detail}",
            hint=f"模型缓存可能不完整，可删除 {model_dir} 下对应模型目录后重新下载。",
        )

    def _wrap_transcribe_error(
        self, exc: BaseException, audio: Path
    ) -> errors.ASRError:
        """把解码 / 转写过程中的异常包装成中文 ASRError。"""
        log.warning("转写 %s 失败：%r", audio, exc)
        chain = _exception_chain_text(exc)
        if is_cuda_device(self.device) and any(
            word in chain for word in _CUDA_ERROR_KEYWORDS
        ):
            return errors.ASRError(
                f"GPU（CUDA）转写失败：{_short_error(exc)}",
                hint="请确认 NVIDIA 显卡驱动与 CUDA 运行库（cublas / cuDNN）安装完整；"
                "如需改用 CPU，请在设置里把识别设备改为 CPU（asr_device=cpu）后重试。",
            )
        if isinstance(exc, FileNotFoundError) or "no such file" in chain:
            return errors.ASRError(
                f"找不到音频文件：{audio.name}",
                hint="请确认文件没有被移动或删除，然后重新选择文件。",
            )
        if isinstance(exc, PermissionError):
            return errors.ASRError(
                f"无法读取音频文件：{audio.name}",
                hint="文件可能被其他程序占用，请关闭占用程序或复制一份后重试。",
            )
        if _is_disk_full_error(exc):
            return errors.ASRError(
                "转写失败：磁盘空间不足",
                hint="请清理音频文件所在磁盘后重试。",
            )
        if _is_memory_error(exc):
            return errors.ASRError(
                "转写失败：内存不足",
                hint="请换用更小的识别模型（tiny/base），或关闭其他程序后重试。",
            )
        if _is_decode_error(exc):
            return errors.ASRError(
                f"无法解码音频文件：{audio.name}",
                hint="请确认文件是完整的音频（wav/mp3/m4a/flac 等），"
                "可以先用播放器打开确认能正常播放。",
            )
        if "no clip timestamps" in chain or "no speech" in chain:
            return errors.ASRError(
                f"音频中没有检测到清晰人声：{audio.name}",
                hint="请确认录音音量正常；如果确实没有语音，请换一段音频。",
            )
        return errors.ASRError(
            f"语音识别失败：{_short_error(exc)}",
            hint="可以重试；若反复失败，可尝试在设置中切换更小的模型，"
            "或删除 data/models 下对应模型缓存后重新下载。",
        )


__all__ = [
    "WhisperASR",
    "ProgressCallback",
    "MODEL_SIZE_HINTS",
    "ZH_INITIAL_PROMPT",
    "is_cuda_device",
    "resolve_compute_type",
    "find_cuda_dll_directories",
    "ensure_cuda_dll_directories",
    "get_cuda_device_count",
    "get_cuda_compute_types",
    "get_gpu_name",
]
