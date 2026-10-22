"""配置管理：读写 ``data/config.json``，API Key 使用 DPAPI 加密保存。"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import secure

log = logging.getLogger(__name__)

APP_NAME = "VoxScribe"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASR_MODELS: Dict[str, str] = {
    "tiny": "tiny（最快，75MB，准确率一般）",
    "base": "base（轻量，145MB，够用）",
    "small": "small（推荐，484MB，中文更好）",
    "medium": "medium（准确，1.5GB，较慢）",
    "large-v3": "large-v3（最准，3.1GB，很慢）",
}

LANGUAGES: Dict[str, str] = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "auto": "自动检测",
}


def data_dir() -> Path:
    """运行数据目录（配置、模型、录音、日志）。"""
    override = os.environ.get("VOXSCRIBE_DATA_DIR", "").strip()
    path = Path(override) if override else PROJECT_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    override = os.environ.get("VOXSCRIBE_CONFIG", "").strip()
    return Path(override) if override else data_dir() / "config.json"


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def recordings_dir() -> Path:
    path = data_dir() / "recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    path = PROJECT_ROOT / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class AppSettings:
    """全部可持久化设置。字段名即 config.json 的键名。"""

    # --- 语音识别（本地 faster-whisper，优先 GPU）---
    asr_model: str = "medium"
    asr_device: str = "cuda"  # cuda / cpu；用户要求默认走 GPU
    compute_type: str = "auto"  # auto = cuda→float16 / cpu→int8
    cpu_threads: int = 0  # 仅 CPU 模式使用；0 = 自动（最多 4 线程）
    language: str = "auto"  # auto / zh / en / ja / ko；自动检测可中英混说
    beam_size: int = 5
    vad_filter: bool = True
    model_dir: str = ""  # 空 = data/models
    input_device: str = ""  # 麦克风设备名；空 = 系统默认输入设备
    idle_unload_minutes: int = 0  # 0 = 常驻不卸载（保证口述响应速度）

    # --- DeepSeek 文本整理（deepseek-chat，非思考模式，低延迟）---
    api_base_url: str = "https://api.deepseek.com"
    api_model: str = "deepseek-chat"
    api_timeout: int = 120
    api_max_retries: int = 2
    api_key_enc: str = ""  # DPAPI 密文；兼容 plain: 明文

    # --- 默认输出 ---
    refine_by_default: bool = True
    default_style: str = "oral"
    export_formats: List[str] = field(default_factory=lambda: ["txt", "md"])
    glossary: str = ""
    output_dir: str = ""  # 空 = outputs/
    auto_copy: bool = True  # 语音输入完成后自动复制到剪贴板

    # --- 界面 ---
    window_geometry: str = ""
    first_run_done: bool = False
    close_to_tray: bool = True  # 点关闭/最小化时隐藏到系统托盘待机
    preload_model: bool = True  # 启动时后台预热识别模型，保证首句口述也够快

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppSettings":
        """读取配置；文件不存在或损坏时返回默认设置，绝不崩溃。"""
        target = Path(path) if path else config_path()
        if not target.exists():
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:  # 损坏文件：备份后使用默认值
            backup = target.with_suffix(target.suffix + ".broken")
            try:
                target.replace(backup)
            except Exception:
                pass
            log.warning("配置文件解析失败，已备份到 %s：%s", backup, exc)
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        clean = {key: value for key, value in raw.items() if key in known}
        try:
            return cls(**clean)
        except TypeError as exc:  # 字段类型极端错误
            log.warning("配置字段异常，已忽略：%s", exc)
            return cls()

    def save(self, path: Optional[Path] = None) -> Path:
        """原子写入配置文件（先写临时文件再替换）。"""
        target = Path(path) if path else config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        fd, tmp_name = tempfile.mkstemp(
            prefix=".config-", suffix=".json", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
        return target

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    # ------------------------------------------------------------------
    # API Key
    # ------------------------------------------------------------------
    @property
    def has_api_key(self) -> bool:
        return bool(self.get_api_key())

    def get_api_key(self) -> str:
        """优先读取本地加密配置，其次读取环境变量（方便临时验证）。"""
        stored = (self.api_key_enc or "").strip()
        if stored:
            if stored.startswith("dpapi:"):
                value = secure.unprotect(stored[6:])
                if value:
                    return value.strip()
            elif stored.startswith("plain:"):
                return stored[6:].strip()
            else:  # 旧版本遗留的明文
                return stored
        for name in ("VOXSCRIBE_API_KEY", "DEEPSEEK_API_KEY"):
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    def set_api_key(self, api_key: str) -> bool:
        """保存 API Key；返回是否使用了系统加密（False 表示降级为受限明文存储）。"""
        api_key = (api_key or "").strip()
        if not api_key:
            self.api_key_enc = ""
            return True
        encrypted = secure.protect(api_key)
        if encrypted:
            self.api_key_enc = "dpapi:" + encrypted
            return True
        self.api_key_enc = "plain:" + api_key
        return False

    # ------------------------------------------------------------------
    # 资源与路径
    # ------------------------------------------------------------------
    def effective_threads(self) -> int:
        """实际使用的 CPU 线程数：默认最多 4 线程。"""
        cpu_count = os.cpu_count() or 4
        if self.cpu_threads and self.cpu_threads > 0:
            return max(1, min(int(self.cpu_threads), cpu_count))
        return max(1, min(4, cpu_count - 1))

    def effective_device(self) -> str:
        """识别设备：cuda / cpu（默认 GPU）。"""
        value = (self.asr_device or "").strip().lower()
        return "cuda" if value in ("cuda", "gpu", "auto", "") else "cpu"

    def effective_compute_type(self) -> str:
        """计算精度：auto 时 GPU 用 float16、CPU 用 int8。"""
        requested = (self.compute_type or "").strip().lower()
        if requested in ("", "auto"):
            return "float16" if self.effective_device() == "cuda" else "int8"
        return requested

    def resolve_model_dir(self) -> Path:
        return Path(self.model_dir).expanduser() if self.model_dir.strip() else models_dir()

    def resolve_output_dir(self) -> Path:
        path = (
            Path(self.output_dir).expanduser()
            if self.output_dir.strip()
            else default_output_dir()
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def idle_unload_seconds(self) -> int:
        return max(0, int(self.idle_unload_minutes)) * 60


__all__ = [
    "APP_NAME",
    "PROJECT_ROOT",
    "ASR_MODELS",
    "LANGUAGES",
    "AppSettings",
    "data_dir",
    "config_path",
    "logs_dir",
    "recordings_dir",
    "models_dir",
    "default_output_dir",
]
