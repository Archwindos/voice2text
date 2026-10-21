"""通用小工具：文件名清洗、时长格式化、打开文件夹、进程内存占用。"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path
from typing import Union

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def sanitize_filename(name: str, fallback: str = "未命名") -> str:
    """把任意标题清洗成合法的 Windows 文件名（不含扩展名）。"""
    cleaned = _INVALID_CHARS.sub("_", (name or "").strip())
    cleaned = cleaned.strip(" .")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return fallback
    if cleaned.lower() in _WINDOWS_RESERVED:
        cleaned += "_"
    return cleaned[:120]


def human_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds or 0))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分{secs:02d}秒"
    if minutes:
        return f"{minutes}分{secs:02d}秒"
    return f"{secs}秒"


def human_size(num_bytes: float) -> str:
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def reveal_path(path: Union[str, Path]) -> bool:
    """在资源管理器中选中文件（或其所在文件夹）。"""
    target = Path(path)
    try:
        if sys.platform == "win32":
            if target.exists() and target.is_file():
                os.system(f'explorer /select,"{target}"')
            else:
                os.startfile(str(target if target.is_dir() else target.parent))  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_memory_mb() -> float:
    """当前进程的内存占用（MB）；非 Windows 返回 0。"""
    if sys.platform != "win32":
        return 0.0
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.windll.kernel32
        process = kernel32.GetCurrentProcess()
        psapi = ctypes.windll.psapi
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if ok:
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


__all__ = [
    "sanitize_filename",
    "human_duration",
    "human_size",
    "reveal_path",
    "process_memory_mb",
]
