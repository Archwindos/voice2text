"""统一的异常类型。GUI / CLI 只捕获 ``VoxError`` 的子类即可分类展示。"""

from __future__ import annotations


class VoxError(Exception):
    """本应用所有可预期错误的基类。"""

    user_hint: str = ""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        if hint:
            self.user_hint = hint


class ConfigError(VoxError):
    """配置文件读写或内容错误。"""


class AudioError(VoxError):
    """音频文件不存在、损坏或不支持的格式。"""


class ASRError(VoxError):
    """语音识别失败（模型下载失败、解码失败等）。"""


class MissingAPIKeyError(VoxError):
    """未配置 DeepSeek API Key，无法进行文字整理。"""


class RefineError(VoxError):
    """调用 DeepSeek API 整理文字失败（网络、鉴权、限流等）。"""


class CancelledError(VoxError):
    """用户主动取消了当前任务。"""


__all__ = [
    "VoxError",
    "ConfigError",
    "AudioError",
    "ASRError",
    "MissingAPIKeyError",
    "RefineError",
    "CancelledError",
]
