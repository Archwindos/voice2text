"""中文简繁字形统一。

产品目标：不管 ASR 原始输出、DeepSeek 润色结果来自哪里，中文最终都按用户
选择的字形输出（默认简体），避免"有时繁体、有时简体"。

实现：OpenCC 的 ``t2s`` / ``s2t`` 转换；依赖很小（约 1MB），单次转换 <1ms。
"""

from __future__ import annotations

import logging
from typing import Dict

log = logging.getLogger(__name__)

#: 设置项 ``chinese_script`` 的合法值 -> 界面文案
SCRIPTS: Dict[str, str] = {
    "simplified": "简体中文",
    "traditional": "繁体中文",
    "preserve": "保持原文",
}
DEFAULT_SCRIPT = "simplified"

_converters: Dict[str, object] = {}


def normalize_script_name(value: str) -> str:
    """把设置值规范化为合法字形；未知值回退简体。"""
    key = str(value or "").strip().lower()
    return key if key in SCRIPTS else DEFAULT_SCRIPT


def is_chinese_language(language: str) -> bool:
    """语言代码是否属于中文（zh / zh-CN / zh-TW / yue 等）。"""
    code = str(language or "").strip().lower().replace("_", "-")
    return code == "zh" or code.startswith("zh-") or code == "yue" or code.startswith("yue-")


def _converter(kind: str) -> object:
    """懒加载 OpenCC 转换器（t2s / s2t）。"""
    if kind in _converters:
        return _converters[kind]
    try:
        import opencc  # type: ignore[import-not-found]

        converter = opencc.OpenCC(kind)
    except Exception:  # pragma: no cover - 依赖缺失时保证软件可用
        log.warning("OpenCC 不可用，中文简繁统一已跳过（%s）", kind, exc_info=True)
        converter = None
    _converters[kind] = converter
    return converter


def convert_script(text: str, script: str) -> str:
    """按设置转换中文文本字形；空文本或保持原文时原样返回。"""
    if not text:
        return text
    normalized = normalize_script_name(script)
    if normalized == "preserve":
        return text
    kind = "t2s" if normalized == "simplified" else "s2t"
    converter = _converter(kind)
    if converter is None:
        return text
    try:
        return str(converter.convert(text))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - 转换异常不应影响主流程
        log.warning("中文简繁转换失败，保留原文", exc_info=True)
        return text


def count_script_changes(text: str, script: str) -> int:
    """统计按目标字形转换后会变化的字符数（用于日志/产品试验）。"""
    if not text:
        return 0
    converted = convert_script(text, script)
    return sum(1 for a, b in zip(text, converted) if a != b)


def script_rule(language: str, script: str) -> str:
    """给 DeepSeek 的简繁硬性规则；非中文或保持原文时返回空串。"""
    if not is_chinese_language(language):
        return ""
    normalized = normalize_script_name(script)
    if normalized == "simplified":
        return (
            "中文输出一律使用简体中文，不得输出繁体字；"
            "即使原文是繁体，也要转换成简体（专有名词按常用简体写法）。"
        )
    if normalized == "traditional":
        return (
            "中文输出一律使用繁体中文，不得输出简体字；"
            "即使原文是简体，也要转换成繁体（专有名词按常用繁体写法）。"
        )
    return ""


def script_label(script: str) -> str:
    """设置值的界面文案。"""
    return SCRIPTS.get(normalize_script_name(script), SCRIPTS[DEFAULT_SCRIPT])


__all__ = [
    "SCRIPTS",
    "DEFAULT_SCRIPT",
    "normalize_script_name",
    "is_chinese_language",
    "convert_script",
    "count_script_changes",
    "script_rule",
    "script_label",
]
