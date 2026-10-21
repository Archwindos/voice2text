"""DPAPI 加密与通用工具函数测试。"""

from __future__ import annotations

import sys

import pytest

from voxscribe import secure
from voxscribe.util import (
    human_duration,
    human_size,
    process_memory_mb,
    sanitize_filename,
)


def test_secure_availability_flag() -> None:
    assert isinstance(secure.is_available(), bool)


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI 仅 Windows 可用")
def test_dpapi_roundtrip() -> None:
    if not secure.is_available():
        pytest.skip("当前环境 DPAPI 不可用")
    token = secure.protect("sk-abcdef-中文-123")
    assert token and isinstance(token, str)
    assert "sk-abcdef" not in token
    assert secure.unprotect(token) == "sk-abcdef-中文-123"


def test_dpapi_garbage_returns_none() -> None:
    assert secure.unprotect("@@@not-base64@@@") is None


def test_sanitize_filename() -> None:
    assert sanitize_filename('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"
    assert sanitize_filename("  ") == "未命名"
    assert sanitize_filename("会议记录 2026") == "会议记录 2026"
    assert sanitize_filename("con") == "con_"
    long_name = "很长" * 200
    assert len(sanitize_filename(long_name)) <= 120


def test_human_duration() -> None:
    assert human_duration(0) == "0秒"
    assert human_duration(61) == "1分01秒"
    assert human_duration(3661) == "1小时01分01秒"


def test_human_size() -> None:
    assert human_size(512) == "512B"
    assert human_size(1536).endswith("KB")
    assert human_size(5 * 1024 * 1024) == "5.0MB"


def test_process_memory_mb() -> None:
    value = process_memory_mb()
    assert value >= 0
    if sys.platform == "win32":
        # 修复 ctypes 签名后应能读到真实 WorkingSet
        assert value > 0
