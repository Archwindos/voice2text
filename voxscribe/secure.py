"""Windows DPAPI 加密，用于本地保存 DeepSeek API Key。

用到的 API：
    CryptProtectData / CryptUnprotectData（crypt32.dll）
    LocalFree（kernel32.dll）

* 只有当前 Windows 用户能解密，密钥不会以明文落盘；
* 非 Windows 平台或调用失败时优雅降级（调用方回退到明文存储并提示）。
"""

from __future__ import annotations

import base64
import ctypes
import sys
from typing import Optional

_ENTROPY = b"VoxScribe::api-key::v1"
_is_windows = sys.platform == "win32"

if _is_windows:  # pragma: no branch - 平台分支
    try:
        from ctypes import wintypes

        class _DataBlob(ctypes.Structure):
            _fields_ = [
                ("cbData", wintypes.DWORD),
                ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
            ]

        _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        _crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_wchar_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        _crypt32.CryptProtectData.restype = wintypes.BOOL
        _crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        _crypt32.CryptUnprotectData.restype = wintypes.BOOL
        _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        _kernel32.LocalFree.restype = ctypes.c_void_p
        _DPAPI_AVAILABLE = True
    except Exception:  # pragma: no cover - 极端环境
        _DPAPI_AVAILABLE = False
else:  # pragma: no cover
    _DPAPI_AVAILABLE = False


def _blob_from_bytes(data: bytes):
    buffer = ctypes.create_string_buffer(data, len(data))
    blob = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    return blob, buffer


def _bytes_from_blob(blob: "_DataBlob") -> bytes:
    return ctypes.string_at(blob.pbData, blob.cbData)


def is_available() -> bool:
    """当前环境是否支持 DPAPI 加密。"""
    return bool(_DPAPI_AVAILABLE)


def protect(text: str) -> Optional[str]:
    """加密字符串，返回 base64（无 ``dpapi:`` 前缀）；失败返回 ``None``。"""
    if not _DPAPI_AVAILABLE or not text:
        return None
    try:
        data_in, _buffer_in = _blob_from_bytes(text.encode("utf-8"))
        entropy_in, _buffer_entropy = _blob_from_bytes(_ENTROPY)
        data_out = _DataBlob()
        ok = _crypt32.CryptProtectData(
            ctypes.byref(data_in),
            "VoxScribe API Key",
            ctypes.byref(entropy_in),
            None,
            None,
            0,
            ctypes.byref(data_out),
        )
        if not ok:
            return None
        try:
            return base64.b64encode(_bytes_from_blob(data_out)).decode("ascii")
        finally:
            _kernel32.LocalFree(data_out.pbData)
    except Exception:
        return None


def unprotect(token: str) -> Optional[str]:
    """解密 :func:`protect` 的返回值；失败返回 ``None``。"""
    if not _DPAPI_AVAILABLE or not token:
        return None
    try:
        raw = base64.b64decode(token.encode("ascii"), validate=True)
    except Exception:
        return None
    try:
        data_in, _buffer_in = _blob_from_bytes(raw)
        entropy_in, _buffer_entropy = _blob_from_bytes(_ENTROPY)
        data_out = _DataBlob()
        ok = _crypt32.CryptUnprotectData(
            ctypes.byref(data_in),
            None,
            ctypes.byref(entropy_in),
            None,
            None,
            0,
            ctypes.byref(data_out),
        )
        if not ok:
            return None
        try:
            return _bytes_from_blob(data_out).decode("utf-8", "replace")
        finally:
            _kernel32.LocalFree(data_out.pbData)
    except Exception:
        return None


__all__ = ["is_available", "protect", "unprotect"]
