"""单实例控制：保证同时只有一个 VoxScribe 窗口。

实现：
* Windows 命名互斥量（mutex）判断是否已有实例；
* 命名事件（auto-reset event）让第二个实例请求第一个实例把窗口显示到前台；
* 第一个实例在 Tk 主循环里轮询事件，收到请求就 ``deiconify`` / 置顶。

非 Windows 平台不启用（返回 True），不影响可移植性。
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Optional

log = logging.getLogger(__name__)

#: 互斥量/事件名（Local 命名空间：同一 Windows 会话内有效）
DEFAULT_MUTEX_NAME = r"Local\VoxScribe.SingleInstance.v1"
DEFAULT_EVENT_NAME = r"Local\VoxScribe.ShowWindow.v1"
DEFAULT_WINDOW_TITLE = "语声成文 VoxScribe"

ERROR_ALREADY_EXISTS = 183
EVENT_MODIFY_STATE = 0x0002
WAIT_OBJECT_0 = 0x00000000
SW_RESTORE = 9


class SingleInstance:
    """命名互斥量 + 激活事件。

    典型用法::

        instance = SingleInstance()
        if not instance.acquire():
            instance.request_activation()
            return 0          # 已有实例：激活它后自己退出
        try:
            run_app(instance)  # 主循环里轮询 consume_activation_request()
        finally:
            instance.close()
    """

    def __init__(
        self,
        *,
        mutex_name: str = DEFAULT_MUTEX_NAME,
        event_name: str = DEFAULT_EVENT_NAME,
        window_title: str = DEFAULT_WINDOW_TITLE,
    ) -> None:
        self.mutex_name = mutex_name
        self.event_name = event_name
        self.window_title = window_title
        self._mutex: Optional[int] = None
        self._event: Optional[int] = None
        self.is_primary = False

    # ------------------------------------------------------------------
    # 主实例
    # ------------------------------------------------------------------
    def acquire(self) -> bool:
        """尝试成为主实例；已有实例时返回 False。"""
        if sys.platform != "win32":
            self.is_primary = True
            return True
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        ctypes.set_last_error(0)
        mutex = kernel32.CreateMutexW(None, 0, self.mutex_name)
        last_error = ctypes.get_last_error()
        if not mutex:
            # 创建失败（权限等异常）：不要因此阻止软件启动
            log.warning("创建单实例互斥量失败（错误码 %s），继续启动", last_error)
            self.is_primary = True
            return True
        if last_error == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(ctypes.c_void_p(mutex))
            log.info("检测到已有 VoxScribe 实例在运行")
            return False
        self._mutex = int(mutex)
        self.is_primary = True
        # 创建激活事件（失败不影响单实例限制）
        kernel32.CreateEventW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        kernel32.CreateEventW.restype = ctypes.c_void_p
        event = kernel32.CreateEventW(None, 0, 0, self.event_name)
        if event:
            self._event = int(event)
        return True

    def consume_activation_request(self) -> bool:
        """主实例轮询：第二个实例是否请求显示窗口。"""
        if not self._event:
            return False
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint
        result = kernel32.WaitForSingleObject(ctypes.c_void_p(self._event), 0)
        return result == WAIT_OBJECT_0

    # ------------------------------------------------------------------
    # 第二个实例
    # ------------------------------------------------------------------
    def request_activation(self) -> bool:
        """请求已运行的实例显示窗口；事件不可用时回退到 FindWindow。"""
        if sys.platform != "win32":
            return False
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenEventW.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_wchar_p,
        ]
        kernel32.OpenEventW.restype = ctypes.c_void_p
        event = kernel32.OpenEventW(EVENT_MODIFY_STATE, 0, self.event_name)
        if event:
            try:
                kernel32.SetEvent.argtypes = [ctypes.c_void_p]
                kernel32.SetEvent.restype = ctypes.c_int
                return bool(kernel32.SetEvent(ctypes.c_void_p(event)))
            finally:
                kernel32.CloseHandle(ctypes.c_void_p(event))
        # 回退：直接按窗口标题激活（兼容旧版本/事件创建失败）
        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        hwnd = user32.FindWindowW(None, self.window_title)
        if hwnd:
            user32.ShowWindow(ctypes.c_void_p(hwnd), SW_RESTORE)
            user32.SetForegroundWindow(ctypes.c_void_p(hwnd))
            return True
        return False

    def close(self) -> None:
        """释放互斥量和事件句柄（进程退出时系统也会自动释放）。"""
        if sys.platform != "win32":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        if self._event:
            kernel32.CloseHandle(ctypes.c_void_p(self._event))
            self._event = None
        if self._mutex:
            kernel32.CloseHandle(ctypes.c_void_p(self._mutex))
            self._mutex = None
        self.is_primary = False


__all__ = [
    "DEFAULT_EVENT_NAME",
    "DEFAULT_MUTEX_NAME",
    "DEFAULT_WINDOW_TITLE",
    "SingleInstance",
]
