"""全局快捷键：Alt+~（不注册系统热键，Windows 下用 GetAsyncKeyState 轮询）。

不依赖第三方库：通过 ``ctypes`` 调用 user32.GetAsyncKeyState，由 Tk 的
``after(50ms)`` 在主线程轮询，检测「按下沿」触发一次，避免长按重复触发。

* 只在 Windows 生效，其他平台 :meth:`HotkeyListener.start` 返回 False（优雅降级）；
* 窗口在前台、最小化到托盘、隐藏时都能触发（GetAsyncKeyState 是全局键盘状态）；
* 不占用系统级热键注册表，避免与其它软件冲突。
"""

from __future__ import annotations

import ctypes
import logging
import sys
import tkinter as tk
from typing import Callable, Optional

log = logging.getLogger(__name__)

#: 界面文案统一使用的快捷键名称
HOTKEY_LABEL = "Alt+~"

VK_MENU = 0x12  # Alt
VK_OEM_3 = 0xC0  # ~ / ` 键（美式键盘左上角，中文键盘同键位）

#: 默认轮询间隔（毫秒）
DEFAULT_INTERVAL_MS = 50


def _load_get_async_key_state():
    """加载 user32.GetAsyncKeyState；非 Windows 或失败时返回 None。"""
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        func = user32.GetAsyncKeyState
        func.argtypes = [ctypes.c_int]
        func.restype = ctypes.c_short
        return func
    except Exception:  # pragma: no cover - 极端环境
        log.warning("无法加载 GetAsyncKeyState，全局快捷键不可用", exc_info=True)
        return None


class HotkeyListener:
    """通过 Tk 定时器轮询 Alt+~ 的按下沿。"""

    def __init__(
        self,
        root: tk.Misc,
        callback: Callable[[], None],
        interval_ms: int = DEFAULT_INTERVAL_MS,
        *,
        get_async_key_state: Optional[Callable[[int], int]] = None,
    ) -> None:
        self.root = root
        self.callback = callback
        self.interval_ms = max(10, int(interval_ms))
        self._get_async_key_state = (
            get_async_key_state
            if get_async_key_state is not None
            else _load_get_async_key_state()
        )
        self._job: Optional[str] = None
        self._pressed = False
        self._enabled = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def supported(self) -> bool:
        return self._get_async_key_state is not None

    def start(self) -> bool:
        """开始轮询；不支持的环境返回 False。"""
        if not self.supported or self._enabled:
            return bool(self.supported)
        self._enabled = True
        self._schedule(0)
        return True

    def stop(self) -> None:
        """停止轮询并取消已排队的 Tk 定时器。"""
        self._enabled = False
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except Exception:  # pragma: no cover - 窗口已销毁
                pass
            self._job = None

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
    def is_pressed_now(self) -> bool:
        """当前 Alt 与 ~ 是否同时按下（读取全局键盘状态）。"""
        getter = self._get_async_key_state
        if getter is None:
            return False
        try:
            alt_down = bool(int(getter(VK_MENU)) & 0x8000)
            if not alt_down:
                return False
            return bool(int(getter(VK_OEM_3)) & 0x8000)
        except Exception:  # pragma: no cover - 极端环境
            log.debug("读取按键状态失败", exc_info=True)
            return False

    def poll_once(self) -> bool:
        """轮询一次；返回本次是否检测到「按下沿」（测试可直接调用）。"""
        pressed = self.is_pressed_now()
        triggered = pressed and not self._pressed
        self._pressed = pressed
        return triggered

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _schedule(self, delay_ms: int) -> None:
        if not self._enabled:
            return
        try:
            self._job = self.root.after(max(0, int(delay_ms)), self._tick)
        except tk.TclError:  # pragma: no cover - 窗口销毁中
            self._job = None

    def _tick(self) -> None:
        self._job = None
        if not self._enabled:
            return
        try:
            if self.poll_once():
                try:
                    self.callback()
                except Exception:  # pragma: no cover - 回调异常不能中断轮询
                    log.exception("全局快捷键回调执行失败")
        finally:
            self._schedule(self.interval_ms)


__all__ = [
    "HOTKEY_LABEL",
    "VK_MENU",
    "VK_OEM_3",
    "DEFAULT_INTERVAL_MS",
    "HotkeyListener",
]
