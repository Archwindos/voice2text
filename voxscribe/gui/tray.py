"""系统托盘常驻（pystray + Pillow）。

* 运行时用 Pillow 绘制麦克风图标，不需要外部图片资源；
* 菜单：显示/隐藏主界面、开始录音/停止录音、退出；双击托盘图标恢复窗口；
* 菜单回调在 pystray 线程中执行，统一通过 ``root.after(0, ...)`` 投递回 Tk 主线程；
* 缺少 pystray / Pillow 或设置了 ``VOXSCRIBE_DISABLE_TRAY=1`` 时优雅降级
  （:func:`tray_available` 返回 False，界面走普通关闭逻辑）。
"""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from types import SimpleNamespace
from typing import Callable, Optional

from .hotkey import HOTKEY_LABEL

log = logging.getLogger(__name__)

TRAY_NAME = "voxscribe"
TOOLTIP_BASE = f"语声成文 VoxScribe · 按 {HOTKEY_LABEL} 开始录音"
TOOLTIP_RECORDING = f"语声成文 VoxScribe · 正在录音（{HOTKEY_LABEL} 停止）"
HIDDEN_HINT_TITLE = "已最小化到托盘"
HIDDEN_HINT_MESSAGE = (
    f"语声成文仍在后台待机：按 {HOTKEY_LABEL} 开始录音；"
    "双击托盘图标恢复窗口，右键托盘图标可退出。"
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def tray_disabled_by_env() -> bool:
    """测试/无托盘环境通过环境变量禁用托盘。"""
    return _env_flag("VOXSCRIBE_DISABLE_TRAY")


def tray_available() -> bool:
    """当前环境是否可以使用托盘（依赖已安装且未被环境变量禁用）。"""
    if tray_disabled_by_env():
        return False
    try:
        import pystray  # noqa: F401
        from PIL import Image, ImageDraw  # noqa: F401
    except Exception:
        log.info("托盘组件不可用（缺少 pystray / Pillow），关闭窗口将直接退出")
        return False
    return True


def _load_backend() -> SimpleNamespace:
    """加载 pystray/Pillow 后端；测试可 monkeypatch 本函数注入假后端。"""
    import pystray
    from PIL import Image, ImageDraw

    return SimpleNamespace(
        Icon=pystray.Icon,
        Menu=pystray.Menu,
        MenuItem=pystray.MenuItem,
        SEPARATOR=pystray.Menu.SEPARATOR,
        Image=Image,
        ImageDraw=ImageDraw,
    )


def create_tray_image(recording: bool = False):
    """用 Pillow 画一个 64x64 麦克风图标（录音时红色，待机时蓝色）。"""
    from PIL import Image, ImageDraw

    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    background = (220, 38, 38, 255) if recording else (37, 99, 235, 255)
    white = (255, 255, 255, 255)
    draw.ellipse((2, 2, size - 3, size - 3), fill=background)
    # 麦克风胶囊 + 支架
    draw.rounded_rectangle((26, 13, 38, 34), radius=6, fill=white)
    draw.arc((20, 23, 44, 46), start=0, end=180, fill=white, width=3)
    draw.line((32, 46, 32, 51), fill=white, width=3)
    draw.line((24, 52, 40, 52), fill=white, width=3)
    return image


class TrayIcon:
    """托盘图标封装（一个实例对应一个托盘图标）。"""

    def __init__(
        self,
        root: tk.Misc,
        *,
        on_toggle_window: Callable[[], None],
        on_toggle_recording: Callable[[], None],
        on_quit: Callable[[], None],
        is_recording: Callable[[], bool],
        is_visible: Callable[[], bool],
    ) -> None:
        self._root = root
        self._on_toggle_window = on_toggle_window
        self._on_toggle_recording = on_toggle_recording
        self._on_quit = on_quit
        self._is_recording = is_recording
        self._is_visible = is_visible
        self._icon = None
        self._backend: Optional[SimpleNamespace] = None
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """创建并启动托盘图标；失败时返回 False（调用方走普通关闭逻辑）。"""
        if self._icon is not None or self._stopped:
            return self._icon is not None
        if not tray_available():
            return False
        try:
            backend = _load_backend()
            image = create_tray_image(recording=False)
            self._backend = backend
            self._icon = backend.Icon(
                TRAY_NAME,
                icon=image,
                title=self._tooltip(),
                menu=self._build_menu(backend),
            )
        except Exception:
            log.warning("创建托盘图标失败，将使用普通关闭逻辑", exc_info=True)
            self._icon = None
            self._backend = None
            return False
        self._thread = threading.Thread(
            target=self._run, name="voxscribe-tray", daemon=True
        )
        self._thread.start()
        log.info("托盘图标已启动")
        return True

    def _run(self) -> None:
        icon = self._icon
        if icon is None:
            return
        try:
            icon.run()
        except Exception:  # pragma: no cover - 取决于系统托盘环境
            log.warning("托盘图标运行异常退出", exc_info=True)

    def stop(self) -> None:
        """停止托盘图标（幂等）。"""
        self._stopped = True
        icon = self._icon
        self._icon = None
        self._backend = None
        if icon is not None:
            try:
                icon.stop()
            except Exception:  # pragma: no cover - 图标已退出
                log.debug("停止托盘图标失败", exc_info=True)

    # ------------------------------------------------------------------
    # 状态刷新
    # ------------------------------------------------------------------
    def update_state(self) -> None:
        """根据当前录音/窗口状态刷新图标、tooltip 与菜单文案。"""
        icon = self._icon
        backend = self._backend
        if icon is None or backend is None or self._stopped:
            return
        try:
            icon.title = self._tooltip()
            icon.icon = create_tray_image(recording=self._recording())
            icon.menu = self._build_menu(backend)
            icon.update_menu()
        except Exception:  # pragma: no cover - 托盘已销毁
            log.debug("刷新托盘状态失败", exc_info=True)

    def notify(self, title: str, message: str) -> bool:
        """显示托盘气泡提示；系统不支持时静默返回 False。"""
        icon = self._icon
        if icon is None or self._stopped:
            return False
        try:
            icon.notify(message, title)
            return True
        except Exception:  # pragma: no cover - 取决于系统通知支持
            log.debug("托盘气泡提示失败", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _recording(self) -> bool:
        try:
            return bool(self._is_recording())
        except Exception:  # pragma: no cover - 防御性
            return False

    def _visible(self) -> bool:
        try:
            return bool(self._is_visible())
        except Exception:  # pragma: no cover - 防御性
            return False

    def _tooltip(self) -> str:
        return TOOLTIP_RECORDING if self._recording() else TOOLTIP_BASE

    def _window_label(self) -> str:
        return "隐藏主界面" if self._visible() else "显示主界面"

    def _record_label(self) -> str:
        action = "停止录音" if self._recording() else "开始录音"
        return f"{action}（{HOTKEY_LABEL}）"

    def _build_menu(self, backend: SimpleNamespace):
        """构建菜单；每次刷新都会重建，保证动态文案（显示/隐藏、录音状态）。"""
        return backend.Menu(
            backend.MenuItem(
                self._window_label(),
                self._dispatch(self._on_toggle_window),
                default=True,
            ),
            backend.MenuItem(
                self._record_label(), self._dispatch(self._on_toggle_recording)
            ),
            backend.SEPARATOR,
            backend.MenuItem("退出", self._dispatch(self._on_quit)),
        )

    def _dispatch(self, callback: Callable[[], None]) -> Callable[..., None]:
        """把托盘线程的回调安全地投递到 Tk 主线程。"""

        def handler(*_args, **_kwargs) -> None:
            try:
                self._root.after(0, lambda: self._safe_call(callback))
            except Exception:  # pragma: no cover - 窗口已销毁
                log.debug("托盘回调投递失败", exc_info=True)

        return handler

    @staticmethod
    def _safe_call(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:  # pragma: no cover - 回调异常不影响托盘
            log.exception("托盘菜单回调执行失败")


__all__ = [
    "HIDDEN_HINT_TITLE",
    "HIDDEN_HINT_MESSAGE",
    "TOOLTIP_BASE",
    "TOOLTIP_RECORDING",
    "TrayIcon",
    "create_tray_image",
    "tray_available",
    "tray_disabled_by_env",
]
