"""图形界面主题：配色、字体、ttk 样式与 Windows 高 DPI 适配。

界面使用 ``clam`` 主题以便统一控制颜色；字体优先 Microsoft YaHei UI。
显式像素（内边距、控件高度等）通过 :meth:`Theme.px` 按 DPI 缩放，
字体尺寸使用磅值（Tk 会按 ``tk scaling`` 自动缩放），两者配合保证
高 DPI 屏幕上既不模糊也不过挤。
"""

from __future__ import annotations

import ctypes
import logging
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Dict, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 调色板：浅色、低饱和蓝灰，适合长时间阅读
# ---------------------------------------------------------------------------
COLORS: Dict[str, str] = {
    "bg": "#f3f5f9",            # 窗口底色
    "card": "#ffffff",          # 面板底色
    "border": "#dce3ed",        # 面板描边
    "text": "#1f2937",          # 主文字
    "muted": "#6b7280",         # 次要文字
    "faint": "#9aa3b2",         # 更弱的提示文字
    "primary": "#2563eb",       # 主色（按钮/进度条/强调）
    "primary_dark": "#1d4ed8",  # 主色深（悬停/标题）
    "primary_light": "#eaf1ff", # 主色浅（选中行）
    "header": "#1e3a5f",        # 顶栏深蓝
    "header_soft": "#2b5080",   # 顶栏按钮悬停
    "header_fg": "#ffffff",
    "header_sub": "#b9cbe0",
    "success": "#15803d",
    "danger": "#b91c1c",
    "warning": "#b45309",
    "input_bg": "#ffffff",
    "trough": "#e6ebf4",
    "heading_bg": "#eef2f8",
    # 语音输入（v1.1 首页）：录音红 + 波形蓝
    "record": "#dc2626",
    "record_dark": "#b91c1c",
    "record_soft": "#fee2e2",
    "wave_bg": "#f8fbff",
    "wave_old": "#c9d9f5",
    "wave_new": "#2563eb",
    "wave_frozen": "#dbe6f7",
}

_FONT_CANDIDATES = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "微软雅黑",
    "Segoe UI",
    "TkDefaultFont",
)
_MONO_CANDIDATES = ("Consolas", "Cascadia Mono", "Courier New", "TkFixedFont")

UI_SIZE = 10          # 界面常规字号（磅）
SMALL_SIZE = 9        # 辅助说明
BODY_SIZE = 11        # 预览正文


def _pick_font(root: tk.Misc, candidates: Tuple[str, ...], fallback: str) -> str:
    try:
        families = set(tkfont.families(root))
    except Exception:  # pragma: no cover - 极端环境
        families = set()
    for name in candidates:
        if name in families:
            return name
    return fallback


def enable_dpi_awareness() -> bool:
    """开启 Windows DPI 感知，避免界面被系统位图拉伸变糊。

    必须在创建 Tk 根窗口 **之前** 调用。非 Windows 平台直接返回 False。
    """
    if sys.platform != "win32":
        return False
    try:
        # 1 = SYSTEM_DPI_AWARE，兼容性最好；失败则回退旧 API
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return True
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            return True
        except Exception:
            log.debug("设置 DPI 感知失败，使用系统默认缩放", exc_info=True)
            return False


class Theme:
    """一次主题初始化后的字体/尺寸工厂。"""

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.family = _pick_font(root, _FONT_CANDIDATES, "TkDefaultFont")
        self.mono_family = _pick_font(root, _MONO_CANDIDATES, "TkFixedFont")
        self.scale = self._detect_scale(root)
        self.colors = COLORS

    # -- 尺寸与字体 ---------------------------------------------------
    @staticmethod
    def _detect_scale(root: tk.Misc) -> float:
        try:
            dpi = float(root.winfo_fpixels("1i"))
        except Exception:  # pragma: no cover - 极端环境
            dpi = 96.0
        return max(1.0, min(3.0, dpi / 96.0))

    def px(self, value: float) -> int:
        """把 96 DPI 下的像素值换算成当前屏幕像素值。"""
        return max(1, int(round(float(value) * self.scale)))

    def font(self, size: int = UI_SIZE, weight: str = "normal") -> Tuple[str, int, str]:
        return (self.family, int(size), weight)

    def mono(self, size: int = SMALL_SIZE) -> Tuple[str, int, str]:
        return (self.mono_family, int(size), "normal")

    # -- 样式 ---------------------------------------------------------
    def apply(self) -> None:
        root = self.root
        colors = COLORS
        try:
            root.configure(bg=colors["bg"])
        except tk.TclError:  # pragma: no cover - Toplevel 已销毁
            return

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:  # pragma: no cover - 主题缺失
            log.debug("clam 主题不可用，使用默认主题", exc_info=True)

        # 字体基准：toplevel/menu 等未显式指定字体的控件也统一
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=self.family, size=UI_SIZE)
            except Exception:  # pragma: no cover - 极端环境
                pass

        base = dict(
            background=colors["bg"],
            foreground=colors["text"],
            font=self.font(UI_SIZE),
            bordercolor=colors["border"],
            troughcolor=colors["trough"],
        )
        style.configure(".", **base)
        style.configure("TFrame", background=colors["bg"])
        style.configure("TLabel", background=colors["bg"], foreground=colors["text"])
        style.configure("Muted.TLabel", background=colors["bg"], foreground=colors["muted"])
        style.configure(
            "Status.TLabel", background=colors["bg"], foreground=colors["muted"],
            font=self.font(SMALL_SIZE),
        )
        style.configure(
            "Memory.TLabel", background=colors["bg"], foreground=colors["faint"],
            font=self.font(SMALL_SIZE),
        )

        # 卡片内控件（面板底色为白色）
        style.configure("Card.TFrame", background=colors["card"])
        style.configure("Card.TLabel", background=colors["card"], foreground=colors["text"])
        style.configure(
            "CardMuted.TLabel", background=colors["card"], foreground=colors["muted"],
            font=self.font(SMALL_SIZE),
        )
        style.configure(
            "CardHint.TLabel", background=colors["card"], foreground=colors["faint"],
            font=self.font(SMALL_SIZE),
        )
        style.configure(
            "CardTitle.TLabel", background=colors["card"],
            foreground=colors["primary_dark"], font=self.font(UI_SIZE + 1, "bold"),
        )
        for widget in ("TCheckbutton", "TRadiobutton"):
            style.configure(
                f"Card.{widget}", background=colors["card"], foreground=colors["text"],
                font=self.font(UI_SIZE), focuscolor=colors["card"],
            )
            style.map(
                f"Card.{widget}",
                background=[("active", colors["card"])],
                foreground=[("disabled", colors["faint"])],
            )

        # 通用按钮
        style.configure(
            "TButton", padding=(self.px(12), self.px(6)), font=self.font(UI_SIZE),
            background=colors["card"], bordercolor=colors["border"],
            lightcolor=colors["card"], darkcolor=colors["card"],
            relief="solid", borderwidth=1,
        )
        style.map(
            "TButton",
            background=[("pressed", colors["trough"]), ("active", colors["primary_light"])],
            bordercolor=[("active", colors["primary"])],
            relief=[("pressed", "solid")],
        )

        # 输入控件
        style.configure(
            "TEntry", fieldbackground=colors["input_bg"], foreground=colors["text"],
            bordercolor=colors["border"], insertcolor=colors["text"],
            padding=(self.px(6), self.px(4)), font=self.font(UI_SIZE),
        )
        style.map("TEntry", bordercolor=[("focus", colors["primary"])])
        style.configure(
            "TCombobox", fieldbackground=colors["input_bg"], background=colors["input_bg"],
            foreground=colors["text"], bordercolor=colors["border"],
            arrowcolor=colors["muted"], padding=(self.px(6), self.px(4)),
            font=self.font(UI_SIZE),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", colors["input_bg"])],
            foreground=[("readonly", colors["text"])],
            bordercolor=[("focus", colors["primary"]), ("active", colors["border"])],
            selectbackground=[("readonly", colors["input_bg"])],
            selectforeground=[("readonly", colors["text"])],
        )
        style.configure(
            "TSpinbox", fieldbackground=colors["input_bg"], foreground=colors["text"],
            bordercolor=colors["border"], arrowcolor=colors["muted"],
            padding=(self.px(4), self.px(3)), font=self.font(UI_SIZE),
        )

        # 文件队列
        style.configure(
            "Files.Treeview", background=colors["card"], fieldbackground=colors["card"],
            foreground=colors["text"], borderwidth=0, relief="flat",
            rowheight=self.px(30), font=self.font(UI_SIZE),
        )
        style.map(
            "Files.Treeview",
            background=[("selected", colors["primary_light"])],
            foreground=[("selected", colors["text"])],
        )
        style.configure(
            "Files.Treeview.Heading", background=colors["heading_bg"],
            foreground=colors["muted"], font=self.font(SMALL_SIZE, "bold"),
            relief="flat", padding=(self.px(6), self.px(4)),
        )
        style.map(
            "Files.Treeview.Heading",
            background=[("active", colors["primary_light"])],
            foreground=[("active", colors["primary_dark"])],
        )

        # 结果标签页
        style.configure("TNotebook", background=colors["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab", background=colors["bg"], foreground=colors["muted"],
            padding=(self.px(16), self.px(6)), font=self.font(UI_SIZE),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", colors["card"])],
            foreground=[("selected", colors["primary_dark"])],
            expand=[("selected", (0, 0, 0, 0))],
        )

        # 滚动条
        for orient in ("Vertical", "Horizontal"):
            style.configure(
                f"{orient}.TScrollbar", background=colors["border"],
                troughcolor=colors["bg"], bordercolor=colors["bg"],
                arrowcolor=colors["muted"], width=self.px(12),
            )
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=colors["trough"], background=colors["primary"],
            bordercolor=colors["trough"], lightcolor=colors["primary"],
            darkcolor=colors["primary"], thickness=self.px(8),
        )


def apply_theme(root: tk.Misc) -> Theme:
    """初始化主题并返回 :class:`Theme`（字体/尺寸工厂）。"""
    theme = Theme(root)
    theme.apply()
    return theme


def center_window(window: tk.Misc, width: int, height: int) -> None:
    """把窗口按指定尺寸居中到主屏幕。"""
    try:
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2 - int(height * 0.04))
        window.geometry(f"{width}x{height}+{x}+{y}")
    except tk.TclError:  # pragma: no cover - 极端环境
        pass


def make_card(
    parent: tk.Misc, theme: Theme, title: str = ""
) -> Tuple[tk.Frame, tk.Frame]:
    """创建白色圆角感面板（细边框卡片），返回 ``(card, body)``。

    ``body`` 是内容容器，调用方把控件放进去即可；卡片标题可选。
    """
    card = tk.Frame(
        parent,
        bg=COLORS["card"],
        highlightbackground=COLORS["border"],
        highlightthickness=1,
        bd=0,
    )
    if title:
        label = tk.Label(
            card,
            text=title,
            bg=COLORS["card"],
            fg=COLORS["primary_dark"],
            font=theme.font(UI_SIZE + 1, "bold"),
            anchor="w",
        )
        label.pack(fill="x", padx=theme.px(14), pady=(theme.px(10), 0))
    body = tk.Frame(card, bg=COLORS["card"])
    body.pack(
        fill="both",
        expand=True,
        padx=theme.px(14),
        pady=(theme.px(8), theme.px(12)),
    )
    return card, body


__all__ = [
    "COLORS",
    "UI_SIZE",
    "SMALL_SIZE",
    "BODY_SIZE",
    "Theme",
    "apply_theme",
    "center_window",
    "enable_dpi_awareness",
    "make_card",
]
