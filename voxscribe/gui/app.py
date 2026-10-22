"""图形界面主窗口与启动入口（v1.2）。

页面结构：
* 首页「🎤 语音输入」：一键口述 → 本地转写 → DeepSeek「口述校对」→ 可编辑、
  可复制的文字。大录音按钮 / 全局 Alt+~ 开始与结束，Canvas 实时波形，自动复制；
  ASR 一完成就先显示原始转写，DeepSeek 润色回来后再原地替换（两段式出字更快）。
* 「📁 文件转写」：原有的文件队列、批量处理、四格式导出、双标签预览。
* 底部全局状态：进度条、状态文字、内存占用。
* 系统托盘常驻：关闭/最小化默认隐藏到托盘，Alt+~ 在后台也能开始录音。

线程模型：
* 主线程只负责界面；
* 语音输入与文件转写都在后台工作线程中执行，通过 ``ui_queue``
  把 :class:`JobEvent` 与自定义 UI 消息投递回主线程，主线程每 100ms 轮询；
* 关闭 / 取消通过 ``threading.Event`` 通知后台；托盘回调通过 ``root.after(0, ...)``
  投递回主线程。

测试辅助：``VOXSCRIBE_GUI_HIDDEN=1`` 时窗口不显示、不创建托盘、关闭时正常销毁；
``VOXSCRIBE_DISABLE_TRAY=1`` 单独禁用托盘；``VOXSCRIBE_GUI_AUTOCLOSE_MS``
存在时窗口打开后指定毫秒自动关闭（隐藏模式冒烟测试用）。
"""

from __future__ import annotations

import logging
import math
import os
import queue
import re
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    APP_NAME,
    ASR_MODELS,
    LANGUAGES,
    PROJECT_ROOT,
    AppSettings,
    default_output_dir,
    recordings_dir,
)
from .. import glossary as glossary_module
from ..errors import CancelledError, VoxError
from ..models import (
    EXPORT_FORMATS,
    QUICK_MODES,
    STYLES,
    JobEvent,
    JobOptions,
    JobResult,
    format_timestamp,
)
from ..pipeline import Pipeline
from ..util import human_duration, process_memory_mb, reveal_path
from . import theme as theme_module
from .dialogs import HelpDialog, SettingsDialog, show_welcome
from .hotkey import HOTKEY_LABEL, HotkeyListener
from .theme import BODY_SIZE, COLORS, SMALL_SIZE, UI_SIZE, Theme, center_window
from .tray import (
    HIDDEN_HINT_MESSAGE,
    HIDDEN_HINT_TITLE,
    TrayIcon,
    tray_available,
)

log = logging.getLogger(__name__)

APP_TITLE = "语声成文 VoxScribe"
APP_DISPLAY_NAME = "语声成文"
APP_SUBTITLE = "口述成文 · 本地转写 + DeepSeek 整理"

#: 首页三种输出模式（键顺序即界面按钮顺序）
QUICK_MODE_ORDER = ("direct", "person", "ai")
HOME_PLACEHOLDER = "转换后的文字会出现在这里，可直接复制或修改"
FULL_WIDTH_SPACE = "\u3000"
START_BUTTON_TEXT = "● 开始录音"
STOP_BUTTON_TEXT = "■ 停止并转换"
CONVERTING_BUTTON_TEXT = "转换中…"
HOME_IDLE_HINT = f"按 {HOTKEY_LABEL} 说一段，再按一次结束"

AUDIO_EXTENSIONS = frozenset(
    {
        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp",
    }
)
AUDIO_FILE_PATTERNS = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
AUDIO_FILETYPES = [
    ("音频 / 视频文件", AUDIO_FILE_PATTERNS),
    ("所有文件", "*.*"),
]

# 文件批量处理各阶段在整体进度中的权重（百分比）
_STAGE_WEIGHTS = {
    "prepare": (0.0, 4.0),
    "asr": (4.0, 68.0),
    "refine": (68.0, 94.0),
    "export": (94.0, 100.0),
}
_STAGE_LABELS = {"prepare": "准备", "asr": "转写", "refine": "整理", "export": "导出"}

# 首页语音输入各阶段权重
_QUICK_STAGE_WEIGHTS = {
    "prepare": (0.0, 3.0),
    "asr": (3.0, 72.0),
    "refine": (72.0, 100.0),
}

#: 波形柱数量（Recorder.recent_levels(90)）
WAVE_BARS = 90

_STATUS_TAGS = {
    "等待中": "pending",
    "处理中": "active",
    "已完成": "done",
    "失败": "failed",
    "已取消": "pending",
    "未处理": "pending",
}

_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?\s*(?:MB|GB))")

#: 可选依赖：拖拽添加文件
_DND_AVAILABLE = False
try:  # pragma: no cover - 取决于运行环境
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore

    _DND_IMPORT_OK = True
except Exception:  # pragma: no cover
    DND_FILES = None  # type: ignore[assignment]
    TkinterDnD = None  # type: ignore[assignment]
    _DND_IMPORT_OK = False

#: 可选依赖：录音（未就绪时首页按钮自动禁用并提示）
try:  # pragma: no cover - 取决于队友进度
    from ..recorder import Recorder  # type: ignore

    _RECORDER_IMPORT_OK = True
except Exception:  # pragma: no cover
    Recorder = None  # type: ignore[assignment]
    _RECORDER_IMPORT_OK = False


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _shorten(text: str, limit: int = 60) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _env_flag(name: str) -> bool:
    """读取布尔环境变量（1/true/yes/on）。"""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _read_autoclose_ms() -> int:
    raw = os.environ.get("VOXSCRIBE_GUI_AUTOCLOSE_MS", "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        log.warning("VOXSCRIBE_GUI_AUTOCLOSE_MS 不是整数：%r", raw)
        return 0
    return value if value > 0 else 0


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _wave_visual(level: float) -> float:
    """把归一化音量映射成波形高度比例（0~0.8）。

    保留顶部 20% 余量：正常说话（0.2~0.8）对应约 24%~78% 高度，
    即使很响的瞬间也不会顶到波形区上沿。
    """
    value = _clamp01(level)
    if value < 0.01:  # 噪声门：安静时不画明显波形
        return 0.0
    return min(0.8, (value ** 0.85) * 0.95)


def _blend_color(start: str, end: str, ratio: float) -> str:
    """在两个 ``#rrggbb`` 颜色之间线性插值。"""
    ratio = max(0.0, min(1.0, float(ratio)))

    def _channel(value: str, index: int) -> int:
        return int(value[1 + index * 2: 3 + index * 2], 16)

    parts = []
    for index in range(3):
        start_value = _channel(start, index)
        end_value = _channel(end, index)
        parts.append(round(start_value + (end_value - start_value) * ratio))
    return "#{:02x}{:02x}{:02x}".format(*parts)


@dataclass
class QueueFile:
    """待处理文件队列中的一项。"""

    path: Path
    uid: int
    status: str = "待处理"


class MainWindow:
    """主窗口（组合 ``tk.Tk``，便于拖拽依赖不可用时降级为普通 Tk）。"""

    POLL_MS = 100
    MEMORY_MS = 2000
    IDLE_CHECK_MS = 30000
    WAVE_MS = 50

    def __init__(self, root: tk.Tk, settings: AppSettings) -> None:
        self.root = root
        self.settings = settings
        self.theme: Theme = theme_module.apply_theme(root)
        self.pipeline = Pipeline(settings)
        # 已学习词语表（data/glossary.json）；启动时加载并注入 pipeline
        self._glossary_entries: List[glossary_module.GlossaryEntry] = []
        self._load_glossary_entries()

        # ---- 页面 ----
        self._page = "home"

        # ---- 文件转写状态 ----
        self.ui_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.files: List[QueueFile] = []
        self.results: List[Tuple[QueueFile, JobResult]] = []
        self._uid = 0
        self._process_id = 0
        self._processing = False
        self._cancel_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._batch: List[QueueFile] = []
        self._batch_index = 0
        self._batch_total = 0
        self._last_result: Optional[JobResult] = None
        self._last_result_item: Optional[QueueFile] = None
        self._last_output_dir: Optional[Path] = None

        # ---- 首页语音输入状态：idle / recording / converting ----
        self._quick_state = "idle"
        self._quick_cancel = threading.Event()
        self._quick_worker: Optional[threading.Thread] = None
        self._quick_seq = 0
        self._quick_started = 0.0
        self._quick_percent = 0.0
        self._quick_stage_label = ""
        self._quick_refine_used = False
        self._quick_has_key = False
        self._last_levels: List[float] = []
        self._level_history: List[float] = []
        self._wave_job: Optional[str] = None
        self._wave_phase = 0
        self._home_placeholder = True
        # ---- 直出模式的词语自动学习状态 ----
        self._learn_baseline_text = ""
        self._learn_baseline_mode = ""
        self._learn_after_id: Optional[str] = None

        # ---- 通用 ----
        self._closing = False
        self._save_job: Optional[str] = None
        self._poll_job: Optional[str] = None
        self._memory_job: Optional[str] = None
        self._idle_job: Optional[str] = None
        self._autoclose_job: Optional[str] = None
        self._startup_jobs: List[str] = []
        self._recorder: Any = None
        self._recorder_note = ""
        self._syncing_options = False
        self._autoclose_ms = _read_autoclose_ms()
        self._dnd_available = _DND_AVAILABLE
        self._dnd_enabled = False

        # ---- v1.2：隐藏测试模式 / 托盘 / 全局快捷键 ----
        self._hidden_mode = _env_flag("VOXSCRIBE_GUI_HIDDEN")
        self._tray_disabled = _env_flag("VOXSCRIBE_DISABLE_TRAY") or self._hidden_mode
        self._quitting = False
        self._tray: Optional[Any] = None
        self._tray_hint_shown = False
        self._tray_job: Optional[str] = None
        self._hotkey: Optional[HotkeyListener] = None
        self._quick_interim_text = ""
        # 模型预热（v1.2：启动后后台加载 GPU 模型，保证首句也快）
        self._warming_up = False
        self._warmup_seq = 0
        self._warmup_job: Optional[str] = None

        self._configure_window()
        self._build_ui()
        self._restore_geometry()
        self._setup_dnd()
        self._init_recorder()
        self._apply_settings_to_ui()
        self._refresh_file_list()
        self._bind_shortcuts()
        self._set_home_timer(None)
        self._draw_home_idle()
        self._init_hotkey()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close_request)
        self.root.report_callback_exception = self._on_callback_exception
        self.root.bind("<Unmap>", self._on_window_unmap)
        if self._hidden_mode:
            # 测试/后台模式：不显示窗口、不创建托盘、关闭时直接销毁
            try:
                self.root.withdraw()
            except tk.TclError:  # pragma: no cover
                pass
        else:
            # 等主窗口真正显示出来后再创建托盘图标
            self._tray_job = self.root.after(300, self._init_tray)

        self._update_memory()
        self._schedule_idle_check()
        self._poll_queue()
        if self._autoclose_ms:
            log.info("GUI 自动关闭已启用：%d ms（测试模式）", self._autoclose_ms)
            self._autoclose_job = self.root.after(self._autoclose_ms, self._on_close)
        else:
            self._startup_jobs.append(self.root.after(450, self._maybe_show_first_run))
            self._startup_jobs.append(self.root.after(700, self._startup_hints))
            if not self._hidden_mode:
                # 启动后后台预热识别模型（GPU），让第一句口述也快
                self._startup_jobs.append(self.root.after(650, self._start_warmup))

    # ==================================================================
    # 窗口与整体布局
    # ==================================================================
    def _configure_window(self) -> None:
        self.root.title(APP_TITLE)
        self.root.option_add("*tearOff", False)
        try:
            self.root.iconname(APP_NAME)
        except tk.TclError:  # pragma: no cover
            pass
        # 窗口/任务栏图标（与桌面快捷方式使用同一个 ico）
        icon_path = PROJECT_ROOT / "assets" / "voxscribe.ico"
        try:
            if icon_path.exists():
                self.root.iconbitmap(default=str(icon_path))
        except tk.TclError:  # pragma: no cover
            pass

    def _restore_geometry(self) -> None:
        theme = self.theme
        min_w, min_h = theme.px(860), theme.px(640)
        self.root.minsize(min_w, min_h)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        saved = re.match(
            r"^(\d+)x(\d+)([+-]-?\d+)([+-]-?\d+)$",
            (self.settings.window_geometry or "").strip(),
        )
        if saved:
            width = max(int(saved.group(1)), min_w)
            height = max(int(saved.group(2)), min_h)
            x = int(saved.group(3))
            y = int(saved.group(4))
            if (
                -theme.px(20) <= x <= screen_w - theme.px(120)
                and -theme.px(20) <= y <= screen_h - theme.px(120)
            ):
                width = min(width, screen_w)
                height = min(height, screen_h)
                x = max(0, min(x, screen_w - theme.px(120)))
                y = max(0, min(y, screen_h - theme.px(120)))
                self.root.geometry(f"{width}x{height}+{x}+{y}")
                return
            log.info("保存的窗口位置不在屏幕内，改为居中显示：%r", saved.group(0))
        center_window(self.root, theme.px(980), theme.px(760))

    def _build_ui(self) -> None:
        theme = self.theme
        colors = COLORS
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=0)  # 标题栏
        root.rowconfigure(1, weight=0)  # 导航
        root.rowconfigure(2, weight=1)  # 页面
        root.rowconfigure(3, weight=0)  # 全局状态栏

        self._build_header(root)
        self._build_nav(root)

        self.page_container = tk.Frame(root, bg=colors["bg"])
        self.page_container.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=theme.px(16),
            pady=(theme.px(10), theme.px(2)),
        )
        self.page_container.columnconfigure(0, weight=1)
        self.page_container.rowconfigure(0, weight=1)

        self.home_page = self._build_home_page(self.page_container)
        self.file_page = self._build_file_page(self.page_container)
        self.home_page.grid(row=0, column=0, sticky="nsew")
        self.file_page.grid(row=0, column=0, sticky="nsew")
        self.file_page.grid_remove()

        self._build_status_bar(root)
        self._update_nav_styles()

    # ------------------------------------------------------------------
    # 顶部标题栏 + 页面导航
    # ------------------------------------------------------------------
    def _build_header(self, root: tk.Tk) -> None:
        theme = self.theme
        colors = COLORS
        header = tk.Frame(root, bg=colors["header"], height=theme.px(54))
        header.grid(row=0, column=0, sticky="ew")
        header.pack_propagate(False)
        tk.Frame(root, bg=colors["border"], height=1).grid(
            row=0, column=0, sticky="sew"
        )

        brand = tk.Frame(header, bg=colors["header"])
        brand.pack(side="left", padx=(theme.px(22), 0), pady=theme.px(9))
        tk.Label(
            brand,
            text=APP_DISPLAY_NAME,
            bg=colors["header"],
            fg=colors["header_fg"],
            font=theme.font(14, "bold"),
        ).pack(side="left")
        tk.Label(
            brand,
            text="VoxScribe",
            bg=colors["header"],
            fg=colors["primary"],
            font=theme.font(SMALL_SIZE, "bold"),
        ).pack(side="left", padx=(theme.px(8), 0), pady=(theme.px(2), 0))
        tk.Label(
            brand,
            text=APP_SUBTITLE,
            bg=colors["header"],
            fg=colors["faint"],
            font=theme.font(SMALL_SIZE),
        ).pack(side="left", padx=(theme.px(16), 0), pady=(theme.px(2), 0))

        header_actions = tk.Frame(header, bg=colors["header"])
        header_actions.pack(side="right", padx=(0, theme.px(16)), pady=theme.px(8))
        self.help_btn = self._header_button(header_actions, "帮助", self._open_help)
        self.help_btn.pack(side="right", padx=(theme.px(4), 0))
        self.settings_btn = self._header_button(header_actions, "设置", self._open_settings)
        self.settings_btn.pack(side="right", padx=(theme.px(4), 0))

    def _build_nav(self, root: tk.Tk) -> None:
        theme = self.theme
        colors = COLORS
        wrapper = tk.Frame(root, bg=colors["bg"])
        wrapper.grid(row=1, column=0, sticky="ew", padx=theme.px(16), pady=(theme.px(8), 0))

        nav = tk.Frame(wrapper, bg=colors["bg"])
        nav.pack(fill="x")
        self.nav_items: Dict[str, Tuple[tk.Button, tk.Frame]] = {}
        for key, text in (("home", "🎤 语音输入"), ("files", "📁 文件转写")):
            item = tk.Frame(nav, bg=colors["bg"])
            item.pack(side="left", padx=(0, theme.px(4)))
            button = tk.Button(
                item,
                text=text,
                command=lambda page=key: self._switch_page(page),
                bg=colors["bg"],
                fg=colors["muted"],
                activebackground=colors["primary_light"],
                activeforeground=colors["primary_dark"],
                font=theme.font(UI_SIZE + 1),
                relief="flat",
                bd=0,
                highlightthickness=0,
                padx=theme.px(16),
                pady=theme.px(7),
                cursor="hand2",
            )
            button.pack()
            underline = tk.Frame(item, bg=colors["bg"], height=theme.px(3))
            underline.pack(fill="x")
            self.nav_items[key] = (button, underline)

        separator = tk.Frame(wrapper, bg=colors["border"], height=1)
        separator.pack(fill="x", pady=(theme.px(4), 0))

    def _update_nav_styles(self) -> None:
        theme = self.theme
        for key, (button, underline) in self.nav_items.items():
            active = key == self._page
            button.configure(
                bg=COLORS["primary_light"] if active else COLORS["bg"],
                fg=COLORS["primary_dark"] if active else COLORS["muted"],
                font=theme.font(UI_SIZE + 1, "bold" if active else "normal"),
            )
            underline.configure(bg=COLORS["primary"] if active else COLORS["bg"])

    def _switch_page(self, page: str) -> None:
        if page == self._page:
            return
        if self._quick_state != "idle":
            messagebox.showinfo(
                "请先结束当前操作",
                f"正在录音或转换文字，请先结束（再按一次 {HOTKEY_LABEL}）或按 Esc 取消后再切换页面。",
                parent=self.root,
            )
            return
        self._page = page
        if page == "home":
            self.file_page.grid_remove()
            self.home_page.grid()
            self._draw_home_idle()
        else:
            self.home_page.grid_remove()
            self.file_page.grid()
        self._update_nav_styles()

    def _header_button(
        self, parent: tk.Misc, text: str, command: Any, *, accent: bool = False
    ) -> tk.Button:
        theme = self.theme
        colors = COLORS
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=colors["header"],
            fg=colors["danger"] if accent else colors["header_sub"],
            activebackground=colors["header_soft"],
            activeforeground=colors["header_fg"],
            disabledforeground=colors["faint"],
            font=theme.font(UI_SIZE, "bold" if accent else "normal"),
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=theme.px(14),
            pady=theme.px(6),
            cursor="hand2",
        )

    def _panel_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Any,
        *,
        accent: bool = False,
        danger: bool = False,
    ) -> tk.Button:
        theme = self.theme
        colors = COLORS
        if accent:
            bg, fg = colors["primary_light"], colors["primary_dark"]
            active_bg, active_fg = colors["primary_light"], colors["primary"]
            border = colors["primary_light"]
        elif danger:
            bg, fg = colors["card"], colors["muted"]
            active_bg, active_fg = colors["record_soft"], colors["danger"]
            border = colors["border"]
        else:
            bg, fg = colors["card"], colors["text"]
            active_bg, active_fg = colors["primary_light"], colors["primary_dark"]
            border = colors["border"]
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=active_fg,
            disabledforeground=colors["faint"],
            font=theme.font(UI_SIZE, "bold" if accent else "normal"),
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=theme.px(10),
            pady=theme.px(5),
            cursor="hand2",
        )

    def _action_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Any,
        *,
        primary: bool = False,
        danger: bool = False,
    ) -> tk.Button:
        theme = self.theme
        colors = COLORS
        if primary:
            bg, fg = colors["primary"], colors["primary_ink"]
            active_bg, active_fg = colors["primary_dark"], colors["primary_ink"]
            border = colors["primary"]
        elif danger:
            bg, fg = colors["card"], colors["danger"]
            active_bg, active_fg = colors["record_soft"], colors["danger"]
            border = colors["border"]
        else:
            bg, fg = colors["card_alt"], colors["text"]
            active_bg, active_fg = colors["primary_light"], colors["primary_dark"]
            border = colors["border"]
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=active_fg,
            disabledforeground=colors["faint"],
            font=theme.font(UI_SIZE, "bold" if primary else "normal"),
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            padx=theme.px(18),
            pady=theme.px(7),
            cursor="hand2",
        )

    def _card_checkbutton(
        self,
        parent: tk.Misc,
        text: str,
        variable: tk.BooleanVar,
        command: Any = None,
    ) -> tk.Checkbutton:
        """卡片上的复选框（深色背景 + 经典勾选）。"""
        theme = self.theme
        colors = COLORS
        return tk.Checkbutton(
            parent,
            text=text,
            variable=variable,
            command=command,
            bg=colors["card"],
            fg=colors["text"],
            activebackground=colors["card"],
            activeforeground=colors["text"],
            selectcolor=colors["input_bg"],
            disabledforeground=colors["faint"],
            font=theme.font(UI_SIZE),
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            anchor="w",
        )

    # ==================================================================
    # 首页：语音输入
    # ==================================================================
    def _build_home_page(self, parent: tk.Frame) -> tk.Frame:
        theme = self.theme
        colors = COLORS
        page = tk.Frame(parent, bg=colors["bg"])
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=0)
        page.rowconfigure(1, weight=1)

        # ---------- 录音控制 + 半高波形（非对称：基线在底部，只向上生长） ----------
        input_card, input_body = theme_module.make_card(page, theme)
        input_card.grid(row=0, column=0, sticky="ew")

        # 输出模式：直出 / 对人 / 对AI（切换只影响 DeepSeek prompt 与思考强度）
        mode_row = tk.Frame(input_body, bg=colors["card"])
        mode_row.pack(pady=(theme.px(8), 0))
        tk.Label(
            mode_row,
            text="模式",
            bg=colors["card"],
            fg=colors["muted"],
            font=theme.font(SMALL_SIZE),
        ).pack(side="left", padx=(0, theme.px(8)))
        self.quick_mode_buttons: Dict[str, tk.Button] = {}
        for mode_key in QUICK_MODE_ORDER:
            label = QUICK_MODES[mode_key]["label"]
            button = tk.Button(
                mode_row,
                text=label,
                command=lambda key=mode_key: self._set_quick_mode(key),
                font=theme.font(SMALL_SIZE, "bold"),
                relief="flat",
                bd=0,
                highlightthickness=0,
                padx=theme.px(12),
                pady=theme.px(2),
                cursor="hand2",
            )
            button.pack(side="left", padx=(0, theme.px(5)))
            self.quick_mode_buttons[mode_key] = button
        self.quick_mode_hint_var = tk.StringVar(value="")
        tk.Label(
            mode_row,
            textvariable=self.quick_mode_hint_var,
            bg=colors["card"],
            fg=colors["muted"],
            font=theme.font(SMALL_SIZE),
        ).pack(side="left", padx=(theme.px(8), 0))
        self._sync_quick_mode_from_settings(refresh=False)
        self._refresh_quick_mode_buttons()

        self.home_record_btn = tk.Button(
            input_body,
            text=START_BUTTON_TEXT,
            command=self._toggle_quick,
            bg=colors["primary"],
            fg=colors["primary_ink"],
            activebackground=colors["primary_dark"],
            activeforeground=colors["primary_ink"],
            disabledforeground=colors["faint"],
            font=theme.font(12, "bold"),
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=theme.px(26),
            pady=theme.px(8),
            cursor="hand2",
        )
        self.home_record_btn.pack(pady=(theme.px(10), theme.px(2)))

        self.home_timer_var = tk.StringVar(value="")
        self.home_timer_label = tk.Label(
            input_body,
            textvariable=self.home_timer_var,
            bg=colors["card"],
            fg=colors["primary"],
            font=theme.mono(18, "bold"),
            anchor="center",
        )
        self.home_timer_label.pack_forget()

        self.home_status_var = tk.StringVar(value=HOME_IDLE_HINT)
        self.home_status_label = tk.Label(
            input_body,
            textvariable=self.home_status_var,
            bg=colors["card"],
            fg=colors["muted"],
            font=theme.font(SMALL_SIZE),
            anchor="center",
        )
        self.home_status_label.pack(fill="x", pady=(theme.px(1), theme.px(4)))

        self.wave_canvas = tk.Canvas(
            input_body,
            height=theme.px(84),
            bg=colors["wave_bg"],
            highlightthickness=0,
            bd=0,
        )
        self.wave_canvas.pack(fill="x")
        self.wave_canvas.bind("<Configure>", self._on_wave_configure)

        # ---------- 结果区（放大文本区） ----------
        result_card = tk.Frame(
            page,
            bg=colors["card"],
            highlightbackground=colors["border"],
            highlightthickness=1,
            bd=0,
        )
        result_card.grid(row=1, column=0, sticky="nsew", pady=(theme.px(10), 0))
        result_card.columnconfigure(0, weight=1)
        result_card.rowconfigure(2, weight=1)

        header_row = tk.Frame(result_card, bg=colors["card"])
        header_row.grid(
            row=0, column=0, sticky="ew", padx=theme.px(14), pady=(theme.px(8), 0)
        )
        tk.Label(
            header_row,
            text="转换结果",
            bg=colors["card"],
            fg=colors["text"],
            font=theme.font(UI_SIZE, "bold"),
        ).pack(side="left")
        self.home_clear_btn = self._panel_button(
            header_row, "清空", self._clear_home_text
        )
        self.home_clear_btn.pack(side="right")
        self.home_copy_btn = self._panel_button(
            header_row, "复制文字", lambda: self._copy_home_text(), accent=True
        )
        self.home_copy_btn.pack(side="right", padx=(0, theme.px(6)))
        self.home_result_info_var = tk.StringVar(value="")
        self.home_result_info_label = tk.Label(
            header_row,
            textvariable=self.home_result_info_var,
            bg=colors["card"],
            fg=colors["muted"],
            font=theme.font(SMALL_SIZE),
            anchor="e",
        )
        self.home_result_info_label.pack(side="right", padx=(0, theme.px(12)))

        self.home_inline_hint = tk.Label(
            result_card,
            text="",
            bg=colors["card"],
            fg=colors["warning"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
            justify="left",
            wraplength=theme.px(820),
        )
        self.home_inline_hint.grid(
            row=1, column=0, sticky="ew", padx=theme.px(14), pady=(theme.px(4), 0)
        )
        self.home_inline_hint.grid_remove()

        text_wrap = tk.Frame(result_card, bg=colors["card"])
        text_wrap.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=theme.px(14),
            pady=(theme.px(6), theme.px(12)),
        )
        text_wrap.columnconfigure(0, weight=1)
        text_wrap.rowconfigure(0, weight=1)
        self.home_text = tk.Text(
            text_wrap,
            wrap="word",
            font=theme.font(BODY_SIZE),
            bg=colors["input_bg"],
            fg=colors["text"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=colors["border"],
            highlightcolor=colors["primary"],
            padx=theme.px(14),
            pady=theme.px(12),
            spacing1=theme.px(3),
            spacing3=theme.px(6),
            selectbackground=colors["selection"],
            selectforeground=colors["text"],
            insertbackground=colors["primary"],
        )
        text_scroll = ttk.Scrollbar(
            text_wrap, orient="vertical", command=self.home_text.yview
        )
        self.home_text.configure(yscrollcommand=text_scroll.set)
        self.home_text.grid(row=0, column=0, sticky="nsew")
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.home_text.tag_configure("placeholder", foreground=colors["faint"])
        self.home_text.bind("<FocusIn>", self._on_home_text_focus_in)
        self.home_text.bind("<FocusOut>", self._on_home_text_focus_out)
        self.home_text.bind("<KeyRelease>", self._schedule_glossary_learning)
        self.home_text.bind("<Control-Return>", self._on_home_copy_shortcut)

        self._show_home_placeholder()
        self._sync_home_text_buttons()
        return page

    def _sync_home_text_buttons(self) -> None:
        """没有文字时禁用复制/清空按钮，保持界面干净。"""
        enabled = bool(self._get_home_text())
        state = "normal" if enabled else "disabled"
        for button in (
            getattr(self, "home_copy_btn", None),
            getattr(self, "home_clear_btn", None),
        ):
            if button is None:
                continue
            try:
                button.configure(state=state)
            except tk.TclError:  # pragma: no cover
                pass

    def _set_home_meta(self, text: str) -> None:
        try:
            self.home_result_info_var.set(text or "")
        except (tk.TclError, AttributeError):  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # 首页：输出模式（直出 / 对人 / 对AI）
    # ------------------------------------------------------------------
    def _quick_mode_config(self) -> Dict[str, str]:
        mode = getattr(self, "_quick_mode", "direct")
        return QUICK_MODES.get(mode, QUICK_MODES["direct"])

    def _sync_quick_mode_from_settings(self, *, refresh: bool = True) -> None:
        """把持久化的 quick_mode 同步成当前模式（非法值回退直出）。"""
        mode = (getattr(self.settings, "quick_mode", "") or "direct").strip().lower()
        if mode not in QUICK_MODES:
            mode = "direct"
        self._quick_mode = mode
        try:
            self.quick_mode_hint_var.set(QUICK_MODES[mode]["hint"])
        except (AttributeError, tk.TclError):  # pragma: no cover - 构建期
            pass
        if refresh:
            self._refresh_quick_mode_buttons()

    def _refresh_quick_mode_buttons(self) -> None:
        mode = getattr(self, "_quick_mode", "direct")
        busy = getattr(self, "_quick_state", "idle") != "idle" or bool(
            getattr(self, "_processing", False)
        )
        for key, button in getattr(self, "quick_mode_buttons", {}).items():
            selected = key == mode
            try:
                button.configure(
                    bg=COLORS["primary"] if selected else COLORS["card_alt"],
                    fg=COLORS["primary_ink"] if selected else COLORS["text"],
                    activebackground=COLORS["primary_dark"] if selected else COLORS["border"],
                    activeforeground=COLORS["primary_ink"] if selected else COLORS["text"],
                    disabledforeground=COLORS["primary_ink"] if selected else COLORS["faint"],
                    state="disabled" if busy else "normal",
                    cursor="arrow" if busy else "hand2",
                )
            except tk.TclError:  # pragma: no cover - 窗口销毁中
                continue

    def _set_quick_mode(self, mode: str) -> None:
        """切换输出模式并持久化；录音/转换过程中不允许切换。"""
        if mode not in QUICK_MODES:
            return
        if getattr(self, "_quick_state", "idle") != "idle" or getattr(self, "_processing", False):
            return
        if mode == getattr(self, "_quick_mode", ""):
            return
        self._quick_mode = mode
        self.settings.quick_mode = mode
        self._schedule_save()
        try:
            self.quick_mode_hint_var.set(QUICK_MODES[mode]["hint"])
        except (AttributeError, tk.TclError):  # pragma: no cover
            pass
        self._refresh_quick_mode_buttons()

    # ------------------------------------------------------------------
    # 首页：文本区
    # ------------------------------------------------------------------
    def _show_home_placeholder(self) -> None:
        self.home_text.delete("1.0", "end")
        self.home_text.insert("1.0", HOME_PLACEHOLDER, "placeholder")
        self._home_placeholder = True

    def _on_home_text_focus_in(self, _event: Any = None) -> None:
        if self._home_placeholder:
            self.home_text.delete("1.0", "end")
            self._home_placeholder = False

    def _on_home_text_focus_out(self, _event: Any = None) -> None:
        if not self._home_placeholder and not self.home_text.get("1.0", "end-1c").strip():
            self._show_home_placeholder()
        self._schedule_glossary_learning()

    # ------------------------------------------------------------------
    # 首页：直出模式的词语自动学习
    # ------------------------------------------------------------------
    def _load_glossary_entries(self) -> None:
        try:
            self._glossary_entries = glossary_module.load_store()
        except Exception:  # pragma: no cover - 存储异常不应影响启动
            log.exception("加载词语表失败")
            self._glossary_entries = []
        try:
            self.pipeline.set_glossary_entries(self._glossary_entries)
        except Exception:  # pragma: no cover
            log.debug("注入词语表到 pipeline 失败", exc_info=True)

    def _reload_glossary_entries(self) -> None:
        """设置页管理词语表后重新加载。"""
        self._load_glossary_entries()

    def _schedule_glossary_learning(self, _event: Any = None) -> None:
        """用户停止编辑 1.2 秒后，比较基线与修改后的文本。"""
        if not bool(getattr(self.settings, "auto_glossary_learn", True)):
            return
        if self._quick_state != "idle" or self._home_placeholder:
            return
        if not self._learn_baseline_text:
            return
        if getattr(self, "_learn_baseline_mode", "") != "direct":
            return
        if self._learn_after_id:
            try:
                self.root.after_cancel(self._learn_after_id)
            except tk.TclError:  # pragma: no cover
                pass
        self._learn_after_id = self.root.after(1200, self._run_glossary_learning)

    def _flush_glossary_learning(self) -> None:
        if self._learn_after_id:
            try:
                self.root.after_cancel(self._learn_after_id)
            except tk.TclError:  # pragma: no cover
                pass
            self._learn_after_id = None
        self._run_glossary_learning()

    def _run_glossary_learning(self) -> None:
        self._learn_after_id = None
        baseline = self._learn_baseline_text
        if not baseline or self._home_placeholder:
            return
        if not bool(getattr(self.settings, "auto_glossary_learn", True)):
            return
        if getattr(self, "_learn_baseline_mode", "") != "direct":
            return
        current = self._get_home_text()
        if not current or current == baseline:
            return
        try:
            result = glossary_module.extract_candidates(baseline, current)
            if not result.candidates:
                self._learn_baseline_text = current
                return
            manual = glossary_module.parse_manual_glossary(self.settings.glossary)
            entries, pending, conflicts = glossary_module.merge_candidates(
                self._glossary_entries, result.candidates, manual_entries=manual
            )
            # 只持久化学习条目；手动术语表仍以 settings.glossary 为准
            learned_entries = [
                entry for entry in entries if entry.source == "learned"
            ]
            glossary_module.save_store(learned_entries)
            self._glossary_entries = learned_entries
            self.pipeline.set_glossary_entries(learned_entries)
            self._learn_baseline_text = current
            accepted = [
                candidate
                for candidate in result.candidates
                if any(
                    entry.wrong == candidate.wrong
                    and entry.correct == candidate.correct
                    and entry.enabled
                    and entry.source == "learned"
                    for entry in learned_entries
                )
            ]
            pending_learned = [
                entry
                for entry in learned_entries
                if not entry.enabled
            ]
            if accepted:
                pairs = "、".join(
                    f"{item.wrong}→{item.correct}" for item in accepted[:3]
                )
                self._set_status(f"已学习词语：{pairs}（设置 → 词语学习 可管理）")
            elif pending_learned:
                self._set_status(
                    f"已记录 {len(pending_learned)} 条候选词语，待确认"
                    "（设置 → 词语学习）"
                )
            if conflicts:
                log.warning("词语学习出现 %d 条冲突，已进入设置页冲突区", len(conflicts))
            log.info(
                "词语学习完成：抽取=%d 采纳=%d 待确认=%d 冲突=%d",
                len(result.candidates),
                len(accepted),
                len(pending_learned),
                len(conflicts),
            )
        except Exception:  # pragma: no cover - 学习失败不应影响主流程
            log.exception("词语自动学习失败")

    def _reset_glossary_learning(self) -> None:
        if self._learn_after_id:
            try:
                self.root.after_cancel(self._learn_after_id)
            except tk.TclError:  # pragma: no cover
                pass
            self._learn_after_id = None
        self._learn_baseline_text = ""
        self._learn_baseline_mode = ""

    def _set_learning_baseline(self, text: str, mode: str) -> None:
        """转换完成后记录基线；只有用户手动修改才会触发学习。"""
        self._learn_baseline_text = text if mode == "direct" else ""
        self._learn_baseline_mode = mode if mode == "direct" else ""
        self._learn_after_id = None

    def _set_home_text(self, text: str) -> None:
        self.home_text.delete("1.0", "end")
        if text:
            self.home_text.insert("1.0", text)
            self._home_placeholder = False
        else:
            self.home_text.insert("1.0", HOME_PLACEHOLDER, "placeholder")
            self._home_placeholder = True
        self._sync_home_text_buttons()

    def _get_home_text(self) -> str:
        if self._home_placeholder:
            return ""
        return self.home_text.get("1.0", "end-1c").strip()

    def _clear_home_text(self) -> None:
        self._reset_glossary_learning()
        self._show_home_placeholder()
        self._set_home_meta("")
        self._set_home_status(HOME_IDLE_HINT)
        self._set_status("语音输入：已清空转换结果")
        self._sync_home_text_buttons()

    def _copy_home_text(self, *, silent: bool = False) -> bool:
        # 复制前先结算用户对直出结果的修改（学习词语）
        self._flush_glossary_learning()
        text = self._get_home_text()
        if not text:
            if not silent:
                messagebox.showinfo(
                    "还没有文字",
                    "先说一句话，或在下方文本框中输入文字后再复制。",
                    parent=self.root,
                )
            return False
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update_idletasks()
        except tk.TclError as exc:  # pragma: no cover - 剪贴板不可用
            if not silent:
                messagebox.showerror(
                    "复制失败", f"无法访问剪贴板：{exc}", parent=self.root
                )
            return False
        if not silent:
            self._set_home_status(f"已复制 {len(text)} 字到剪贴板")
            self._set_status(f"语音输入：已复制 {len(text)} 字到剪贴板")
        return True

    def _on_home_copy_shortcut(self, _event: Any = None) -> str:
        self._copy_home_text()
        return "break"

    # ------------------------------------------------------------------
    # 首页：状态显示
    # ------------------------------------------------------------------
    def _set_home_status(self, text: str) -> None:
        try:
            self.home_status_var.set(text)
        except tk.TclError:  # pragma: no cover
            pass

    def _set_home_timer(self, text: Optional[str]) -> None:
        try:
            if text:
                self.home_timer_var.set(text)
                if not self.home_timer_label.winfo_manager():
                    self.home_timer_label.pack(before=self.home_status_label)
            else:
                self.home_timer_var.set("")
                self.home_timer_label.pack_forget()
        except tk.TclError:  # pragma: no cover
            pass

    def _show_home_inline_hint(self, text: str, *, warning: bool = True) -> None:
        try:
            self.home_inline_hint.configure(
                text=text, fg=COLORS["warning"] if warning else COLORS["muted"]
            )
            self.home_inline_hint.grid()
        except tk.TclError:  # pragma: no cover
            pass

    def _hide_home_inline_hint(self) -> None:
        try:
            self.home_inline_hint.grid_remove()
        except tk.TclError:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # 首页：波形绘制
    # ------------------------------------------------------------------
    def _wave_canvas_size(self) -> Tuple[int, int]:
        try:
            width = self.wave_canvas.winfo_width()
            height = self.wave_canvas.winfo_height()
        except tk.TclError:  # pragma: no cover
            width, height = 0, 0
        if width <= 10:
            width = self.theme.px(860)
        if height <= 10:
            height = self.theme.px(84)
        return int(width), int(height)

    def _on_wave_configure(self, _event: Any = None) -> None:
        if self._quick_state == "recording":
            self._draw_home_recording(self._last_levels)
        elif self._quick_state == "converting":
            self._draw_home_converting()
        else:
            self._draw_home_idle()

    def _draw_home_idle(self, message: Optional[str] = None) -> None:
        """空闲：只画底部基线/细刻度，不放任何文字（保持界面安静）。"""
        try:
            self.wave_canvas.delete("all")
            width, height = self._wave_canvas_size()
            theme = self.theme
            baseline = height - theme.px(6)
            self.wave_canvas.create_line(
                0, baseline, width, baseline, fill=COLORS["border"], width=1
            )
            step = theme.px(46)
            x = step
            while x < width:
                self.wave_canvas.create_line(
                    x, baseline - theme.px(3), x, baseline,
                    fill=COLORS["grid"], width=1,
                )
                x += step
        except tk.TclError:  # pragma: no cover - 窗口销毁中
            pass

    def _draw_home_recording(self, levels: List[float]) -> None:
        """录音：非对称波形，基线在底部，只向上生长。"""
        try:
            self.wave_canvas.delete("all")
            width, height = self._wave_canvas_size()
            theme = self.theme
            baseline = height - theme.px(6)
            max_height = max(theme.px(8), height - theme.px(12))
            self.wave_canvas.create_line(
                0, baseline, width, baseline, fill=COLORS["border"], width=1
            )
            values = list(levels or [])[-WAVE_BARS:]
            if not values:
                return
            slot = width / float(WAVE_BARS)
            bar_width = max(theme.px(2), int(slot * 0.55))
            min_height = theme.px(2)
            offset = WAVE_BARS - len(values)
            for index, value in enumerate(values):
                x = (offset + index + 0.5) * slot
                level = max(0.0, min(1.0, value))
                if level < 0.01:  # 噪声门：安静时只留最小柱
                    amplitude = min_height
                else:
                    # 视觉映射：保留顶部 20% 余量，正常说话不会顶到上沿
                    visual = _wave_visual(level)
                    amplitude = max(min_height, visual * max_height)
                ratio = (index + 1) / float(len(values))
                color = _blend_color(COLORS["wave_old"], COLORS["wave_new"], ratio)
                self.wave_canvas.create_line(
                    x,
                    baseline - amplitude,
                    x,
                    baseline,
                    fill=color,
                    width=bar_width,
                    capstyle="round",
                )
        except tk.TclError:  # pragma: no cover - 窗口销毁中
            pass

    def _draw_home_converting(self) -> None:
        """转换中：定格已录音波形（暗青），叠加一条扫过的青色扫描线。"""
        try:
            self.wave_canvas.delete("all")
            width, height = self._wave_canvas_size()
            theme = self.theme
            baseline = height - theme.px(6)
            max_height = max(theme.px(8), height - theme.px(12))
            self.wave_canvas.create_line(
                0, baseline, width, baseline, fill=COLORS["border"], width=1
            )
            values = list(self._last_levels or [])[-WAVE_BARS:]
            if values:
                slot = width / float(WAVE_BARS)
                bar_width = max(theme.px(2), int(slot * 0.5))
                offset = WAVE_BARS - len(values)
                for index, value in enumerate(values):
                    x = (offset + index + 0.5) * slot
                    level = max(0.0, min(1.0, value))
                    amplitude = max(
                        theme.px(1), _wave_visual(level) * max_height
                    )
                    self.wave_canvas.create_line(
                        x,
                        baseline - amplitude,
                        x,
                        baseline,
                        fill=COLORS["wave_frozen"],
                        width=bar_width,
                        capstyle="round",
                    )
            # 扫描线：轻微脉冲，无文字
            phase = (self._wave_phase * 0.02) % 1.0
            scan_x = phase * width
            self.wave_canvas.create_line(
                scan_x,
                baseline - max_height,
                scan_x,
                baseline,
                fill=COLORS["primary"],
                width=theme.px(1),
            )
        except tk.TclError:  # pragma: no cover - 窗口销毁中
            pass

    # ------------------------------------------------------------------
    # 首页：录音 / 转换状态机
    # ------------------------------------------------------------------
    def _recording_elapsed(self) -> float:
        try:
            if self._recorder is not None:
                return max(0.0, float(self._recorder.elapsed_seconds))
        except Exception:  # pragma: no cover - 录音器状态不可用
            pass
        return max(0.0, time.monotonic() - self._quick_started)

    def _quick_elapsed(self) -> float:
        return max(0.0, time.monotonic() - self._quick_started)

    def _toggle_quick(self) -> None:
        if self._closing:
            return
        if self._quick_state == "recording":
            self._quick_stop_and_convert()
        elif self._quick_state == "converting":
            return
        else:
            self._quick_start()

    def _on_hotkey(self) -> None:
        """全局 Alt+~：弹出主窗口，并开始/停止录音（托盘隐藏时也生效）。"""
        if self._quitting:
            return
        if not self._window_visible():
            self._show_window_from_tray()
        else:
            try:
                self.root.deiconify()
                self.root.state("normal")
                self.root.lift()
                self.root.focus_force()
            except tk.TclError:  # pragma: no cover
                pass
        self._toggle_quick()

    def _resolve_input_device(self) -> Optional[int]:
        """把设置里保存的麦克风名字解析为 sounddevice 设备索引。

        找不到（设备被拔掉/改名）时回退到系统默认输入设备，并记录日志。
        """
        name = (self.settings.input_device or "").strip()
        if not name or self._recorder is None:
            return None
        try:
            for index, device_name in Recorder.list_input_devices():
                if device_name == name:
                    return int(index)
        except Exception:  # pragma: no cover - 设备枚举失败
            log.debug("枚举麦克风设备失败", exc_info=True)
        log.warning("找不到已保存的麦克风 %r，改用系统默认输入设备", name)
        return None

    def _quick_start(self) -> None:
        if self._quick_state != "idle":
            return
        if self._processing:
            messagebox.showinfo(
                "文件转写进行中",
                "批量文件转写还在进行，请等它完成后再使用语音输入。",
                parent=self.root,
            )
            return
        # 开始新录音前结算上一次直出结果的修改，避免丢失学习数据
        self._flush_glossary_learning()
        self._reset_glossary_learning()
        if self._recorder is None:
            messagebox.showinfo(
                "录音不可用",
                self._recorder_note
                or "未检测到可用的麦克风。请检查系统麦克风设备后重新打开程序。",
                parent=self.root,
            )
            return
        try:
            self._recorder.start(device=self._resolve_input_device())
        except NotImplementedError:
            messagebox.showinfo(
                "录音暂不可用", "录音模块尚未完成，请稍后再试。", parent=self.root
            )
            return
        except VoxError as exc:
            detail = str(exc)
            if exc.user_hint:
                detail += f"\n\n提示：{exc.user_hint}"
            messagebox.showerror("无法开始录音", detail, parent=self.root)
            return
        except Exception as exc:  # pragma: no cover - 设备级异常
            log.exception("启动录音失败")
            messagebox.showerror(
                "无法开始录音",
                f"启动录音失败：{exc}\n\n请检查麦克风是否被其他程序占用。",
                parent=self.root,
            )
            return

        self._quick_state = "recording"
        self._quick_has_key = self.settings.has_api_key
        self._quick_refine_used = False
        self._quick_started = time.monotonic()
        self._last_levels = []
        self._level_history = []
        self._wave_phase = 0
        self._record_peak = 0.0
        self._silence_warned = False
        self._set_home_meta("")
        self._set_home_timer("00:00")
        self._set_home_status(self._listening_status())
        self._hide_home_inline_hint()
        self._refresh_action_states()
        self._draw_home_recording([])
        self._start_wave_tick()
        self._set_status(
            f"语音输入：正在录音…（再按 {HOTKEY_LABEL} 或点击按钮结束）"
        )

    def _quick_stop_and_convert(self) -> None:
        if self._quick_state != "recording":
            return
        self._stop_wave_tick()
        wav_path: Optional[Path] = None
        try:
            if self._recorder is not None:
                wav_path = self._recorder.stop()
        except NotImplementedError:
            wav_path = None
        except VoxError as exc:
            detail = str(exc)
            if exc.user_hint:
                detail += f"\n\n提示：{exc.user_hint}"
            self._reset_quick_idle(
                status="录音没有保存，请再试一次。",
                wave_hint="录音未保存，可以重新开始",
            )
            messagebox.showerror("录音保存失败", detail, parent=self.root)
            return
        except Exception as exc:  # pragma: no cover - 设备级异常
            log.exception("停止录音失败")
            self._reset_quick_idle(
                status="录音保存失败，请再试一次。",
                wave_hint="录音保存失败，可以重新开始",
            )
            messagebox.showerror(
                "录音保存失败", f"无法保存录音：{exc}", parent=self.root
            )
            return

        if wav_path is None:
            self._reset_quick_idle(
                status="没有录到声音，请再试一次。",
                wave_hint="没有录到声音，可以重新开始",
            )
            return
        self._start_quick_conversion(Path(wav_path))

    def _start_quick_conversion(self, audio_path: Path) -> None:
        """后台执行「转写 + DeepSeek 整理」，不写文件。"""
        refine = self.settings.has_api_key
        mode_key = getattr(self, "_quick_mode", "direct")
        if mode_key not in QUICK_MODES:
            mode_key = "direct"
        mode_config = QUICK_MODES[mode_key]
        self._quick_mode_key = mode_key
        job = JobOptions(
            refine=refine,
            style=mode_config["style"],
            language=self.settings.language,
            export_formats=[],
            reasoning_effort=mode_config["reasoning_effort"],
        )
        self._quick_state = "converting"
        self._quick_seq += 1
        seq = self._quick_seq
        self._quick_cancel = threading.Event()
        self._quick_started = time.monotonic()
        self._quick_has_key = refine
        self._quick_refine_used = refine
        self._quick_stage_label = "正在转写…"
        self._quick_percent = 0.0
        self._quick_interim_text = ""
        self._wave_phase = 0
        self._set_home_meta("")
        self._set_home_timer("00:00")
        self._set_home_status(f"正在转写…{' 随后 DeepSeek 整理' if refine else ''}")
        if not refine:
            self._show_home_inline_hint(
                "未配置 DeepSeek Key：本次只做本地转写。可在「设置」中填写 Key 启用 AI 整理。"
            )
        else:
            self._hide_home_inline_hint()
        self._refresh_action_states()
        self._draw_home_converting()
        self._start_wave_tick()
        self.progress_var.set(0.0)
        self._set_status(
            "语音输入：正在转写" + ("并整理…" if refine else "（仅本地转写）…")
        )

        emit = lambda event: self.ui_queue.put(("quick_event", seq, event))

        def worker() -> None:
            try:
                result = self.pipeline.transcribe_and_refine(
                    audio_path,
                    job,
                    emit,
                    self._quick_cancel,
                    on_transcript=lambda transcript: self.ui_queue.put(
                        ("quick_transcript", seq, transcript)
                    ),
                )
            except CancelledError:
                self.ui_queue.put(("quick_cancelled", seq, None))
            except Exception as exc:  # 交给主线程弹窗 + 恢复状态
                log.exception("语音输入转换失败")
                self.ui_queue.put(("quick_error", seq, exc))
            else:
                self.ui_queue.put(("quick_done", seq, result))

        self._quick_worker = threading.Thread(
            target=worker, name="voxscribe-quick", daemon=True
        )
        self._quick_worker.start()

    def _reset_quick_idle(self, *, status: str, wave_hint: str = "") -> None:
        self._stop_wave_tick()
        self._quick_state = "idle"
        self._set_home_timer(None)
        self._set_home_status(status)
        self._set_status("语音输入：" + status)
        self._set_home_meta("")
        self.progress_var.set(0.0)
        self._draw_home_idle()
        self._refresh_action_states()

    def _cancel_quick_recording(self) -> None:
        if self._quick_state != "recording":
            return
        try:
            if self._recorder is not None:
                self._recorder.cancel()
        except NotImplementedError:
            pass
        except Exception:  # pragma: no cover - 设备级异常
            log.warning("取消录音失败", exc_info=True)
        self._reset_quick_idle(
            status="录音已取消（未保存）。",
            wave_hint="录音已取消，可以重新开始",
        )

    def _cancel_quick_conversion(self) -> None:
        if self._quick_state != "converting":
            return
        self._quick_cancel.set()
        self._set_home_status("正在取消…")
        self._set_status("语音输入：正在取消转换…")
        self._quick_stage_label = "正在取消…"

    def _on_quick_event(self, event: JobEvent) -> None:
        stage = (event.stage or "").strip()
        if stage in _QUICK_STAGE_WEIGHTS:
            start, end = _QUICK_STAGE_WEIGHTS[stage]
            ratio = event.progress if event.progress is not None else 0.0
            ratio = max(0.0, min(1.0, float(ratio))) if ratio >= 0 else 0.0
            self._quick_percent = start + (end - start) * ratio
            self.progress_var.set(self._quick_percent)
            if stage in ("prepare", "asr"):
                label = "正在转写…"
            elif self._quick_interim_text:
                label = "已转写，正在 DeepSeek 润色…"
            else:
                label = "DeepSeek 整理中…"
            self._quick_stage_label = label
            self._set_home_status(f"{label}  {int(round(self._quick_percent))}%")
            message = _shorten(event.message or label, 44)
            self._set_status(f"语音输入：{label} {message}")
        elif stage == "done":
            self.progress_var.set(100.0)
            self._set_status(f"语音输入：{event.message or '转换完成'}")

    def _on_quick_transcript(self, transcript: Any) -> None:
        """ASR 完成、DeepSeek 开始前先显示原始转写（两段式：先出字，再润色）。"""
        if self._quick_state != "converting":
            return
        text = (getattr(transcript, "text", "") or "").strip()
        if not text:
            return
        self._quick_interim_text = text
        self._set_home_text(text)
        if self._quick_has_key:
            stage_label = "已转写，正在 DeepSeek 润色…"
            self._set_status("语音输入：已转写，正在 DeepSeek 润色…")
        else:
            stage_label = "已转写，正在本地整理…"
            self._set_status("语音输入：已转写，正在本地整理…")
        self._quick_stage_label = stage_label
        self._set_home_status(stage_label)
        mode_label = QUICK_MODES.get(
            getattr(self, "_quick_mode_key", "direct"), QUICK_MODES["direct"]
        )["label"]
        self._set_home_meta(
            f"{mode_label} · {'已转写 · 正在润色…' if self._quick_has_key else '已转写 · 本地整理…'}"
        )
        self._draw_home_converting()

    def _on_quick_done(self, result: JobResult) -> None:
        self._stop_wave_tick()
        self._quick_state = "idle"
        self._quick_interim_text = ""
        # 最终状态一律以 JobResult.refined 为准，避免「未配置 Key」等中间状态串台
        refined = bool(result.refined and (result.refined.text or "").strip())
        text = (result.final_text or "").strip()
        self._set_home_text(text)
        # 记录直出模式的最终稿基线；用户后续手动修改时再学习词语
        self._set_learning_baseline(text, getattr(self, "_quick_mode_key", "direct"))
        elapsed = result.elapsed or self._quick_elapsed()
        chars = len(text)
        copied = False
        if self.settings.auto_copy and text:
            copied = self._copy_home_text(silent=True)
        self._set_home_timer(None)
        # 结果元信息只保留在结果区右上角一处；录音区恢复「按 Alt+~」提示，底部状态栏保持干净
        self._set_home_status(HOME_IDLE_HINT)
        self._set_status("就绪")
        mode_label = QUICK_MODES.get(
            getattr(self, "_quick_mode_key", "direct"), QUICK_MODES["direct"]
        )["label"]
        meta = (
            f"{mode_label} · {'AI 已整理' if refined else '仅本地转写'} · 用时 "
            f"{elapsed:.1f} 秒 · {chars} 字"
        )
        if copied:
            meta += " · 已复制"
        self._set_home_meta(meta)
        self.progress_var.set(0.0)
        self._draw_home_idle()
        if refined:
            # 元信息已能说明结果状态，不再额外堆小字
            self._hide_home_inline_hint()
        elif not self.settings.has_api_key:
            self._show_home_inline_hint(
                "未配置 DeepSeek Key：本次只做本地转写。可在「设置」中填写 Key 启用 AI 整理。"
            )
        else:
            self._show_home_inline_hint(
                "本次仅本地转写：AI 整理未生效。可检查「设置 → DeepSeek」后重新录音。"
            )
        self._sync_home_text_buttons()
        self._refresh_action_states()

    def _on_quick_cancelled(self) -> None:
        kept = bool(self._get_home_text())
        status = "转换已取消。" if not kept else "转换已取消（已保留已转写的文字）。"
        self._reset_quick_idle(
            status=status,
            wave_hint="转换已取消，可以重新说一段",
        )

    def _on_quick_error(self, exc: BaseException) -> None:
        message = str(exc).strip() or type(exc).__name__
        hint = ""
        if isinstance(exc, VoxError):
            hint = (exc.user_hint or "").strip()
        elif isinstance(exc, NotImplementedError):
            hint = "对应功能模块尚未完成，请稍后再试。"
        else:
            hint = "请查看 data/logs/app.log 了解详细信息。"
        self._reset_quick_idle(
            status="转换失败，请再试一次。",
            wave_hint="转换失败，可以重新开始",
        )
        detail = message
        if hint:
            detail += f"\n\n提示：{hint}"
        messagebox.showerror("语音输入失败", detail, parent=self.root)

    # ------------------------------------------------------------------
    # 首页：波形定时器
    # ------------------------------------------------------------------
    def _start_wave_tick(self) -> None:
        self._stop_wave_tick()
        try:
            self._wave_job = self.root.after(0, self._wave_tick)
        except tk.TclError:  # pragma: no cover
            self._wave_job = None

    def _stop_wave_tick(self) -> None:
        if self._wave_job is not None:
            try:
                self.root.after_cancel(self._wave_job)
            except Exception:  # pragma: no cover
                pass
            self._wave_job = None

    def _wave_tick(self) -> None:
        self._wave_job = None
        if self._closing:
            return
        if self._quick_state == "recording":
            levels = self._read_levels()
            self._last_levels = levels
            self._draw_home_recording(levels)
            self._set_home_timer(format_timestamp(self._recording_elapsed()))
            self._set_home_status(self._listening_status())
            self._check_recording_sound(levels)
        elif self._quick_state == "converting":
            self._wave_phase += 1
            self._draw_home_converting()
            self._set_home_timer(format_timestamp(self._quick_elapsed()))
        else:
            return
        try:
            self._wave_job = self.root.after(self.WAVE_MS, self._wave_tick)
        except tk.TclError:  # pragma: no cover
            self._wave_job = None

    def _check_recording_sound(self, levels: List[float]) -> None:
        """录音 1.5 秒仍没有明显音量时提示检查麦克风，避免“说了没反应”。"""
        peak = max(levels) if levels else 0.0
        self._record_peak = max(getattr(self, "_record_peak", 0.0), peak)
        if getattr(self, "_silence_warned", False):
            if peak >= 0.05:  # 又检测到声音：撤掉警告
                self._silence_warned = False
                self._hide_home_inline_hint()
            return
        if self._recording_elapsed() >= 1.5 and self._record_peak < 0.02:
            self._silence_warned = True
            self._show_home_inline_hint(
                "没有检测到声音：请检查麦克风是否静音、系统是否允许应用使用麦克风，"
                "或在「设置」中换一个麦克风设备。"
            )

    def _read_levels(self) -> List[float]:
        """读取实时音量：优先 Recorder.recent_levels，退化为 level 自行累积。"""
        if self._recorder is None:
            return []
        try:
            getter = getattr(self._recorder, "recent_levels", None)
            if callable(getter):
                raw = getter(WAVE_BARS)
                values = [
                    _clamp01(value)
                    for value in (raw or [])
                    if isinstance(value, (int, float))
                ]
                if values:
                    return values[-WAVE_BARS:]
        except NotImplementedError:  # 队友实现尚未就绪
            pass
        except Exception:  # pragma: no cover - 录音器状态不可用
            log.debug("读取录音波形失败", exc_info=True)
        try:
            level = getattr(self._recorder, "level", None)
            if level is not None:
                self._level_history.append(_clamp01(level))
                del self._level_history[:-WAVE_BARS]
                return list(self._level_history)
        except NotImplementedError:
            pass
        except Exception:  # pragma: no cover
            log.debug("读取录音音量失败", exc_info=True)
        return []

    # ------------------------------------------------------------------
    # 首页：统一刷新按钮状态
    # ------------------------------------------------------------------
    def _refresh_action_states(self) -> None:
        quick_idle = self._quick_state == "idle"
        recorder_ok = self._recorder is not None
        try:
            button = self.home_record_btn
            if self._quick_state == "recording":
                button.configure(
                    text=STOP_BUTTON_TEXT,
                    bg=COLORS["record"],
                    activebackground=COLORS["record_dark"],
                    fg="#ffffff",
                    state="normal",
                    cursor="hand2",
                )
            elif self._quick_state == "converting":
                button.configure(
                    text=CONVERTING_BUTTON_TEXT,
                    bg=COLORS["trough"],
                    activebackground=COLORS["trough"],
                    fg=COLORS["muted"],
                    state="disabled",
                    cursor="arrow",
                )
            else:
                if recorder_ok and not self._processing:
                    button.configure(
                        text=START_BUTTON_TEXT,
                        bg=COLORS["primary"],
                        activebackground=COLORS["primary_dark"],
                        fg=COLORS["primary_ink"],
                        state="normal",
                        cursor="hand2",
                    )
                else:
                    button.configure(
                        text="● 录音不可用",
                        bg=COLORS["trough"],
                        activebackground=COLORS["trough"],
                        fg=COLORS["muted"],
                        state="disabled",
                        cursor="arrow",
                    )
            enabled = (not self._processing) and quick_idle
            self.start_btn.configure(
                state="normal" if enabled else "disabled",
                text="处理中…" if self._processing else "开始处理",
            )
            self.cancel_btn.configure(
                state="normal" if self._processing else "disabled"
            )
            for button in (self.add_btn, self.remove_btn, self.clear_btn):
                button.configure(
                    state="disabled" if self._processing else "normal"
                )
            self.settings_btn.configure(
                state="normal" if (quick_idle and not self._processing) else "disabled"
            )
            self._set_options_enabled(quick_idle and not self._processing)
            self._sync_home_text_buttons()
            self._refresh_quick_mode_buttons()
        except (tk.TclError, AttributeError):  # pragma: no cover - 构建期调用
            return
        self._refresh_tray()

    # ==================================================================
    # 文件转写页
    # ==================================================================
    def _build_file_page(self, parent: tk.Frame) -> tk.Frame:
        theme = self.theme
        colors = COLORS
        page = tk.Frame(parent, bg=colors["bg"])
        page.columnconfigure(0, weight=1)
        page.rowconfigure(0, weight=1)

        main = tk.Frame(page, bg=colors["bg"])
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0, minsize=theme.px(300))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        self._build_file_panel(main)
        self._build_right_panel(main)

        actions = tk.Frame(page, bg=colors["bg"])
        actions.grid(row=1, column=0, sticky="ew", pady=(theme.px(10), 0))
        self.start_btn = self._action_button(
            actions, "开始处理", self._start_processing, primary=True
        )
        self.start_btn.pack(side="right")
        self.cancel_btn = self._action_button(
            actions, "取消", self._cancel_processing, danger=True
        )
        self.cancel_btn.configure(state="disabled")
        self.cancel_btn.pack(side="right", padx=(0, theme.px(8)))
        self.copy_btn = self._action_button(
            actions, "复制整理稿", self._copy_result
        )
        self.copy_btn.pack(side="left")
        self.open_dir_btn = self._action_button(
            actions, "打开输出目录", self._open_output_dir
        )
        self.open_dir_btn.pack(side="left", padx=(theme.px(8), 0))
        return page

    # ------------------------------------------------------------------
    # 文件转写：左侧文件队列
    # ------------------------------------------------------------------
    def _build_file_panel(self, parent: tk.Frame) -> None:
        theme = self.theme
        colors = COLORS
        card, body = theme_module.make_card(parent, theme)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, theme.px(12)))
        self.file_card = card

        head = tk.Frame(body, bg=colors["card"])
        head.pack(fill="x")
        tk.Label(
            head,
            text="待处理文件",
            bg=colors["card"],
            fg=colors["primary_dark"],
            font=theme.font(UI_SIZE + 1, "bold"),
            anchor="w",
        ).pack(side="left")
        self.file_count_var = tk.StringVar(value="共 0 个")
        tk.Label(
            head,
            textvariable=self.file_count_var,
            bg=colors["card"],
            fg=colors["muted"],
            font=theme.font(SMALL_SIZE),
            anchor="e",
        ).pack(side="right")

        toolbar = tk.Frame(body, bg=colors["card"])
        toolbar.pack(fill="x", pady=(theme.px(8), theme.px(8)))
        self.add_btn = self._panel_button(toolbar, "+ 添加文件", self._add_files, accent=True)
        self.add_btn.pack(side="left")
        self.remove_btn = self._panel_button(toolbar, "移除选中", self._remove_selected)
        self.remove_btn.pack(side="left", padx=(theme.px(6), 0))
        self.clear_btn = self._panel_button(
            toolbar, "清空", self._clear_files, danger=True
        )
        self.clear_btn.pack(side="left", padx=(theme.px(6), 0))

        tree_wrap = tk.Frame(body, bg=colors["card"])
        tree_wrap.pack(fill="both", expand=True)
        tree_wrap.columnconfigure(0, weight=1)
        tree_wrap.rowconfigure(0, weight=1)

        self.file_tree = ttk.Treeview(
            tree_wrap,
            columns=("name", "status"),
            show="headings",
            selectmode="extended",
            style="Files.Treeview",
            height=10,
        )
        self.file_tree.heading("name", text="文件名", anchor="w")
        self.file_tree.heading("status", text="状态", anchor="center")
        self.file_tree.column("name", width=theme.px(180), anchor="w", stretch=True)
        self.file_tree.column(
            "status", width=theme.px(76), anchor="center", stretch=False
        )
        scrollbar = ttk.Scrollbar(
            tree_wrap, orient="vertical", command=self.file_tree.yview
        )
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.file_tree.tag_configure("placeholder", foreground=colors["faint"])
        self.file_tree.tag_configure("pending", foreground=colors["muted"])
        self.file_tree.tag_configure("active", foreground=colors["primary_dark"])
        self.file_tree.tag_configure("done", foreground=colors["success"])
        self.file_tree.tag_configure("failed", foreground=colors["danger"])
        self.file_tree.bind("<Delete>", lambda _event: self._remove_selected())
        self.file_tree.bind("<Control-a>", self._select_all_files)

        self.file_hint_var = tk.StringVar(value="支持 mp3 / wav / m4a 等格式 · 音频仅在本机处理")
        tk.Label(
            body,
            textvariable=self.file_hint_var,
            bg=colors["card"],
            fg=colors["faint"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
            justify="left",
            wraplength=theme.px(250),
        ).pack(fill="x", pady=(theme.px(8), 0))

    def _select_all_files(self, _event: Any = None) -> str:
        items = [iid for iid in self.file_tree.get_children() if iid != "__placeholder__"]
        if items:
            self.file_tree.selection_set(items)
        return "break"

    # ------------------------------------------------------------------
    # 文件转写：右侧选项 + 结果预览
    # ------------------------------------------------------------------
    def _build_right_panel(self, parent: tk.Frame) -> None:
        theme = self.theme
        colors = COLORS
        right = tk.Frame(parent, bg=colors["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=1)

        options_card, body = theme_module.make_card(right, theme, "处理选项")
        options_card.grid(row=0, column=0, sticky="ew")
        for column in (1, 3):
            body.columnconfigure(column, weight=1)

        self.refine_var = tk.BooleanVar(value=True)
        self.style_var = tk.StringVar()
        self.language_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        self._style_keys = list(STYLES)
        self._style_values = [STYLES[key] for key in self._style_keys]
        self._language_keys = list(LANGUAGES)
        self._language_values = [LANGUAGES[key] for key in self._language_keys]
        self._model_keys = list(ASR_MODELS)
        self._model_values = [ASR_MODELS[key] for key in self._model_keys]

        field_label = lambda text: tk.Label(  # noqa: E731
            body,
            text=text,
            bg=colors["card"],
            fg=colors["text"],
            font=theme.font(UI_SIZE),
            anchor="e",
        )

        self.refine_check = self._card_checkbutton(
            body,
            "启用 AI 整理（DeepSeek 润色）",
            self.refine_var,
            self._on_option_changed,
        )
        self.refine_check.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=theme.px(4)
        )
        field_label("整理风格").grid(
            row=0, column=2, sticky="e", padx=(theme.px(12), theme.px(6)), pady=theme.px(4)
        )
        self.style_combo = ttk.Combobox(
            body,
            textvariable=self.style_var,
            values=self._style_values,
            state="readonly",
            width=16,
            font=theme.font(UI_SIZE),
        )
        self.style_combo.grid(row=0, column=3, sticky="ew", pady=theme.px(4))
        self.style_combo.bind("<<ComboboxSelected>>", self._on_option_changed)

        field_label("音频语言").grid(
            row=1, column=0, sticky="e", padx=(0, theme.px(6)), pady=theme.px(4)
        )
        self.language_combo = ttk.Combobox(
            body,
            textvariable=self.language_var,
            values=self._language_values,
            state="readonly",
            width=10,
            font=theme.font(UI_SIZE),
        )
        self.language_combo.grid(row=1, column=1, sticky="ew", pady=theme.px(4))
        self.language_combo.bind("<<ComboboxSelected>>", self._on_option_changed)

        field_label("识别模型").grid(
            row=1, column=2, sticky="e", padx=(theme.px(12), theme.px(6)), pady=theme.px(4)
        )
        self.model_combo = ttk.Combobox(
            body,
            textvariable=self.model_var,
            values=self._model_values,
            state="readonly",
            width=26,
            font=theme.font(UI_SIZE),
        )
        self.model_combo.grid(row=1, column=3, sticky="ew", pady=theme.px(4))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_option_changed)

        field_label("导出格式").grid(
            row=2, column=0, sticky="e", padx=(0, theme.px(6)), pady=theme.px(4)
        )
        formats_frame = tk.Frame(body, bg=colors["card"])
        formats_frame.grid(
            row=2, column=1, columnspan=3, sticky="w", pady=theme.px(4)
        )
        self.format_vars: Dict[str, tk.BooleanVar] = {}
        format_labels = {
            "txt": "TXT 文本",
            "md": "Markdown",
            "docx": "Word 文档",
            "srt": "SRT 字幕",
        }
        self.format_checks: List[ttk.Checkbutton] = []
        for fmt in EXPORT_FORMATS:
            var = tk.BooleanVar(value=fmt in ("txt", "md"))
            self.format_vars[fmt] = var
            check = self._card_checkbutton(
                formats_frame,
                format_labels.get(fmt, fmt),
                var,
                self._on_option_changed,
            )
            check.pack(side="left", padx=(0, theme.px(12)))
            self.format_checks.append(check)

        field_label("输出目录").grid(
            row=3, column=0, sticky="e", padx=(0, theme.px(6)), pady=theme.px(4)
        )
        self.output_entry = ttk.Entry(body, textvariable=self.output_dir_var)
        self.output_entry.grid(
            row=3, column=1, columnspan=2, sticky="ew", pady=theme.px(4)
        )
        self.browse_btn = ttk.Button(body, text="浏览…", command=self._browse_output_dir)
        self.browse_btn.grid(
            row=3, column=3, sticky="w", padx=(theme.px(8), 0), pady=theme.px(4)
        )
        self.output_dir_var.trace_add("write", self._on_output_dir_changed)

        results_card, results_body = theme_module.make_card(right, theme, "处理结果")
        results_card.grid(row=1, column=0, sticky="nsew", pady=(theme.px(12), 0))

        info_row = tk.Frame(results_body, bg=colors["card"])
        info_row.pack(fill="x")
        self.result_info_var = tk.StringVar(value="最近完成：—")
        tk.Label(
            info_row,
            textvariable=self.result_info_var,
            bg=colors["card"],
            fg=colors["muted"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        self.notebook = ttk.Notebook(results_body)
        self.notebook.pack(fill="both", expand=True, pady=(theme.px(8), 0))
        refined_tab = tk.Frame(self.notebook, bg=colors["card"])
        raw_tab = tk.Frame(self.notebook, bg=colors["card"])
        self.notebook.add(refined_tab, text="整理稿")
        self.notebook.add(raw_tab, text="原始转写")
        self.refined_view = self._make_preview(refined_tab)
        self.raw_view = self._make_preview(raw_tab)
        self._set_text(self.refined_view, "处理完成后，这里会显示 AI 整理稿正文。")
        self._set_text(
            self.raw_view, "处理完成后，这里会显示带时间戳的原始转写。"
        )

    def _make_preview(self, parent: tk.Frame) -> tk.Text:
        theme = self.theme
        colors = COLORS
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        text = tk.Text(
            parent,
            wrap="word",
            font=theme.font(BODY_SIZE),
            bg=colors["input_bg"],
            fg=colors["text"],
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=theme.px(12),
            pady=theme.px(10),
            spacing1=theme.px(2),
            spacing3=theme.px(4),
            selectbackground=colors["selection"],
            selectforeground=colors["text"],
            insertbackground=colors["primary"],
            cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(state="disabled")
        return text

    @staticmethod
    def _set_text(widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content or "")
        widget.configure(state="disabled")
        widget.yview_moveto(0.0)

    # ------------------------------------------------------------------
    # 全局状态栏
    # ------------------------------------------------------------------
    def _build_status_bar(self, root: tk.Tk) -> None:
        theme = self.theme
        colors = COLORS
        bottom = tk.Frame(root, bg=colors["bg"])
        bottom.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=theme.px(16),
            pady=(theme.px(4), theme.px(10)),
        )
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            bottom,
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
            style="Accent.Horizontal.TProgressbar",
            length=theme.px(200),
        )
        self.progress_bar.pack(side="left", pady=theme.px(3))
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(
            bottom,
            textvariable=self.status_var,
            bg=colors["bg"],
            fg=colors["muted"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=(theme.px(12), theme.px(8)))
        # 内存占用仍保留在 memory_var 中（供诊断/集成测试），但不在主界面显示，避免视觉噪音
        self.memory_var = tk.StringVar(value="内存 --")

    def _set_status(self, message: str) -> None:
        try:
            self.status_var.set(message)
        except tk.TclError:  # pragma: no cover
            pass

    # ==================================================================
    # 快捷键
    # ==================================================================
    def _bind_shortcuts(self) -> None:
        # Alt+~ 是全局快捷键（gui/hotkey.py 轮询全局键盘状态，不占用系统热键）
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Control-Return>", self._on_home_copy_shortcut)
        self.root.bind("<Control-o>", lambda _event: self._add_files())
        self.root.bind("<Control-O>", lambda _event: self._add_files())
        self.root.bind("<F5>", lambda _event: self._start_processing())

    def _on_escape(self, _event: Any = None) -> None:
        if self._quick_state == "recording":
            self._cancel_quick_recording()
        elif self._quick_state == "converting":
            self._cancel_quick_conversion()
        elif self._processing:
            self._cancel_processing()

    # ==================================================================
    # 设置同步
    # ==================================================================
    def _apply_settings_to_ui(self) -> None:
        self._syncing_options = True
        try:
            self.refine_var.set(bool(self.settings.refine_by_default))
            self.style_var.set(
                STYLES.get(self.settings.default_style, STYLES["oral"])
            )
            self.language_var.set(
                LANGUAGES.get(self.settings.language, LANGUAGES["auto"])
            )
            self.model_var.set(
                ASR_MODELS.get(self.settings.asr_model, ASR_MODELS["small"])
            )
            enabled_formats = set(self.settings.export_formats or [])
            for fmt, var in self.format_vars.items():
                var.set(fmt in enabled_formats)
            try:
                default_out = str(default_output_dir())
            except Exception:  # pragma: no cover - 目录创建失败
                default_out = ""
            self.output_dir_var.set(self.settings.output_dir.strip() or default_out)
        finally:
            self._syncing_options = False
        self._sync_quick_mode_from_settings(refresh=False)
        self._refresh_action_states()
        self._sync_home_text_buttons()

    def _on_output_dir_changed(self, *_args: Any) -> None:
        if self._syncing_options:
            return
        value = self.output_dir_var.get().strip()
        try:
            default_out = str(default_output_dir())
        except Exception:  # pragma: no cover
            default_out = ""
        self.settings.output_dir = "" if (not value or value == default_out) else value
        self._schedule_save()

    def _browse_output_dir(self) -> None:
        current = self.output_dir_var.get().strip()
        initial = current if current and Path(current).is_dir() else str(default_output_dir())
        chosen = filedialog.askdirectory(
            parent=self.root, title="选择输出目录", initialdir=initial
        )
        if chosen:
            self.output_dir_var.set(chosen)

    def _on_option_changed(self, *_args: Any) -> None:
        if self._syncing_options or self._closing:
            return
        old_model = self.settings.asr_model
        self.settings.refine_by_default = bool(self.refine_var.get())
        self.settings.default_style = self._current_style_key()
        self.settings.language = self._current_language_key()
        self.settings.asr_model = self._current_model_key()
        self.settings.export_formats = [
            fmt for fmt in EXPORT_FORMATS if self.format_vars[fmt].get()
        ]
        self._schedule_save()
        if self.settings.asr_model != old_model:
            size = self._model_size_hint(self.settings.asr_model)
            hint = f"识别模型已设为 {self.settings.asr_model}"
            if size:
                hint += f"（{size}）"
            self._set_status(hint + "，下次处理时生效")

    def _current_style_key(self) -> str:
        display = self.style_var.get()
        for key, value in STYLES.items():
            if value == display:
                return key
        return self.settings.default_style if self.settings.default_style in STYLES else "oral"

    def _current_language_key(self) -> str:
        display = self.language_var.get()
        for key, value in LANGUAGES.items():
            if value == display:
                return key
        return self.settings.language if self.settings.language in LANGUAGES else "zh"

    def _current_model_key(self) -> str:
        display = self.model_var.get()
        for key, value in ASR_MODELS.items():
            if value == display:
                return key
        return self.settings.asr_model if self.settings.asr_model in ASR_MODELS else "small"

    def _schedule_save(self) -> None:
        if self._closing:
            return
        if self._save_job is not None:
            try:
                self.root.after_cancel(self._save_job)
            except Exception:  # pragma: no cover
                pass
        self._save_job = self.root.after(400, self._save_settings_now)

    def _save_settings_now(self) -> None:
        self._save_job = None
        if self._closing:
            return
        try:
            self.settings.save()
        except Exception as exc:  # pragma: no cover - 磁盘异常
            log.exception("保存设置失败")
            self._set_status(f"保存设置失败：{_shorten(str(exc), 50)}")

    def _set_options_enabled(self, enabled: bool) -> None:
        for widget in [self.refine_check, *self.format_checks]:
            try:
                widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:  # pragma: no cover
                pass
        for combo in (self.style_combo, self.language_combo, self.model_combo):
            try:
                combo.configure(state="readonly" if enabled else "disabled")
            except tk.TclError:  # pragma: no cover
                pass
        try:
            self.output_entry.configure(state="normal" if enabled else "disabled")
            self.browse_btn.configure(state="normal" if enabled else "disabled")
        except tk.TclError:  # pragma: no cover
            pass

    # ==================================================================
    # 文件队列
    # ==================================================================
    def _add_files(self) -> None:
        if self._processing:
            messagebox.showinfo("请稍候", "正在处理文件，暂时不能添加新文件。", parent=self.root)
            return
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="选择音频或视频文件",
            filetypes=AUDIO_FILETYPES,
        )
        if paths:
            self._add_paths(paths)

    def _add_paths(self, raw_paths: Any) -> None:
        added: List[QueueFile] = []
        skipped_duplicate = 0
        invalid: List[str] = []
        existing = {item.path for item in self.files}
        for raw in raw_paths or []:
            try:
                path = Path(str(raw)).expanduser().resolve()
            except (OSError, ValueError):
                invalid.append(str(raw))
                continue
            if not path.exists() or not path.is_file():
                invalid.append(path.name)
                continue
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                invalid.append(path.name)
                continue
            if path in existing:
                skipped_duplicate += 1
                continue
            existing.add(path)
            self._uid += 1
            item = QueueFile(path=path, uid=self._uid)
            self.files.append(item)
            added.append(item)
        self._refresh_file_list()
        if added:
            last = str(added[-1].uid)
            if self.file_tree.exists(last):
                self.file_tree.selection_set(last)
                self.file_tree.see(last)
        messages: List[str] = []
        if added:
            messages.append(f"已添加 {len(added)} 个文件")
        if skipped_duplicate:
            messages.append(f"跳过 {skipped_duplicate} 个重复文件")
        if invalid:
            preview = "、".join(invalid[:3]) + ("…" if len(invalid) > 3 else "")
            messages.append(f"忽略 {len(invalid)} 个不支持的文件（{preview}）")
        if messages:
            self._set_status("；".join(messages))

    def _on_drop(self, event: Any) -> Any:
        try:
            raw = self.root.tk.splitlist(event.data)
        except Exception:
            raw = [event.data]
        if not self._processing:
            self._add_paths(raw)
        return getattr(event, "action", None)

    def _setup_dnd(self) -> None:
        if not self._dnd_available or DND_FILES is None:
            return
        try:
            self.file_tree.drop_target_register(DND_FILES)
            self.file_tree.dnd_bind("<<Drop>>", self._on_drop)
            self._dnd_enabled = True
            self.file_hint_var.set("可拖拽文件到列表 · 音频仅在本机处理")
        except Exception:  # pragma: no cover - 取决于 tkdnd 环境
            log.warning("拖拽添加文件初始化失败，将只支持按钮添加", exc_info=True)
            self._dnd_available = False

    def _refresh_file_list(self) -> None:
        tree = self.file_tree
        for iid in tree.get_children():
            tree.delete(iid)
        if not self.files:
            tree.insert(
                "",
                "end",
                iid="__placeholder__",
                values=("点击「添加文件」添加音频", ""),
                tags=("placeholder",),
            )
        else:
            for item in self.files:
                tree.insert(
                    "",
                    "end",
                    iid=str(item.uid),
                    values=(item.path.name, item.status),
                    tags=(_STATUS_TAGS.get(item.status, "pending"),),
                )
        self.file_count_var.set(f"共 {len(self.files)} 个")

    def _set_file_status(self, item: QueueFile, status: str) -> None:
        item.status = status
        if self._closing:
            return
        try:
            iid = str(item.uid)
            if self.file_tree.exists(iid):
                self.file_tree.set(iid, "status", status)
                self.file_tree.item(
                    iid, tags=(_STATUS_TAGS.get(status, "pending"),)
                )
        except tk.TclError:  # pragma: no cover
            pass

    def _selected_items(self) -> List[QueueFile]:
        selected: List[QueueFile] = []
        by_uid = {item.uid: item for item in self.files}
        for iid in self.file_tree.selection():
            if iid == "__placeholder__":
                continue
            try:
                item = by_uid.get(int(iid))
            except (TypeError, ValueError):
                item = None
            if item is not None:
                selected.append(item)
        return selected

    def _remove_selected(self) -> None:
        if self._processing:
            messagebox.showinfo("请稍候", "正在处理文件，暂时不能移除文件。", parent=self.root)
            return
        selected = self._selected_items()
        if not selected:
            self._set_status("请先在列表中选择要移除的文件")
            return
        selected_ids = {item.uid for item in selected}
        self.files = [item for item in self.files if item.uid not in selected_ids]
        self._refresh_file_list()
        self._set_status(f"已移除 {len(selected)} 个文件")

    def _clear_files(self) -> None:
        if self._processing:
            messagebox.showinfo("请稍候", "正在处理文件，暂时不能清空列表。", parent=self.root)
            return
        if not self.files:
            self._set_status("文件列表已经是空的")
            return
        if not messagebox.askyesno(
            "清空列表",
            f"确定要清空全部 {len(self.files)} 个文件吗？（不会删除原文件）",
            parent=self.root,
        ):
            return
        self.files.clear()
        self._refresh_file_list()
        self._set_status("已清空文件列表")

    # ==================================================================
    # 文件批量处理
    # ==================================================================
    def _resolve_output_dir(self) -> Optional[Path]:
        raw = self.output_dir_var.get().strip()
        try:
            path = Path(raw).expanduser() if raw else self.settings.resolve_output_dir()
        except Exception as exc:  # pragma: no cover - 极端路径
            messagebox.showerror(
                "输出目录不可用", f"无法确定输出目录：\n{exc}", parent=self.root
            )
            return None
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                "输出目录不可用",
                f"无法创建或写入输出目录：\n{path}\n\n{exc}\n\n"
                "请在「设置」中换一个文件夹。",
                parent=self.root,
            )
            return None
        if not path.is_dir():
            messagebox.showerror(
                "输出目录不可用",
                f"输出路径不是文件夹：\n{path}",
                parent=self.root,
            )
            return None
        return path

    def _start_processing(self) -> None:
        if self._processing or self._closing:
            return
        if self._quick_state != "idle":
            messagebox.showinfo(
                "语音输入进行中",
                "请先结束或取消当前录音/转换，再开始文件转写。",
                parent=self.root,
            )
            return
        if not self.files:
            messagebox.showinfo(
                "还没有文件",
                "请先添加要处理的音频文件。也可以回到「语音输入」页直接口述转文字。",
                parent=self.root,
            )
            return

        pending = [item for item in self.files if item.status != "已完成"]
        if not pending:
            if not messagebox.askyesno(
                "全部完成",
                "列表中的文件都已处理完成。是否重新处理一遍？",
                parent=self.root,
            ):
                return
            for item in self.files:
                self._set_file_status(item, "待处理")
            pending = list(self.files)

        formats = [fmt for fmt in EXPORT_FORMATS if self.format_vars[fmt].get()]
        if not formats:
            messagebox.showwarning(
                "请选择导出格式",
                "至少选择一种导出格式（TXT / Markdown / Word / SRT）。",
                parent=self.root,
            )
            return

        out_dir = self._resolve_output_dir()
        if out_dir is None:
            return

        refine_enabled = bool(self.refine_var.get())
        if refine_enabled and not self.settings.get_api_key():
            if not messagebox.askokcancel(
                "未配置 DeepSeek API Key",
                "还没有配置 DeepSeek API Key，AI 整理无法使用。\n\n"
                "点击「确定」先只做本地语音转写；\n"
                "点击「取消」去「设置」中填写 API Key。",
                parent=self.root,
            ):
                self._open_settings()
                return
            refine_enabled = False

        job = JobOptions(
            refine=refine_enabled,
            style=self._current_style_key(),
            language=self._current_language_key(),
            export_formats=formats,
            output_dir=str(out_dir),
            title=None,
        )

        self._process_id += 1
        self._processing = True
        self._cancel_event = threading.Event()
        self._batch = list(pending)
        self._batch_total = len(pending)
        self._batch_index = 0
        self.progress_var.set(0.0)
        for item in pending:
            self._set_file_status(item, "等待中")
        self._set_processing_ui(True)
        self._set_status(f"开始处理 {len(pending)} 个文件 · 输出到 {out_dir}")

        self._worker = threading.Thread(
            target=self._worker_main,
            args=(tuple(pending), job, self._cancel_event, self._process_id),
            name="voxscribe-worker",
            daemon=True,
        )
        self._worker.start()

    def _worker_main(
        self,
        files: Tuple[QueueFile, ...],
        job: JobOptions,
        cancel: threading.Event,
        process_id: int,
    ) -> None:
        """后台单工作线程：顺序处理文件，所有 UI 变化都投递到队列。"""
        total = len(files)
        for index, item in enumerate(files):
            if cancel.is_set():
                for rest in files[index:]:
                    self.ui_queue.put(("file_status", process_id, rest, "未处理"))
                self.ui_queue.put(("batch_done", process_id, "cancelled"))
                return
            self.ui_queue.put(("file_index", process_id, index))
            self.ui_queue.put(("file_status", process_id, item, "处理中"))
            try:
                result = self.pipeline.run(
                    item.path,
                    job,
                    lambda event: self.ui_queue.put(("event", process_id, event)),
                    cancel,
                )
            except CancelledError:
                self.ui_queue.put(("file_status", process_id, item, "已取消"))
                for rest in files[index + 1:]:
                    self.ui_queue.put(("file_status", process_id, rest, "未处理"))
                self.ui_queue.put(("batch_done", process_id, "cancelled"))
                return
            except Exception as exc:  # 单个文件失败不影响后续文件
                log.exception("处理文件失败：%s", item.path)
                self.ui_queue.put(("file_error", process_id, item, exc))
                continue
            self.ui_queue.put(("file_result", process_id, item, result))
        self.ui_queue.put(("batch_done", process_id, "done"))

    def _cancel_processing(self) -> None:
        if not self._processing:
            return
        self._cancel_event.set()
        self.cancel_btn.configure(state="disabled")
        self._set_status("正在取消…（当前阶段结束后停止）")

    def _set_processing_ui(self, processing: bool) -> None:
        self._processing = processing
        self._refresh_action_states()

    # ------------------------------------------------------------------
    # 队列消息处理（主线程）
    # ------------------------------------------------------------------
    def _poll_queue(self) -> None:
        self._poll_job = None
        if self._closing:
            return
        try:
            while True:
                message = self.ui_queue.get_nowait()
                try:
                    self._handle_message(message)
                except Exception:  # pragma: no cover - 防御性
                    log.exception("处理界面消息失败：%r", message)
        except queue.Empty:
            pass
        try:
            self._poll_job = self.root.after(self.POLL_MS, self._poll_queue)
        except tk.TclError:  # pragma: no cover - 窗口已销毁
            pass

    def _handle_message(self, message: Tuple[Any, ...]) -> None:
        kind = message[0]
        if kind == "event":
            _, process_id, event = message
            if process_id == self._process_id:
                self._on_job_event(event)
        elif kind == "file_index":
            _, process_id, index = message
            if process_id == self._process_id:
                self._batch_index = int(index)
        elif kind == "file_status":
            _, process_id, item, status = message
            if process_id == self._process_id:
                self._set_file_status(item, status)
        elif kind == "file_result":
            _, process_id, item, result = message
            if process_id == self._process_id:
                self._on_file_result(item, result)
        elif kind == "file_error":
            _, process_id, item, exc = message
            if process_id == self._process_id:
                self._on_file_error(item, exc)
        elif kind == "batch_done":
            _, process_id, outcome = message
            if process_id == self._process_id:
                self._on_batch_done(outcome)
        elif kind == "quick_event":
            _, seq, event = message
            if seq == self._quick_seq:
                self._on_quick_event(event)
        elif kind == "quick_transcript":
            _, seq, transcript = message
            if seq == self._quick_seq:
                self._on_quick_transcript(transcript)
        elif kind == "quick_done":
            _, seq, result = message
            if seq == self._quick_seq:
                self._on_quick_done(result)
        elif kind == "quick_cancelled":
            _, seq, _payload = message
            if seq == self._quick_seq:
                self._on_quick_cancelled()
        elif kind == "quick_error":
            _, seq, exc = message
            if seq == self._quick_seq:
                self._on_quick_error(exc)
        elif kind == "warmup_done":
            _, seq, ok = message
            if seq == self._warmup_seq:
                self._on_warmup_done(bool(ok))
        elif kind == "warmup_error":
            _, seq, exc = message
            if seq == self._warmup_seq:
                self._on_warmup_error(exc)

    def _overall_percent(self, stage: str, ratio: Optional[float]) -> float:
        start, end = _STAGE_WEIGHTS.get(stage, (0.0, 100.0))
        value = max(0.0, min(1.0, float(ratio))) if ratio is not None and ratio >= 0 else 0.0
        file_percent = start + (end - start) * value
        total = max(1, self._batch_total)
        index = max(0, min(self._batch_index, total - 1))
        return max(0.0, min(100.0, (index + file_percent / 100.0) / total * 100.0))

    def _on_job_event(self, event: JobEvent) -> None:
        stage = (event.stage or "").strip()
        if stage in _STAGE_WEIGHTS:
            self.progress_var.set(self._overall_percent(stage, event.progress))
            label = _STAGE_LABELS[stage]
            index = min(self._batch_index + 1, max(1, self._batch_total))
            total = max(1, self._batch_total)
            message = _shorten(event.message or "", 48)
            if event.progress is not None and event.progress >= 0:
                percent = int(round(event.progress * 100))
                text = f"[{index}/{total}] {label} {percent}% · {message}"
            else:
                text = f"[{index}/{total}] {label} · {message}"
            self._set_status(text.strip(" ·"))
        elif stage == "done":
            self.progress_var.set(
                ((self._batch_index + 1) / max(1, self._batch_total)) * 100.0
            )
            index = min(self._batch_index + 1, max(1, self._batch_total))
            self._set_status(f"[{index}/{max(1, self._batch_total)}] {event.message}")
        elif stage in ("error", "cancelled"):
            if event.message:
                self._set_status(_shorten(event.message, 70))

    def _on_file_result(self, item: QueueFile, result: JobResult) -> None:
        self._set_file_status(item, "已完成")
        self.results.append((item, result))
        self._last_result = result
        self._last_result_item = item
        if result.output_files:
            try:
                self._last_output_dir = Path(result.output_files[0]).parent
            except (OSError, ValueError):  # pragma: no cover
                self._last_output_dir = None
        self._show_result(item, result)
        duration = human_duration(result.transcript.duration or 0.0)
        outputs = len(result.output_files)
        self._set_status(
            f"完成：{item.path.name} · 音频 {duration} · 用时 "
            f"{human_duration(result.elapsed)} · 已导出 {outputs} 个文件"
        )
        self.open_dir_btn.configure(state="normal")
        self.progress_var.set(
            ((self._batch_index + 1) / max(1, self._batch_total)) * 100.0
        )

    def _on_file_error(self, item: QueueFile, exc: BaseException) -> None:
        self._set_file_status(item, "失败")
        message = str(exc).strip() or type(exc).__name__
        hint = ""
        if isinstance(exc, VoxError):
            hint = (exc.user_hint or "").strip()
        elif isinstance(exc, NotImplementedError):
            hint = "对应功能模块尚未完成，请稍后再试。"
        else:
            hint = "请查看 data/logs/app.log 了解详细信息。"
        detail = f"{item.path.name}\n\n{message}"
        if hint:
            detail += f"\n\n提示：{hint}"
        self._set_status(f"失败：{item.path.name} · {_shorten(message, 40)}")
        messagebox.showerror("处理失败", detail, parent=self.root)

    def _on_batch_done(self, outcome: str) -> None:
        self._processing = False
        self._worker = None
        self._batch = []
        self._batch_total = 0
        self._batch_index = 0
        self._set_processing_ui(False)
        if outcome == "cancelled":
            self.progress_var.set(0.0)
            self._set_status("已取消 · 剩余文件标记为未处理")
        else:
            self.progress_var.set(100.0)
            done = sum(1 for item in self.files if item.status == "已完成")
            failed = sum(1 for item in self.files if item.status == "失败")
            summary = f"全部处理完成：成功 {done} 个文件"
            if failed:
                summary += f"，失败 {failed} 个文件"
            if self._last_output_dir is not None:
                summary += f" · 输出目录：{self._last_output_dir}"
            self._set_status(summary)

    # ------------------------------------------------------------------
    # 文件结果预览与操作
    # ------------------------------------------------------------------
    def _show_result(self, item: QueueFile, result: JobResult) -> None:
        raw_text = result.transcript.with_timestamps().strip() or result.transcript.text.strip()
        refined_text = ""
        if result.refined is not None:
            refined_text = (result.refined.text or "").strip()
        if refined_text:
            self._set_text(self.refined_view, refined_text)
            self._set_text(self.raw_view, raw_text or "（没有识别到文本）")
            self.notebook.select(0)
        else:
            self._set_text(
                self.refined_view,
                "本次没有启用 AI 整理。\n\n"
                "可以勾选上方的「启用 AI 整理（DeepSeek 润色）」后重新处理，"
                "也可以在「原始转写」标签页查看识别结果。",
            )
            self._set_text(self.raw_view, raw_text or "（没有识别到文本）")
            self.notebook.select(1)
        duration = human_duration(result.transcript.duration or 0.0)
        chars = len(result.final_text)
        info = (
            f"最近完成：{item.path.name} · 音频 {duration} · "
            f"用时 {human_duration(result.elapsed)} · {chars} 字 · "
            f"输出 {len(result.output_files)} 个文件"
        )
        self.result_info_var.set(_shorten(info, 90))

    def _copy_result(self) -> None:
        result = self._last_result
        if result is None:
            messagebox.showinfo(
                "还没有结果", "请先处理一个音频文件，再复制整理稿。", parent=self.root
            )
            return
        text = result.final_text
        if not text.strip():
            messagebox.showinfo(
                "没有可复制的文字",
                "这次处理没有产生文字内容，请检查音频后重试。",
                parent=self.root,
            )
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update_idletasks()
        except tk.TclError as exc:  # pragma: no cover - 剪贴板不可用
            messagebox.showerror(
                "复制失败", f"无法访问剪贴板：{exc}", parent=self.root
            )
            return
        self._set_status(f"已复制整理稿到剪贴板（{len(text)} 字）")

    def _open_output_dir(self) -> None:
        target: Optional[Path] = self._last_output_dir
        if target is None:
            raw = self.output_dir_var.get().strip()
            try:
                target = Path(raw).expanduser() if raw else self.settings.resolve_output_dir()
            except Exception:  # pragma: no cover
                target = None
        if target is None or not Path(target).exists():
            messagebox.showinfo(
                "还没有输出目录",
                "处理完成一个文件后，就可以在这里打开输出目录了。",
                parent=self.root,
            )
            return
        if reveal_path(target):
            self._set_status(f"已在资源管理器中打开：{target}")
        else:  # pragma: no cover - 系统级失败
            messagebox.showwarning(
                "无法打开目录", f"无法打开文件管理器：\n{target}", parent=self.root
            )

    # ==================================================================
    # 录音设备初始化
    # ==================================================================
    def _init_recorder(self) -> None:
        self._recorder = None
        self._recorder_note = ""
        if not _RECORDER_IMPORT_OK or Recorder is None:
            self._refresh_action_states()
            return
        try:
            recorder = Recorder(recordings_dir=recordings_dir())
            available = bool(recorder.available())
        except NotImplementedError:
            log.debug("录音模块尚未实现，语音输入暂不可用")
            self._refresh_action_states()
            return
        except Exception as exc:
            log.info("录音初始化失败：%s", exc)
            self._recorder_note = f"录音不可用：{_shorten(str(exc), 40)}"
            self._refresh_action_states()
            return
        if not available:
            self._recorder_note = "未检测到可用的麦克风，录音按钮已禁用"
            self._refresh_action_states()
            return
        self._recorder = recorder
        self._refresh_action_states()

    # ==================================================================
    # 设置 / 帮助 / 首次引导
    # ==================================================================
    def _open_settings(self) -> None:
        if self._processing or self._quick_state != "idle":
            messagebox.showinfo(
                "请稍候",
                "正在处理任务，请等待完成或取消后再修改设置。",
                parent=self.root,
            )
            return
        dialog = SettingsDialog(
            self.root,
            settings=self.settings,
            pipeline=self.pipeline,
            theme=self.theme,
            on_saved=self._on_settings_saved,
        )
        if dialog.show():
            log.info("设置已更新")

    def _on_settings_saved(self) -> None:
        self._apply_settings_to_ui()
        # 设置页可能增删/启停学习词条，重新加载后注入 pipeline
        self._reload_glossary_entries()
        self._set_status("设置已保存")
        self._update_memory()

    def _open_help(self) -> None:
        HelpDialog(self.root, self.theme, self.settings).show()

    def _maybe_show_first_run(self) -> None:
        if self._closing or self._hidden_mode:
            return
        if self.settings.first_run_done:
            return
        if self.settings.has_api_key:
            self.settings.first_run_done = True
            self._schedule_save()
            return
        self.settings.first_run_done = True
        self._schedule_save()
        try:
            choice = show_welcome(self.root, self.theme)
        except Exception:  # pragma: no cover - 引导失败不应影响主界面
            log.exception("显示首次使用引导失败")
            return
        if self._closing:
            return
        if choice == "settings":
            self._open_settings()
        elif choice == "transcribe":
            self._set_status(
                "已选择本地转写体验，随时可在「设置」中填写 DeepSeek Key 启用 AI 整理"
            )

    @staticmethod
    def _model_size_hint(model_key: str) -> str:
        match = _MODEL_SIZE_RE.search(ASR_MODELS.get(model_key, "") or "")
        if not match:
            return ""
        return match.group(1).replace(" ", "")

    def _startup_hints(self) -> None:
        if self._closing:
            return
        if self._recorder_note:
            self._set_status(self._recorder_note)

    # ==================================================================
    # 资源监控
    # ==================================================================
    def _update_memory(self) -> None:
        self._memory_job = None
        if self._closing:
            return
        try:
            value = float(process_memory_mb())
        except Exception:  # pragma: no cover - 极端环境
            value = 0.0
        self.memory_var.set(f"内存 {value:.0f} MB" if value > 0 else "内存 --")
        try:
            self._memory_job = self.root.after(self.MEMORY_MS, self._update_memory)
        except tk.TclError:  # pragma: no cover
            pass

    def _schedule_idle_check(self) -> None:
        self._idle_job = None
        if self._closing:
            return
        try:
            self._idle_job = self.root.after(self.IDLE_CHECK_MS, self._idle_check)
        except tk.TclError:  # pragma: no cover
            pass

    def _idle_check(self) -> None:
        self._idle_job = None
        if self._closing:
            return
        try:
            if not self._processing and self.pipeline.maybe_unload_idle():
                self._set_status("模型已卸载，内存已释放")
                self._update_memory()
        except Exception:  # pragma: no cover - 卸载失败不影响界面
            log.exception("空闲卸载检查失败")
        self._schedule_idle_check()

    # ==================================================================
    # 全局快捷键 / 系统托盘 / 显示隐藏
    # ==================================================================
    def _init_hotkey(self) -> None:
        """创建全局 Alt+~ 监听；隐藏测试模式下不启用真实键盘轮询。"""
        try:
            listener = HotkeyListener(
                self.root, self._on_hotkey, interval_ms=50
            )
        except Exception:  # pragma: no cover - 极端环境
            log.warning("初始化全局快捷键失败", exc_info=True)
            self._hotkey = None
            return
        self._hotkey = listener
        if self._hidden_mode:
            return
        if not listener.start():
            log.info("全局快捷键不可用（仅 Windows 支持）")

    # ------------------------------------------------------------------
    # 模型预热（启动后后台加载，保证首句口述也快）
    # ------------------------------------------------------------------
    def _start_warmup(self) -> None:
        """后台预热识别模型；失败只记录日志/状态，绝不弹窗。"""
        self._warmup_job = None
        if self._closing or self._hidden_mode:
            return
        if not bool(getattr(self.settings, "preload_model", True)):
            log.info("设置中已关闭模型预热")
            return
        if self._quick_state != "idle" or self._processing:
            return
        warm_up = getattr(self.pipeline, "warm_up", None)
        if not callable(warm_up):
            return
        self._warming_up = True
        self._warmup_seq += 1
        seq = self._warmup_seq
        self._set_status("正在预热识别模型（首次使用稍等）…")

        def worker() -> None:
            try:
                ok = bool(warm_up())
            except Exception as exc:  # warm_up 约定不抛异常，这里仍兜底
                log.warning("识别模型预热失败", exc_info=True)
                self.ui_queue.put(("warmup_error", seq, exc))
            else:
                self.ui_queue.put(("warmup_done", seq, ok))

        threading.Thread(
            target=worker, name="voxscribe-warmup", daemon=True
        ).start()

    def _on_warmup_done(self, ok: bool) -> None:
        self._warming_up = False
        if ok:
            log.info("识别模型预热完成")
            if self._quick_state == "idle" and not self._processing:
                self._set_status("识别模型已就绪（GPU 预热完成）")
        else:
            log.info("识别模型预热未成功，首次转写时会自动加载")
            if self._quick_state == "idle" and not self._processing:
                self._set_status("模型预热未成功，首次转写时会自动加载")

    def _on_warmup_error(self, exc: BaseException) -> None:
        self._warming_up = False
        log.warning("识别模型预热异常：%s", exc)
        if self._quick_state == "idle" and not self._processing:
            self._set_status("模型预热未成功，首次转写时会自动加载")

    def _listening_status(self) -> str:
        if self._warming_up:
            return "正在聆听，请说话…（识别模型正在预热，首次使用稍等）"
        return "正在聆听，请说话…"

    def _window_visible(self) -> bool:
        try:
            return self.root.state() == "normal"
        except tk.TclError:  # pragma: no cover
            return False

    def _init_tray(self) -> None:
        """创建系统托盘；隐藏测试模式、禁用环境变量或缺少依赖时跳过。"""
        self._tray_job = None
        if self._quitting or self._tray is not None:
            return
        if self._hidden_mode or self._tray_disabled:
            log.info("托盘已禁用（隐藏测试模式或 VOXSCRIBE_DISABLE_TRAY=1）")
            return
        if not tray_available():
            return
        try:
            tray = TrayIcon(
                self.root,
                on_toggle_window=self._toggle_window_from_tray,
                on_toggle_recording=self._toggle_quick_from_tray,
                on_quit=self._quit_from_tray,
                is_recording=lambda: self._quick_state == "recording",
                is_visible=self._window_visible,
            )
            if tray.start():
                self._tray = tray
                self._refresh_tray()
                log.info("系统托盘已就绪")
            else:
                self._tray = None
        except Exception:
            log.warning("创建系统托盘失败，将使用普通关闭逻辑", exc_info=True)
            self._tray = None

    def _refresh_tray(self) -> None:
        tray = self._tray
        if tray is None:
            return
        try:
            tray.update_state()
        except Exception:  # pragma: no cover - 托盘已销毁
            log.debug("刷新托盘状态失败", exc_info=True)

    def _show_window_from_tray(self) -> None:
        if self._quitting:
            return
        try:
            self.root.deiconify()
            self.root.state("normal")
            self.root.lift()
            self.root.focus_force()
        except tk.TclError:  # pragma: no cover
            return
        self._set_status("已从托盘恢复窗口")
        self._refresh_tray()

    def _hide_to_tray(self, *, notify: bool = True) -> bool:
        """隐藏窗口到托盘；没有托盘时返回 False（调用方退回普通关闭）。"""
        tray = self._tray
        if tray is None or self._quitting:
            return False
        try:
            self.root.withdraw()
        except tk.TclError:  # pragma: no cover
            return False
        if notify and not self._tray_hint_shown:
            self._tray_hint_shown = True
            tray.notify(HIDDEN_HINT_TITLE, HIDDEN_HINT_MESSAGE)
        self._refresh_tray()
        return True

    def _toggle_window_from_tray(self) -> None:
        if self._window_visible():
            self._hide_to_tray(notify=False)
        else:
            self._show_window_from_tray()

    def _toggle_quick_from_tray(self) -> None:
        """托盘菜单「开始/停止录音」：先恢复窗口，再开始录音。"""
        if self._quitting:
            return
        if not self._window_visible():
            self._show_window_from_tray()
        self._toggle_quick()
        self._refresh_tray()

    def _quit_from_tray(self) -> None:
        """托盘菜单「退出」：跳过确认，直接销毁。"""
        if self._quitting:
            return
        self._quitting = True
        self._on_close()

    def _on_close_request(self) -> None:
        """点窗口关闭按钮：默认隐藏到托盘；无托盘时正常退出。"""
        if self._quitting:
            return
        if self._hidden_mode or self._tray is None:
            self._on_close()
            return
        if not bool(getattr(self.settings, "close_to_tray", True)):
            self._on_close()
            return
        if not self._hide_to_tray(notify=True):
            self._on_close()

    def _on_window_unmap(self, _event: Any = None) -> None:
        """最小化（iconify）时也隐藏到托盘。"""
        if self._hidden_mode or self._tray is None or self._quitting:
            return
        try:
            self.root.after(80, self._maybe_hide_iconified)
        except tk.TclError:  # pragma: no cover
            pass

    def _maybe_hide_iconified(self) -> None:
        if self._quitting or self._tray is None:
            return
        try:
            state = self.root.state()
        except tk.TclError:  # pragma: no cover
            return
        if state == "iconic":
            self._hide_to_tray(notify=True)

    # ==================================================================
    # 关闭与异常
    # ==================================================================
    def _on_callback_exception(
        self, exc_type: Any, exc_value: Any, exc_tb: Any
    ) -> None:
        log.error("界面回调未捕获异常", exc_info=(exc_type, exc_value, exc_tb))
        if self._closing:
            return
        try:
            messagebox.showerror(
                "程序错误",
                f"发生未预期的错误：\n{exc_value}\n\n"
                "详细信息已写入 data/logs/app.log。",
                parent=self.root,
            )
        except Exception:  # pragma: no cover
            pass

    def _on_close(self) -> None:
        if self._closing:
            return
        if self._hidden_mode or self._quitting:
            # 隐藏测试模式 / 托盘「退出」：直接销毁，绝不弹确认框（避免测试挂住）
            self._finish_close()
            return
        if self._quick_state == "recording":
            if not messagebox.askyesno(
                "正在录音",
                "正在录音，确定要退出吗？\n本次录音将被丢弃。",
                parent=self.root,
            ):
                return
        if self._quick_state == "converting":
            if not messagebox.askyesno(
                "正在转换",
                "正在把语音转换成文字，确定要退出吗？\n本次转换将被中断。",
                parent=self.root,
            ):
                return
        if self._processing:
            if not messagebox.askyesno(
                "正在处理",
                "还有文件正在处理中，确定要退出吗？\n退出会中断当前任务。",
                parent=self.root,
            ):
                return
        self._finish_close()

    def _finish_close(self) -> None:
        """真正的退出清理：停止热键/托盘、取消任务、保存配置、销毁窗口。"""
        if self._closing:
            return
        self._closing = True
        self._quick_cancel.set()
        self._cancel_event.set()

        # 先停全局热键与托盘，避免退出过程中还有后台回调
        try:
            if self._hotkey is not None:
                self._hotkey.stop()
                self._hotkey = None
        except Exception:  # pragma: no cover
            log.debug("退出时停止全局快捷键失败", exc_info=True)
        try:
            if self._tray_job is not None:
                self.root.after_cancel(self._tray_job)
                self._tray_job = None
        except Exception:  # pragma: no cover
            pass
        try:
            if self._tray is not None:
                self._tray.stop()
                self._tray = None
        except Exception:  # pragma: no cover
            log.debug("退出时停止托盘失败", exc_info=True)

        for attribute in (
            "_poll_job",
            "_memory_job",
            "_idle_job",
            "_save_job",
            "_wave_job",
            "_autoclose_job",
            "_warmup_job",
        ):
            job = getattr(self, attribute, None)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except Exception:  # pragma: no cover
                    pass
                setattr(self, attribute, None)
        for job in self._startup_jobs:
            try:
                self.root.after_cancel(job)
            except Exception:  # pragma: no cover
                pass
        self._startup_jobs = []

        try:
            if self._recorder is not None and self._quick_state == "recording":
                self._recorder.cancel()
        except Exception:  # pragma: no cover
            log.warning("退出时取消录音失败", exc_info=True)
        try:
            self.pipeline.unload()
        except Exception:  # pragma: no cover - 卸载失败不应阻止退出
            log.warning("退出时卸载模型失败", exc_info=True)

        try:
            self.settings.window_geometry = self.root.wm_geometry()
            self.settings.save()
        except Exception:  # pragma: no cover - 配置保存失败不阻止退出
            log.exception("退出时保存配置失败")

        try:
            self.root.destroy()
        except Exception:  # pragma: no cover
            pass

    def run(self) -> int:
        """进入 Tk 主循环，退出后返回 0。"""
        self.root.mainloop()
        return 0


# ======================================================================
# 启动入口
# ======================================================================
def _create_root() -> tk.Tk:
    """优先创建支持拖拽的 Tk 根窗口；失败则降级为普通 Tk。

    测试/自动化场景下设置环境变量 ``VOXSCRIBE_GUI_HIDDEN=1``，
    创建的根窗口会立即隐藏，避免在用户桌面上弹出调试窗口。
    """
    global _DND_AVAILABLE
    root: Optional[tk.Tk] = None
    if _DND_IMPORT_OK and TkinterDnD is not None:
        try:
            root = TkinterDnD.Tk()  # type: ignore[misc]
            _DND_AVAILABLE = True
        except Exception:  # pragma: no cover - 取决于 tkdnd 环境
            log.warning("tkinterdnd2 初始化失败，将不支持拖拽添加文件", exc_info=True)
    if root is None:
        _DND_AVAILABLE = False
        root = tk.Tk()
    if os.environ.get("VOXSCRIBE_GUI_HIDDEN", "").strip() == "1":
        try:
            root.withdraw()
        except tk.TclError:  # pragma: no cover
            pass
    return root


def launch() -> int:
    """启动图形界面，返回进程退出码（0 = 正常退出）。"""
    theme_module.enable_dpi_awareness()
    try:
        settings = AppSettings.load()
    except Exception:  # pragma: no cover - config.load 已兜底
        log.exception("读取配置失败，使用默认设置")
        settings = AppSettings()

    root: Optional[tk.Tk] = None
    try:
        root = _create_root()
        window = MainWindow(root, settings)
        return window.run()
    except Exception as exc:
        log.exception("图形界面启动失败")
        try:
            if root is not None:
                messagebox.showerror(
                    "启动失败",
                    f"无法启动图形界面：\n{exc}\n\n"
                    "可以尝试运行 setup.bat 修复依赖，或使用命令行模式："
                    "python -m voxscribe -i 音频文件",
                    parent=root,
                )
        except Exception:  # pragma: no cover
            pass
        print(f"无法启动图形界面：{exc}", file=sys.stderr)
        try:
            if root is not None:
                root.destroy()
        except Exception:  # pragma: no cover
            pass
        return 1


__all__ = ["launch", "MainWindow"]
