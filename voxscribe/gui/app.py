"""图形界面主窗口与启动入口。

启动方式：``python -m voxscribe`` → :func:`voxscribe.gui.app.launch`。

线程模型：
* 主线程只负责界面；
* 点击「开始处理」后创建一个后台单工作线程，顺序处理队列中的文件；
* 后台通过 ``queue.Queue`` 投递 :class:`voxscribe.models.JobEvent` 与
  自定义 UI 消息，主线程每 100ms 轮询一次并刷新界面。

测试辅助：环境变量 ``VOXSCRIBE_GUI_AUTOCLOSE_MS`` 存在时，窗口打开后
指定毫秒自动关闭（用于烟雾测试 / 截图），不影响正常使用。
"""

from __future__ import annotations

import logging
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
    AppSettings,
    default_output_dir,
    recordings_dir,
)
from ..errors import CancelledError, VoxError
from ..models import EXPORT_FORMATS, STYLES, JobEvent, JobOptions, JobResult, format_timestamp
from ..pipeline import Pipeline
from ..util import human_duration, process_memory_mb, reveal_path
from . import theme as theme_module
from .dialogs import HelpDialog, SettingsDialog, show_welcome
from .theme import BODY_SIZE, COLORS, SMALL_SIZE, UI_SIZE, Theme, center_window

log = logging.getLogger(__name__)

APP_TITLE = "语声成文 VoxScribe"
APP_DISPLAY_NAME = "语声成文"
APP_SUBTITLE = "本地转写 · DeepSeek 整理"

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

# 各阶段在整体进度中的权重（百分比）
_STAGE_WEIGHTS = {
    "prepare": (0.0, 4.0),
    "asr": (4.0, 68.0),
    "refine": (68.0, 94.0),
    "export": (94.0, 100.0),
}
_STAGE_LABELS = {"prepare": "准备", "asr": "转写", "refine": "整理", "export": "导出"}

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

#: 可选依赖：录音（由 asr-dev 实现，未就绪时按钮自动禁用）
try:  # pragma: no cover - 取决于队友进度
    from ..recorder import Recorder  # type: ignore

    _RECORDER_IMPORT_OK = True
except Exception:  # pragma: no cover
    Recorder = None  # type: ignore[assignment]
    _RECORDER_IMPORT_OK = False


def _shorten(text: str, limit: int = 60) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


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

    def __init__(self, root: tk.Tk, settings: AppSettings) -> None:
        self.root = root
        self.settings = settings
        self.theme: Theme = theme_module.apply_theme(root)
        self.pipeline = Pipeline(settings)

        # ---- 运行时状态（先初始化，避免回调早于控件创建） ----
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
        self._closing = False
        self._save_job: Optional[str] = None
        self._poll_job: Optional[str] = None
        self._memory_job: Optional[str] = None
        self._idle_job: Optional[str] = None
        self._record_job: Optional[str] = None
        self._recorder: Any = None
        self._recorder_note = ""
        self._recording = False
        self._record_started = 0.0
        self._syncing_options = False
        self._autoclose_ms = _read_autoclose_ms()
        self._dnd_available = _DND_AVAILABLE
        self._dnd_enabled = False

        self._configure_window()
        self._build_ui()
        self._restore_geometry()
        self._setup_dnd()
        self._init_recorder()
        self._apply_settings_to_ui()
        self._refresh_file_list()
        self._bind_shortcuts()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.report_callback_exception = self._on_callback_exception

        self._update_memory()
        self._schedule_idle_check()
        self._poll_queue()
        if self._autoclose_ms:
            log.info("GUI 自动关闭已启用：%d ms（测试模式）", self._autoclose_ms)
            self.root.after(self._autoclose_ms, self._on_close)
        else:
            self.root.after(450, self._maybe_show_first_run)
            self.root.after(700, self._startup_hints)

    # ==================================================================
    # 窗口与布局
    # ==================================================================
    def _configure_window(self) -> None:
        self.root.title(APP_TITLE)
        self.root.option_add("*tearOff", False)
        try:
            self.root.iconname(APP_NAME)
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
        center_window(self.root, theme.px(980), theme.px(720))

    def _build_ui(self) -> None:
        theme = self.theme
        colors = COLORS
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=0)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)

        # ---------- 顶部标题栏 ----------
        header = tk.Frame(root, bg=colors["header"], height=theme.px(80))
        header.grid(row=0, column=0, sticky="ew")
        header.pack_propagate(False)

        brand = tk.Frame(header, bg=colors["header"])
        brand.pack(side="left", padx=(theme.px(22), 0), pady=theme.px(12))
        title_row = tk.Frame(brand, bg=colors["header"])
        title_row.pack(anchor="w")
        tk.Label(
            title_row,
            text=APP_DISPLAY_NAME,
            bg=colors["header"],
            fg=colors["header_fg"],
            font=theme.font(17, "bold"),
        ).pack(side="left")
        tk.Label(
            title_row,
            text="VoxScribe",
            bg=colors["header"],
            fg=colors["header_sub"],
            font=theme.font(12),
        ).pack(side="left", padx=(theme.px(8), 0), pady=(theme.px(4), 0))
        tk.Label(
            brand,
            text=APP_SUBTITLE,
            bg=colors["header"],
            fg=colors["header_sub"],
            font=theme.font(SMALL_SIZE),
        ).pack(anchor="w", pady=(theme.px(1), 0))

        header_actions = tk.Frame(header, bg=colors["header"])
        header_actions.pack(side="right", padx=(0, theme.px(18)), pady=theme.px(12))
        self.help_btn = self._header_button(header_actions, "帮助", self._open_help)
        self.help_btn.pack(side="right", padx=(theme.px(6), 0))
        self.settings_btn = self._header_button(header_actions, "设置", self._open_settings)
        self.settings_btn.pack(side="right", padx=(theme.px(6), 0))
        self.record_btn = self._header_button(
            header_actions, "● 录音", self._toggle_recording, accent=True
        )
        self.record_btn.pack(side="right", padx=(0, theme.px(10)))

        # ---------- 主体 ----------
        main = tk.Frame(root, bg=colors["bg"])
        main.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=theme.px(16),
            pady=(theme.px(14), theme.px(6)),
        )
        main.columnconfigure(0, weight=0, minsize=theme.px(300))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_file_panel(main)
        self._build_right_panel(main)
        self._build_bottom_bar(root)

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
            fg="#f7b0b0" if accent else colors["header_sub"],
            activebackground=colors["header_soft"],
            activeforeground=colors["header_fg"],
            disabledforeground="#7f8ea3",
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
            active_bg, active_fg = "#dbeafe", colors["primary_dark"]
            border = "#bfdbfe"
        elif danger:
            bg, fg = colors["card"], colors["muted"]
            active_bg, active_fg = "#fef2f2", colors["danger"]
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
            bg, fg = colors["primary"], "#ffffff"
            active_bg, active_fg = colors["primary_dark"], "#ffffff"
            border = colors["primary_dark"]
        elif danger:
            bg, fg = colors["card"], colors["danger"]
            active_bg, active_fg = "#fef2f2", colors["danger"]
            border = "#f0c9c9"
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

    # ------------------------------------------------------------------
    # 左侧：待处理文件
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
        body.bind("<Button-1>", self._clear_tree_selection, add="+")

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

    def _clear_tree_selection(self, event: Any = None) -> None:
        if event is not None and event.widget is self.file_tree:
            return
        for iid in self.file_tree.selection():
            self.file_tree.selection_remove(iid)

    # ------------------------------------------------------------------
    # 右侧：处理选项 + 处理结果
    # ------------------------------------------------------------------
    def _build_right_panel(self, parent: tk.Frame) -> None:
        theme = self.theme
        colors = COLORS
        right = tk.Frame(parent, bg=colors["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=1)

        # --- 处理选项 ---
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

        self.refine_check = ttk.Checkbutton(
            body,
            text="启用 AI 整理（DeepSeek 润色）",
            variable=self.refine_var,
            command=self._on_option_changed,
            style="Card.TCheckbutton",
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
            check = ttk.Checkbutton(
                formats_frame,
                text=format_labels.get(fmt, fmt),
                variable=var,
                command=self._on_option_changed,
                style="Card.TCheckbutton",
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

        # --- 处理结果 ---
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
            bg="#fdfefe",
            fg=colors["text"],
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=theme.px(12),
            pady=theme.px(10),
            spacing1=theme.px(2),
            spacing3=theme.px(4),
            selectbackground=colors["primary_light"],
            selectforeground=colors["text"],
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
    # 底部状态栏与操作按钮
    # ------------------------------------------------------------------
    def _build_bottom_bar(self, root: tk.Tk) -> None:
        theme = self.theme
        colors = COLORS
        bottom = tk.Frame(root, bg=colors["bg"])
        bottom.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=theme.px(16),
            pady=(theme.px(2), theme.px(12)),
        )

        status_row = tk.Frame(bottom, bg=colors["bg"])
        status_row.pack(fill="x")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            status_row,
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
            style="Accent.Horizontal.TProgressbar",
            length=theme.px(220),
        )
        self.progress_bar.pack(side="left", pady=theme.px(4))
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(
            status_row,
            textvariable=self.status_var,
            bg=colors["bg"],
            fg=colors["text"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=(theme.px(12), theme.px(8)))
        self.memory_var = tk.StringVar(value="内存 --")
        tk.Label(
            status_row,
            textvariable=self.memory_var,
            bg=colors["bg"],
            fg=colors["faint"],
            font=theme.font(SMALL_SIZE),
            anchor="e",
        ).pack(side="right")

        button_row = tk.Frame(bottom, bg=colors["bg"])
        button_row.pack(fill="x", pady=(theme.px(8), 0))

        self.start_btn = self._action_button(
            button_row, "开始处理", self._start_processing, primary=True
        )
        self.start_btn.pack(side="right")
        self.cancel_btn = self._action_button(
            button_row, "取消", self._cancel_processing, danger=True
        )
        self.cancel_btn.configure(state="disabled")
        self.cancel_btn.pack(side="right", padx=(0, theme.px(8)))
        self.copy_btn = self._action_button(
            button_row, "复制整理稿", self._copy_result
        )
        self.copy_btn.pack(side="left")
        self.open_dir_btn = self._action_button(
            button_row, "打开输出目录", self._open_output_dir
        )
        self.open_dir_btn.pack(side="left", padx=(theme.px(8), 0))

    def _set_status(self, message: str) -> None:
        try:
            self.status_var.set(message)
        except tk.TclError:  # pragma: no cover
            pass

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _event: self._add_files())
        self.root.bind("<Control-O>", lambda _event: self._add_files())
        self.root.bind("<F5>", lambda _event: self._start_processing())
        self.root.bind("<Escape>", self._on_escape)

    def _on_escape(self, _event: Any = None) -> None:
        if self._processing:
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
        self._set_options_enabled(not self._processing)
        self._sync_record_button()

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
        state = ["!disabled"] if enabled else ["disabled"]
        for widget in [self.refine_check, *self.format_checks]:
            try:
                widget.state(state)
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
    # 处理流程
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
        if not self.files:
            messagebox.showinfo(
                "还没有文件",
                "请先添加要处理的音频文件，或点击顶部的「● 录音」现场录制。",
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
        self._set_status(
            f"开始处理 {len(pending)} 个文件 · 输出到 {out_dir}"
        )

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
                    self.ui_queue.put(
                        ("file_status", process_id, rest, "未处理")
                    )
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
                    self.ui_queue.put(
                        ("file_status", process_id, rest, "未处理")
                    )
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
        state = "disabled" if processing else "normal"
        self.start_btn.configure(
            state=state, text="处理中…" if processing else "开始处理"
        )
        self.cancel_btn.configure(state="normal" if processing else "disabled")
        for button in (self.add_btn, self.remove_btn, self.clear_btn, self.settings_btn):
            button.configure(state=state)
        self._set_options_enabled(not processing)
        self._sync_record_button()

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
    # 结果预览与操作
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
    # 录音
    # ==================================================================
    def _init_recorder(self) -> None:
        self._recorder = None
        self._recorder_note = ""
        if not _RECORDER_IMPORT_OK or Recorder is None:
            self._sync_record_button()
            return
        try:
            recorder = Recorder(recordings_dir=recordings_dir())
            available = bool(recorder.available())
        except NotImplementedError:
            log.debug("录音模块尚未实现，录音按钮暂不可用")
            self._sync_record_button()
            return
        except Exception as exc:
            log.info("录音初始化失败：%s", exc)
            self._recorder_note = f"录音不可用：{_shorten(str(exc), 40)}"
            self._sync_record_button()
            return
        if not available:
            self._recorder_note = "未检测到可用的麦克风，录音按钮已禁用"
            self._sync_record_button()
            return
        self._recorder = recorder
        self._sync_record_button()

    def _sync_record_button(self) -> None:
        try:
            if self._recorder is None or self._processing:
                self.record_btn.configure(
                    state="disabled",
                    text="● 录音",
                    bg=COLORS["header"],
                    fg=COLORS["header_sub"],
                )
            elif self._recording:
                elapsed = self._recording_elapsed()
                self.record_btn.configure(
                    state="normal",
                    text=f"■ 停止录音 {format_timestamp(elapsed)}",
                    bg="#a33737",
                    fg="#ffffff",
                )
            else:
                self.record_btn.configure(
                    state="normal",
                    text="● 录音",
                    bg=COLORS["header"],
                    fg="#f7b0b0",
                )
        except tk.TclError:  # pragma: no cover
            pass

    def _recording_elapsed(self) -> float:
        try:
            if self._recorder is not None:
                return max(0.0, float(self._recorder.elapsed_seconds))
        except Exception:  # pragma: no cover - 录音器状态不可用
            pass
        return max(0.0, time.monotonic() - self._record_started)

    def _toggle_recording(self) -> None:
        if self._processing:
            return
        if self._recorder is None:
            messagebox.showinfo(
                "录音不可用",
                self._recorder_note
                or "未检测到可用的麦克风。请检查系统麦克风设备后重新打开程序。",
                parent=self.root,
            )
            return
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        try:
            self._recorder.start()
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
        self._recording = True
        self._record_started = time.monotonic()
        self._sync_record_button()
        self._set_status("录音中… 再次点击「■ 停止录音」结束并加入待处理列表")
        self._tick_recording()

    def _tick_recording(self) -> None:
        self._record_job = None
        if not self._recording or self._closing:
            return
        elapsed = self._recording_elapsed()
        self._sync_record_button()
        self._set_status(
            f"录音中 {format_timestamp(elapsed)} · 点击「■ 停止录音」结束并加入列表"
        )
        try:
            self._record_job = self.root.after(500, self._tick_recording)
        except tk.TclError:  # pragma: no cover
            pass

    def _stop_recording(self) -> None:
        self._recording = False
        if self._record_job is not None:
            try:
                self.root.after_cancel(self._record_job)
            except Exception:  # pragma: no cover
                pass
            self._record_job = None
        saved_path: Optional[Path] = None
        try:
            if self._recorder is not None:
                saved_path = self._recorder.stop()
        except NotImplementedError:
            saved_path = None
        except Exception as exc:  # pragma: no cover - 设备级异常
            log.exception("保存录音失败")
            self._sync_record_button()
            messagebox.showerror(
                "录音保存失败", f"无法保存录音：{exc}", parent=self.root
            )
            self._set_status("录音保存失败")
            return
        self._sync_record_button()
        if saved_path is None:
            self._set_status("录音已取消")
            return
        path = Path(saved_path)
        self._add_paths([str(path)])
        self._set_status(f"录音已保存并加入列表：{path.name}")

    # ==================================================================
    # 设置 / 帮助 / 首次引导
    # ==================================================================
    def _open_settings(self) -> None:
        if self._processing:
            messagebox.showinfo(
                "请稍候", "正在处理文件，请等待完成后再修改设置。", parent=self.root
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
        self._set_status("设置已保存")
        self._update_memory()

    def _open_help(self) -> None:
        HelpDialog(self.root, self.theme, self.settings).show()

    def _maybe_show_first_run(self) -> None:
        if self._closing:
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
            if self.refine_var.get():
                self.refine_var.set(False)
                self._on_option_changed()
            self._set_status("已切换为「只做本地转写」，随时可在「设置」中填写 API Key")

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
            return
        if self.settings.asr_model != "small":
            size = self._model_size_hint(self.settings.asr_model)
            hint = f"识别模型：{self.settings.asr_model}"
            if size:
                hint += f"（{size}）"
            self._set_status(hint + " · 首次使用会自动下载")

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
        if self._processing:
            if not messagebox.askyesno(
                "正在处理",
                "还有文件正在处理中，确定要退出吗？\n退出会中断当前任务。",
                parent=self.root,
            ):
                return
        if self._recording:
            if not messagebox.askyesno(
                "正在录音",
                "正在录音，确定要退出吗？\n本次录音将被丢弃。",
                parent=self.root,
            ):
                return
        self._closing = True
        self._cancel_event.set()

        for attribute in ("_poll_job", "_memory_job", "_idle_job", "_save_job", "_record_job"):
            job = getattr(self, attribute, None)
            if job is not None:
                try:
                    self.root.after_cancel(job)
                except Exception:  # pragma: no cover
                    pass
                setattr(self, attribute, None)

        try:
            self.pipeline.unload()
        except Exception:  # pragma: no cover - 卸载失败不应阻止退出
            log.warning("退出时卸载模型失败", exc_info=True)
        try:
            if self._recorder is not None and self._recording:
                self._recorder.cancel()
        except Exception:  # pragma: no cover
            log.warning("退出时取消录音失败", exc_info=True)

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
    """优先创建支持拖拽的 Tk 根窗口；失败则降级为普通 Tk。"""
    global _DND_AVAILABLE
    if _DND_IMPORT_OK and TkinterDnD is not None:
        try:
            root = TkinterDnD.Tk()  # type: ignore[misc]
            _DND_AVAILABLE = True
            return root
        except Exception:  # pragma: no cover - 取决于 tkdnd 环境
            log.warning("tkinterdnd2 初始化失败，将不支持拖拽添加文件", exc_info=True)
    _DND_AVAILABLE = False
    return tk.Tk()


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
