"""设置 / 欢迎引导 / 帮助对话框。"""

from __future__ import annotations

import logging
import queue
import threading
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Callable, Optional, Tuple

from ..config import ASR_MODELS, AppSettings, default_output_dir
from ..errors import VoxError
from ..pipeline import Pipeline
from .hotkey import HOTKEY_LABEL
from .theme import (
    COLORS,
    SMALL_SIZE,
    UI_SIZE,
    Theme,
    center_window,
    make_card,
)

log = logging.getLogger(__name__)

DEEPSEEK_KEYS_URL = "https://platform.deepseek.com/api_keys"


# ---------------------------------------------------------------------------
# 通用小控件
# ---------------------------------------------------------------------------
def _card_label(
    parent: tk.Misc,
    theme: Theme,
    text: str,
    *,
    muted: bool = False,
    bold: bool = False,
    wraplength: int = 0,
) -> tk.Label:
    return tk.Label(
        parent,
        text=text,
        bg=COLORS["card"],
        fg=COLORS["muted"] if muted else COLORS["text"],
        font=theme.font(SMALL_SIZE if muted else UI_SIZE, "bold" if bold else "normal"),
        anchor="w",
        justify="left",
        wraplength=wraplength or 0,
    )


def _dialog_button(
    parent: tk.Misc,
    theme: Theme,
    text: str,
    command: Callable[[], None],
    *,
    primary: bool = False,
    width_chars: int = 0,
) -> tk.Button:
    """与主界面风格一致的扁平按钮。"""
    if primary:
        bg, fg = COLORS["primary"], COLORS["primary_ink"]
        active_bg, active_fg = COLORS["primary_dark"], COLORS["primary_ink"]
        border = COLORS["primary"]
    else:
        bg, fg = COLORS["card"], COLORS["text"]
        active_bg, active_fg = COLORS["primary_light"], COLORS["primary_dark"]
        border = COLORS["border"]
    button = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground=active_fg,
        disabledforeground=COLORS["faint"],
        font=theme.font(UI_SIZE, "bold" if primary else "normal"),
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        padx=theme.px(16),
        pady=theme.px(6),
        cursor="hand2",
    )
    if width_chars:
        button.configure(width=width_chars)
    return button


# ---------------------------------------------------------------------------
# 设置对话框
# ---------------------------------------------------------------------------
class SettingsDialog(tk.Toplevel):
    """DeepSeek / 性能 / 术语表 / 输出目录设置。"""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        settings: AppSettings,
        pipeline: Pipeline,
        theme: Theme,
        on_saved: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.pipeline = pipeline
        self.theme = theme
        self.on_saved = on_saved
        self.saved = False
        self._closing = False
        self._test_running = False
        self._test_queue: "queue.Queue[Tuple[str, ...]]" = queue.Queue()

        self.title("设置 · 语声成文 VoxScribe")
        self.configure(bg=COLORS["bg"])
        self.transient(parent)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _event: self._on_cancel())

        self._build()
        self._load_values()
        self._fit_geometry()
        self._center_on_parent()

    # -- 构建界面 -----------------------------------------------------
    def _build(self) -> None:
        theme = self.theme
        outer = tk.Frame(self, bg=COLORS["bg"])
        outer.pack(fill="both", expand=True, padx=theme.px(20), pady=(theme.px(14), theme.px(10)))

        # 底部按钮先占位，保证窗口变矮时也可见
        buttons = tk.Frame(outer, bg=COLORS["bg"])
        buttons.pack(side="bottom", fill="x", pady=(theme.px(12), 0))
        _dialog_button(
            buttons, theme, "保存", self._on_save, primary=True, width_chars=8
        ).pack(side="right", padx=(theme.px(8), 0))
        _dialog_button(buttons, theme, "取消", self._on_cancel, width_chars=8).pack(
            side="right"
        )

        # --- DeepSeek ---
        deepseek_card, deepseek_body = make_card(outer, theme, "DeepSeek 文字整理")
        deepseek_card.pack(fill="x", pady=(0, theme.px(8)))
        deepseek_body.columnconfigure(1, weight=1)

        self.api_key_var = tk.StringVar()
        self.api_show_var = tk.BooleanVar(value=False)
        self.base_url_var = tk.StringVar()
        self.api_model_var = tk.StringVar()
        self.key_state_var = tk.StringVar(value="")
        self._has_saved_key = False

        _card_label(deepseek_body, theme, "API Key").grid(
            row=0, column=0, sticky="w", padx=(0, theme.px(10)), pady=theme.px(4)
        )
        self.api_key_entry = ttk.Entry(
            deepseek_body, textvariable=self.api_key_var, show="●", width=30
        )
        self.api_key_entry.grid(row=0, column=1, sticky="ew", pady=theme.px(4))
        ttk.Checkbutton(
            deepseek_body,
            text="显示",
            variable=self.api_show_var,
            command=self._toggle_key_visible,
            style="Card.TCheckbutton",
        ).grid(row=0, column=2, sticky="w", padx=(theme.px(8), 0), pady=theme.px(4))

        self.key_state_label = tk.Label(
            deepseek_body,
            textvariable=self.key_state_var,
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
            justify="left",
            wraplength=theme.px(400),
        )
        self.key_state_label.grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(theme.px(2), theme.px(2))
        )
        self.clear_key_button = ttk.Button(
            deepseek_body, text="清除已保存 Key", command=self._on_clear_api_key
        )
        self.clear_key_button.grid(
            row=1, column=2, sticky="e", pady=(theme.px(2), theme.px(2))
        )

        _card_label(deepseek_body, theme, "Base URL").grid(
            row=2, column=0, sticky="w", padx=(0, theme.px(10)), pady=theme.px(4)
        )
        ttk.Entry(deepseek_body, textvariable=self.base_url_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=theme.px(4)
        )

        _card_label(deepseek_body, theme, "模型名").grid(
            row=3, column=0, sticky="w", padx=(0, theme.px(10)), pady=theme.px(4)
        )
        ttk.Entry(deepseek_body, textvariable=self.api_model_var).grid(
            row=3, column=1, sticky="ew", pady=theme.px(4)
        )
        _card_label(deepseek_body, theme, "默认 deepseek-flash", muted=True).grid(
            row=3, column=2, sticky="w", padx=(theme.px(8), 0)
        )

        hint = (
            "API Key 会用 Windows DPAPI 加密后保存在本机，只有当前 Windows 用户能解密；"
            "只有 AI 整理需要 Key，本地语音转写完全免费。"
        )
        _card_label(
            deepseek_body, theme, hint, muted=True, wraplength=theme.px(500)
        ).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(theme.px(6), 0)
        )

        link_row = tk.Frame(deepseek_body, bg=COLORS["card"])
        link_row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(theme.px(6), 0))
        link = tk.Label(
            link_row,
            text="去 platform.deepseek.com 申请 API Key →",
            bg=COLORS["card"],
            fg=COLORS["primary"],
            font=theme.font(SMALL_SIZE, "bold"),
            cursor="hand2",
        )
        link.pack(side="left")
        link.bind("<Button-1>", lambda _event: webbrowser.open(DEEPSEEK_KEYS_URL))

        self.test_button = ttk.Button(
            link_row, text="测试连接", command=self._on_test_connection
        )
        self.test_button.pack(side="right")

        # --- 识别与性能 ---
        perf_card, perf_body = make_card(outer, theme, "识别与性能")
        perf_card.pack(fill="x", pady=(0, theme.px(8)))
        for column in (1, 3):
            perf_body.columnconfigure(column, weight=1)

        self.threads_var = tk.StringVar()
        self.idle_var = tk.StringVar()
        _card_label(perf_body, theme, "CPU 线程数").grid(
            row=0, column=0, sticky="w", padx=(0, theme.px(10)), pady=theme.px(4)
        )
        ttk.Spinbox(
            perf_body, from_=0, to=64, textvariable=self.threads_var, width=6
        ).grid(row=0, column=1, sticky="w", pady=theme.px(4))
        _card_label(perf_body, theme, "空闲卸载模型").grid(
            row=0, column=2, sticky="w", padx=(theme.px(18), theme.px(10)), pady=theme.px(4)
        )
        ttk.Spinbox(
            perf_body, from_=0, to=600, textvariable=self.idle_var, width=6
        ).grid(row=0, column=3, sticky="w", pady=theme.px(4))

        perf_hint = (
            "线程数 0=自动（最多4）；空闲卸载 0=常驻，>0=空闲该分钟数后释放模型。"
        )

        # 识别模型（语音输入与文件转写共用这一个设置）
        self._model_keys = list(ASR_MODELS)
        self._model_values = [ASR_MODELS[key] for key in self._model_keys]
        self.model_var = tk.StringVar()
        _card_label(perf_body, theme, "识别模型").grid(
            row=1, column=0, sticky="w", padx=(0, theme.px(10)), pady=(theme.px(8), 0)
        )
        self.model_combo = ttk.Combobox(
            perf_body,
            textvariable=self.model_var,
            values=self._model_values,
            state="readonly",
            width=40,
        )
        self.model_combo.grid(
            row=1, column=1, columnspan=3, sticky="w", pady=(theme.px(8), 0)
        )

        # 麦克风设备选择（默认 = 系统默认输入设备）
        self.mic_var = tk.StringVar()
        mic_values = ["默认输入设备"]
        try:
            from ..recorder import Recorder as _Recorder  # noqa: PLC0415

            seen_names = set()
            for _index, _name in _Recorder.list_input_devices():
                if not _name or _name in seen_names:
                    continue
                if _name.startswith(("Microsoft 声音映射器", "主声音", "Input ()")):
                    continue
                if _name.startswith("电脑扬声器"):
                    continue
                seen_names.add(_name)
                mic_values.append(_name)
        except Exception:  # pragma: no cover - 无录音环境
            log.debug("枚举麦克风设备失败", exc_info=True)
        _card_label(perf_body, theme, "麦克风设备").grid(
            row=2, column=0, sticky="w", padx=(0, theme.px(10)), pady=(theme.px(8), 0)
        )
        self.mic_combo = ttk.Combobox(
            perf_body,
            textvariable=self.mic_var,
            values=mic_values,
            state="readonly",
            width=40,
        )
        self.mic_combo.grid(
            row=2, column=1, columnspan=3, sticky="w", pady=(theme.px(8), 0)
        )

        # 语音输入行为（收纳进「识别与性能」卡片，减少界面区块）
        self.auto_copy_var = tk.BooleanVar(
            value=bool(getattr(self.settings, "auto_copy", True))
        )
        ttk.Checkbutton(
            perf_body,
            text="完成后自动复制到剪贴板",
            variable=self.auto_copy_var,
            style="Card.TCheckbutton",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(theme.px(8), 0))
        self.close_to_tray_var = tk.BooleanVar(
            value=bool(getattr(self.settings, "close_to_tray", True))
        )
        ttk.Checkbutton(
            perf_body,
            text="关闭/最小化时隐藏到托盘",
            variable=self.close_to_tray_var,
            style="Card.TCheckbutton",
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=(theme.px(8), 0))
        _card_label(
            perf_body, theme, perf_hint, muted=True, wraplength=theme.px(520)
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(theme.px(6), 0))

        # --- 术语表 ---
        glossary_card, glossary_body = make_card(outer, theme, "术语表（可选）")
        glossary_card.pack(fill="x", pady=(0, theme.px(8)))
        self.glossary_text = tk.Text(
            glossary_body,
            height=2,
            wrap="word",
            font=theme.font(UI_SIZE),
            bg=COLORS["input_bg"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["primary"],
            padx=theme.px(6),
            pady=theme.px(4),
        )
        self.glossary_text.pack(fill="x")
        _card_label(
            glossary_body,
            theme,
            "每行一个术语（可写成「正确写法：常见误写」），整理时尽量保持原文用词。",
            muted=True,
            wraplength=theme.px(500),
        ).pack(anchor="w", pady=(theme.px(4), 0))

        # --- 默认输出 ---
        output_card, output_body = make_card(outer, theme, "默认输出目录")
        output_card.pack(fill="x")
        output_body.columnconfigure(0, weight=1)
        self.output_var = tk.StringVar()
        ttk.Entry(output_body, textvariable=self.output_var).grid(
            row=0, column=0, sticky="ew", pady=theme.px(4)
        )
        ttk.Button(output_body, text="浏览…", command=self._browse_output).grid(
            row=0, column=1, padx=(theme.px(8), 0), pady=theme.px(4)
        )
        _card_label(
            output_body,
            theme,
            "处理结果会导出到这里；留空则使用程序目录下的 outputs/。",
            muted=True,
            wraplength=theme.px(500),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(theme.px(4), 0))

    def _fit_geometry(self) -> None:
        theme = self.theme
        self.update_idletasks()
        width = theme.px(680)
        height = max(theme.px(620), self.winfo_reqheight() + theme.px(8))
        height = min(height, int(self.winfo_screenheight() * 0.92))
        self._dialog_width, self._dialog_height = width, height
        self.geometry(f"{width}x{height}")

    def _center_on_parent(self) -> None:
        try:
            parent = self.master
            assert parent is not None
            width = getattr(self, "_dialog_width", self.winfo_reqwidth())
            height = getattr(self, "_dialog_height", self.winfo_reqheight())
            if parent.winfo_viewable():
                x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
                y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
            else:
                x = (self.winfo_screenwidth() - width) // 2
                y = (self.winfo_screenheight() - height) // 2
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:  # pragma: no cover - 极端环境
            center_window(self, self.winfo_reqwidth(), self.winfo_reqheight())

    # -- 取值 ---------------------------------------------------------
    def _load_values(self) -> None:
        settings = self.settings
        # 安全策略：不在界面回显已保存的 Key；留空保存 = 保持原 Key 不变。
        self._has_saved_key = bool(settings.has_api_key)
        self.api_key_var.set("")
        self._update_key_state()
        self.base_url_var.set(settings.api_base_url or "https://api.deepseek.com")
        self.api_model_var.set(settings.api_model or "deepseek-flash")
        self.model_var.set(
            ASR_MODELS.get(settings.asr_model, ASR_MODELS.get("medium", ""))
        )
        self.threads_var.set(str(int(settings.cpu_threads or 0)))
        self.idle_var.set(str(int(settings.idle_unload_minutes or 0)))
        self.output_var.set(
            settings.output_dir.strip()
            if settings.output_dir.strip()
            else str(default_output_dir())
        )
        self.close_to_tray_var.set(bool(getattr(settings, "close_to_tray", True)))
        self.auto_copy_var.set(bool(getattr(settings, "auto_copy", True)))
        try:
            self.mic_var.set(settings.input_device.strip() or "默认输入设备")
        except Exception:  # pragma: no cover - 旧配置
            self.mic_var.set("默认输入设备")
        if settings.glossary:
            self.glossary_text.delete("1.0", "end")
            self.glossary_text.insert("1.0", settings.glossary)

    def _update_key_state(self) -> None:
        try:
            if self._has_saved_key:
                self.key_state_var.set(
                    "已保存 API Key：●●●●●●●●（留空表示不修改；如需更换，直接输入新 Key）"
                )
                self.key_state_label.configure(fg=COLORS["muted"])
                self.clear_key_button.configure(state="normal")
            else:
                self.key_state_var.set(
                    "尚未配置 API Key（本地转写不需要 Key；AI 整理需要）"
                )
                self.key_state_label.configure(fg=COLORS["faint"])
                self.clear_key_button.configure(state="disabled")
        except tk.TclError:  # pragma: no cover
            pass

    def _on_clear_api_key(self) -> None:
        if not self._has_saved_key:
            return
        if not messagebox.askyesno(
            "清除 API Key",
            "确定要清除本机保存的 DeepSeek API Key 吗？\n\n"
            "清除后语音输入将退化为仅本地转写，可随时重新填写。",
            parent=self,
        ):
            return
        self.settings.set_api_key("")
        self._has_saved_key = False
        self.api_key_var.set("")
        try:
            self.settings.save()
            self.pipeline.apply_settings(self.settings)
        except Exception as exc:  # pragma: no cover - 磁盘异常
            log.exception("清除 API Key 后保存配置失败")
            messagebox.showwarning(
                "已清除但保存失败",
                f"API Key 已从当前设置中清除，但写入配置文件失败：\n{exc}",
                parent=self,
            )
        self._update_key_state()

    def _toggle_key_visible(self) -> None:
        self.api_key_entry.configure(show="" if self.api_show_var.get() else "●")

    def _browse_output(self) -> None:
        initial = self.output_var.get().strip() or str(default_output_dir())
        chosen = filedialog.askdirectory(
            parent=self,
            title="选择默认输出目录",
            initialdir=initial if Path(initial).exists() else None,
        )
        if chosen:
            self.output_var.set(chosen)

    # -- 测试连接（后台线程） ----------------------------------------
    def _on_test_connection(self) -> None:
        if self._test_running:
            return
        entered_key = self.api_key_var.get().strip()
        api_key = entered_key
        if not api_key and self._has_saved_key:
            api_key = (self.settings.get_api_key() or "").strip()
            if api_key:
                self.key_state_var.set("正在使用已保存的 API Key 测试连接…")
        if not api_key:
            messagebox.showwarning(
                "缺少 API Key",
                "请先填写 DeepSeek API Key，再测试连接。\n"
                "Key 可在 platform.deepseek.com 申请（仅 AI 整理需要）。",
                parent=self,
            )
            return
        base_url = self.base_url_var.get().strip() or "https://api.deepseek.com"
        model = self.api_model_var.get().strip() or "deepseek-flash"
        self._test_running = True
        self.test_button.configure(state="disabled", text="测试中…")
        threading.Thread(
            target=self._test_worker,
            args=(api_key, base_url, model),
            name="voxscribe-test-connection",
            daemon=True,
        ).start()
        self.after(150, self._poll_test_result)

    def _test_worker(self, api_key: str, base_url: str, model: str) -> None:
        try:
            from ..refine import DeepSeekRefiner

            refiner = DeepSeekRefiner(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=30,
                max_retries=0,
            )
            reply = refiner.test_connection()
            self._test_queue.put(("ok", str(reply or "")))
        except NotImplementedError:
            self._test_queue.put(
                ("pending", "DeepSeek 整理模块尚未完成，暂时无法测试连接。")
            )
        except VoxError as exc:
            self._test_queue.put(("error", str(exc), getattr(exc, "user_hint", "")))
        except Exception as exc:  # pragma: no cover - 网络等未知错误
            log.exception("测试 DeepSeek 连接失败")
            self._test_queue.put(("error", f"连接失败：{exc}", ""))

    def _poll_test_result(self) -> None:
        if self._closing or not self.winfo_exists():
            return
        try:
            result = self._test_queue.get_nowait()
        except queue.Empty:
            self.after(150, self._poll_test_result)
            return
        self._test_running = False
        self.test_button.configure(state="normal", text="测试连接")
        self._update_key_state()
        kind = result[0]
        if kind == "ok":
            reply = (result[1] or "").strip()
            messagebox.showinfo(
                "连接成功",
                "DeepSeek 连接正常，可以开始整理文字了。\n\n"
                f"模型回复：{reply[:200]}" if reply else "DeepSeek 连接正常。",
                parent=self,
            )
        elif kind == "pending":
            messagebox.showinfo("暂不可用", result[1], parent=self)
        else:
            detail = result[1]
            hint = result[2] if len(result) > 2 else ""
            if hint:
                detail += f"\n\n提示：{hint}"
            messagebox.showerror("连接失败", detail, parent=self)

    # -- 保存 / 关闭 ---------------------------------------------------
    def _on_save(self) -> None:
        try:
            threads = int(self.threads_var.get().strip() or "0")
            idle_minutes = int(self.idle_var.get().strip() or "0")
        except ValueError:
            messagebox.showerror(
                "输入有误", "CPU 线程数和空闲卸载分钟数必须是整数。", parent=self
            )
            return
        if not 0 <= threads <= 64:
            messagebox.showerror("输入有误", "CPU 线程数应在 0 ~ 64 之间。", parent=self)
            return
        if not 0 <= idle_minutes <= 600:
            messagebox.showerror(
                "输入有误", "空闲卸载分钟数应在 0 ~ 600 之间。", parent=self
            )
            return

        base_url = self.base_url_var.get().strip() or "https://api.deepseek.com"
        if not base_url.lower().startswith(("http://", "https://")):
            messagebox.showerror(
                "输入有误",
                "Base URL 需要以 http:// 或 https:// 开头，例如 https://api.deepseek.com",
                parent=self,
            )
            return
        model = self.api_model_var.get().strip() or "deepseek-flash"

        raw_output = self.output_var.get().strip()
        output_path: Optional[Path] = None
        if raw_output:
            output_path = Path(raw_output).expanduser()
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror(
                    "输出目录不可用",
                    f"无法创建或写入该目录：\n{output_path}\n\n{exc}",
                    parent=self,
                )
                return

        entered_key = self.api_key_var.get().strip()
        encrypted = True
        if entered_key:
            encrypted = self.settings.set_api_key(entered_key)
            self._has_saved_key = bool(self.settings.has_api_key)
        elif not self._has_saved_key:
            # 从未配置过 Key：保持空，不影响本地转写
            self.settings.set_api_key("")
        # 留空且已有 Key：保持已保存的 Key 不变（显式清除请点「清除已保存 Key」）
        self.settings.api_base_url = base_url
        self.settings.api_model = model
        self.settings.cpu_threads = threads
        self.settings.idle_unload_minutes = idle_minutes
        self.settings.close_to_tray = bool(self.close_to_tray_var.get())
        self.settings.auto_copy = bool(self.auto_copy_var.get())
        self.settings.glossary = self.glossary_text.get("1.0", "end-1c").strip()
        # 识别模型：语音输入与文件转写共用这一个设置
        model_choice = self.model_var.get().strip()
        for key, label in ASR_MODELS.items():
            if label == model_choice:
                self.settings.asr_model = key
                break
        mic_choice = self.mic_var.get().strip()
        self.settings.input_device = (
            "" if mic_choice in ("", "默认输入设备") else mic_choice
        )
        self.settings.output_dir = str(output_path) if output_path else ""

        try:
            self.settings.save()
            self.pipeline.apply_settings(self.settings)
        except Exception as exc:
            log.exception("保存设置失败")
            messagebox.showerror(
                "保存失败",
                f"无法保存设置：{exc}\n\n请检查 data/ 目录是否有写入权限。",
                parent=self,
            )
            return

        if entered_key and not encrypted:
            messagebox.showwarning(
                "提示",
                "当前系统不支持 DPAPI 加密，API Key 将以受限明文形式保存在本机配置文件中。",
                parent=self,
            )
        self.saved = True
        if self.on_saved is not None:
            try:
                self.on_saved()
            except Exception:  # pragma: no cover - 回调异常不能影响关闭
                log.exception("设置保存回调执行失败")
        self._close()

    def _on_cancel(self) -> None:
        self._close()

    def _close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    # -- 展示 ---------------------------------------------------------
    def show(self) -> bool:
        """模态显示，返回是否保存了设置。"""
        try:
            self.grab_set()
            self.api_key_entry.focus_set()
            self.wait_window()
        except tk.TclError:  # pragma: no cover - 主窗口已销毁
            pass
        return self.saved


# ---------------------------------------------------------------------------
# 首次使用引导
# ---------------------------------------------------------------------------
class WelcomeDialog(tk.Toplevel):
    """首次启动引导：说明本地转写免费、AI 整理需要 Key。"""

    def __init__(self, parent: tk.Misc, theme: Theme) -> None:
        super().__init__(parent)
        self.theme = theme
        self.choice = ""  # "settings" / "transcribe"
        self.title("欢迎使用 · 语声成文 VoxScribe")
        self.configure(bg=COLORS["card"])
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._choose_transcribe)
        self.bind("<Escape>", lambda _event: self._choose_transcribe())

        outer = tk.Frame(self, bg=COLORS["card"])
        outer.pack(fill="both", expand=True, padx=theme.px(28), pady=theme.px(24))

        tk.Label(
            outer,
            text="欢迎使用 语声成文",
            bg=COLORS["card"],
            fg=COLORS["text"],
            font=theme.font(18, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            outer,
            text="VoxScribe · 说出来，就是文字",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
        ).pack(fill="x", pady=(theme.px(2), theme.px(14)))

        tk.Frame(outer, bg=COLORS["border"], height=1).pack(fill="x", pady=(0, theme.px(14)))

        bullets = [
            (
                "点一下，说话就变成文字",
                f"点大按钮或按 {HOTKEY_LABEL} 开始说话，再按一次结束；"
                "几秒后自动得到整理好的文字，不用打字。",
            ),
            (
                "本地转写 + DeepSeek 纠错",
                "语音在本机转写、音频不上传；配置 DeepSeek Key 后自动纠正错别字、去口头禅、补标点。",
            ),
            (
                "可修改、一键复制",
                "结果直接显示在首页下方，可以随手修改并一键复制；已保存的音频也可以用「文件转写」批量处理。",
            ),
        ]
        for title, detail in bullets:
            row = tk.Frame(outer, bg=COLORS["card"])
            row.pack(fill="x", pady=theme.px(6))
            tk.Label(
                row,
                text="●",
                bg=COLORS["card"],
                fg=COLORS["primary"],
                font=theme.font(UI_SIZE, "bold"),
            ).pack(side="left", anchor="n", padx=(0, theme.px(8)))
            text_box = tk.Frame(row, bg=COLORS["card"])
            text_box.pack(side="left", fill="x", expand=True)
            tk.Label(
                text_box,
                text=title,
                bg=COLORS["card"],
                fg=COLORS["text"],
                font=theme.font(UI_SIZE, "bold"),
                anchor="w",
            ).pack(fill="x")
            tk.Label(
                text_box,
                text=detail,
                bg=COLORS["card"],
                fg=COLORS["muted"],
                font=theme.font(SMALL_SIZE),
                anchor="w",
                justify="left",
                wraplength=theme.px(420),
            ).pack(fill="x")

        note = (
            f"还没有 API Key？可以先体验本地转写：按 {HOTKEY_LABEL} 说一段，"
            "文字会直接出现在首页；稍后在「设置」中补充 Key 即可启用 AI 纠错整理。"
        )
        tk.Label(
            outer,
            text=note,
            bg=COLORS["primary_light"],
            fg=COLORS["primary_dark"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
            justify="left",
            wraplength=theme.px(470),
            padx=theme.px(10),
            pady=theme.px(8),
        ).pack(fill="x", pady=(theme.px(14), 0))

        buttons = tk.Frame(outer, bg=COLORS["card"])
        buttons.pack(fill="x", pady=(theme.px(18), 0))
        _dialog_button(
            buttons, theme, "先体验本地转写", self._choose_transcribe, width_chars=14
        ).pack(side="right", padx=(theme.px(8), 0))
        _dialog_button(
            buttons, theme, "去配置 API Key", self._choose_settings, primary=True,
            width_chars=14,
        ).pack(side="right")

        self.update_idletasks()
        width = theme.px(560)
        height = self.winfo_reqheight() + theme.px(8)
        center_window(self, width, height)

    def _choose_settings(self) -> None:
        self.choice = "settings"
        self._close()

    def _choose_transcribe(self) -> None:
        self.choice = "transcribe"
        self._close()

    def _close(self) -> None:
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def show(self) -> str:
        try:
            self.grab_set()
            self.wait_window()
        except tk.TclError:  # pragma: no cover - 主窗口已销毁
            pass
        return self.choice


# ---------------------------------------------------------------------------
# 帮助
# ---------------------------------------------------------------------------
_HELP_TEXT = f"""语音输入：说一句，立刻成文（推荐）

1. 开始说话
   点击首页的大按钮「● 开始录音」，或直接按 {HOTKEY_LABEL}，对着麦克风正常说话。
   快捷键是全局的：窗口在后台、最小化到托盘时也能用。
   录音时波形会实时跳动，顶部显示录音时长，说明麦克风工作正常。

2. 选择输出模式（首页顶部「模式」）
   · 直出：只纠错别字，最大程度保留你的原话和语气；
   · 对人：整理成可以直接发给同事/客户的正式消息；
   · 对AI：整理成结构清晰、可以直接发给 AI 的提示词。
   三种模式分别使用「不思考 / 轻思考 / 深度思考」，选择会被记住。

3. 再按一次结束
   说完后再按一次 {HOTKEY_LABEL}（或点击「■ 停止并转换」）。
   程序会自动：本地 GPU 语音转写 → 按所选模式交给 DeepSeek 整理。
   转写一完成就会先把原始文字显示出来，再等 DeepSeek 整理后原地替换，等待感更小。

4. 得到文字
   几秒后文字出现在首页下方，可以直接修改。
   点「复制文字」或按 Ctrl+Enter 复制，默认会「完成后自动复制到剪贴板」。
   快捷键：{HOTKEY_LABEL} 开始/结束录音；录音中按 Esc = 丢弃本次录音；转换中按 Esc = 取消转换。

文件转写：处理已有的音频 / 视频

· 切换到「📁 文件转写」页，添加 mp3 / wav / m4a / flac 等文件（也可以直接把文件拖进列表）。
· 选择整理风格（口述成文 / 会议纪要 / 逐字稿润色 / 要点摘要）、音频语言、识别模型和导出格式。
· 点击「开始处理」，程序会依次完成：语音转写 → DeepSeek 整理 → 导出文件。
· 导出支持 txt / md / docx / srt，默认保存到程序目录下的 outputs/ 文件夹。

系统托盘常驻

· 点窗口的关闭按钮或最小化窗口，默认会隐藏到系统托盘待机（可在设置中关闭）。
· 隐藏后按 {HOTKEY_LABEL} 仍然可以开始录音；双击托盘图标恢复窗口，右键托盘图标可退出。
· 托盘菜单同样提供「显示/隐藏主界面」「开始/停止录音」「退出」。

轻量说明

· 本地转写免费：语音识别完全在本机 GPU 上完成，音频不会上传到任何服务器。
· AI 整理可选：在「设置」中填写 DeepSeek API Key 后启用；没有 Key 时语音输入会自动退化为仅本地转写。
· API Key 安全：使用 Windows DPAPI 加密保存在本机，只有当前 Windows 用户能解密；设置里留空保存不会清掉已保存的 Key。
· 空闲卸载模型：默认 0（常驻不卸载，保证口述速度）；如需节省内存可在设置中改为指定分钟数。
· CPU 占用可控：默认最多使用 4 个线程，避免占满电脑资源。
· 数据都在本地：配置在 data/config.json，录音在 data/recordings，日志在 data/logs/app.log。

小提示

· 第一次使用会自动加载识别模型，之后会复用本地缓存，速度快很多。
· 识别效果不好时，可以在设置里换成更大的模型（medium / large-v3），但速度会更慢。
· 语音输入的「直出 / 对人 / 对AI」三种模式会记住上次选择，可在首页顶部随时切换。
"""


class HelpDialog(tk.Toplevel):
    """使用帮助。"""

    def __init__(self, parent: tk.Misc, theme: Theme, settings: AppSettings) -> None:
        super().__init__(parent)
        self.theme = theme
        self.settings = settings
        self.title("使用帮助 · 语声成文 VoxScribe")
        self.configure(bg=COLORS["bg"])
        self.transient(parent)

        outer = tk.Frame(self, bg=COLORS["bg"])
        outer.pack(fill="both", expand=True, padx=theme.px(18), pady=theme.px(16))

        buttons = tk.Frame(outer, bg=COLORS["bg"])
        buttons.pack(side="bottom", fill="x", pady=(theme.px(10), 0))
        _dialog_button(buttons, theme, "知道了", self._close, primary=True, width_chars=10).pack(
            side="right"
        )
        _dialog_button(
            buttons, theme, "打开输出目录", self._open_output_dir, width_chars=12
        ).pack(side="right", padx=(0, theme.px(8)))

        text_frame = tk.Frame(
            outer,
            bg=COLORS["card"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        text_frame.pack(fill="both", expand=True)
        text = tk.Text(
            text_frame,
            wrap="word",
            font=theme.font(UI_SIZE),
            bg=COLORS["card"],
            fg=COLORS["text"],
            relief="flat",
            bd=0,
            padx=theme.px(16),
            pady=theme.px(12),
            spacing1=theme.px(2),
            spacing3=theme.px(4),
            cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.insert("1.0", _HELP_TEXT)
        text.configure(state="disabled")

        self.update_idletasks()
        width = theme.px(720)
        height = min(
            theme.px(620),
            int(self.winfo_screenheight() * 0.86),
        )
        center_window(self, width, height)

    def _open_output_dir(self) -> None:
        from ..util import reveal_path

        try:
            target = self.settings.resolve_output_dir()
        except Exception:  # pragma: no cover - 目录创建失败
            target = default_output_dir()
        reveal_path(target)

    def _close(self) -> None:
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def show(self) -> None:
        try:
            self.grab_set()
            self.wait_window()
        except tk.TclError:  # pragma: no cover - 主窗口已销毁
            pass


def show_welcome(parent: tk.Misc, theme: Theme) -> str:
    """首次启动引导（模态），返回 "settings" / "transcribe" / ""。"""
    return WelcomeDialog(parent, theme).show()


__all__ = ["SettingsDialog", "WelcomeDialog", "HelpDialog", "show_welcome"]
