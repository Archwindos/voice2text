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

from ..config import AppSettings, default_output_dir
from ..errors import VoxError
from ..pipeline import Pipeline
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
        bg, fg = COLORS["primary"], "#ffffff"
        active_bg, active_fg = COLORS["primary_dark"], "#ffffff"
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
        outer.pack(fill="both", expand=True, padx=theme.px(20), pady=(theme.px(16), theme.px(14)))

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
        deepseek_card.pack(fill="x", pady=(0, theme.px(12)))
        deepseek_body.columnconfigure(1, weight=1)

        self.api_key_var = tk.StringVar()
        self.api_show_var = tk.BooleanVar(value=False)
        self.base_url_var = tk.StringVar()
        self.api_model_var = tk.StringVar()

        _card_label(deepseek_body, theme, "API Key").grid(
            row=0, column=0, sticky="w", padx=(0, theme.px(10)), pady=theme.px(4)
        )
        self.api_key_entry = ttk.Entry(
            deepseek_body, textvariable=self.api_key_var, show="●", width=32
        )
        self.api_key_entry.grid(row=0, column=1, sticky="ew", pady=theme.px(4))
        ttk.Checkbutton(
            deepseek_body,
            text="显示",
            variable=self.api_show_var,
            command=self._toggle_key_visible,
            style="Card.TCheckbutton",
        ).grid(row=0, column=2, sticky="w", padx=(theme.px(8), 0), pady=theme.px(4))

        _card_label(deepseek_body, theme, "Base URL").grid(
            row=1, column=0, sticky="w", padx=(0, theme.px(10)), pady=theme.px(4)
        )
        ttk.Entry(deepseek_body, textvariable=self.base_url_var).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=theme.px(4)
        )

        _card_label(deepseek_body, theme, "模型名").grid(
            row=2, column=0, sticky="w", padx=(0, theme.px(10)), pady=theme.px(4)
        )
        ttk.Entry(deepseek_body, textvariable=self.api_model_var).grid(
            row=2, column=1, sticky="ew", pady=theme.px(4)
        )
        _card_label(deepseek_body, theme, "默认 deepseek-chat", muted=True).grid(
            row=2, column=2, sticky="w", padx=(theme.px(8), 0)
        )

        hint = (
            "API Key 会用 Windows DPAPI 加密后保存在本机，只有当前 Windows 用户能解密；"
            "只有 AI 整理需要 Key，本地语音转写完全免费。"
        )
        _card_label(
            deepseek_body, theme, hint, muted=True, wraplength=theme.px(500)
        ).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(theme.px(6), 0)
        )

        link_row = tk.Frame(deepseek_body, bg=COLORS["card"])
        link_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(theme.px(6), 0))
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
        perf_card.pack(fill="x", pady=(0, theme.px(12)))
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
            "线程数 0 = 自动（最多 4 线程，尽量不抢满 CPU）；"
            "空闲卸载分钟数 0 = 不卸载，默认 5 分钟空闲后释放识别模型内存。"
        )
        _card_label(
            perf_body, theme, perf_hint, muted=True, wraplength=theme.px(500)
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(theme.px(4), 0))

        # --- 术语表 ---
        glossary_card, glossary_body = make_card(outer, theme, "术语表（可选）")
        glossary_card.pack(fill="x", pady=(0, theme.px(12)))
        self.glossary_text = tk.Text(
            glossary_body,
            height=4,
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
        self.api_key_var.set(settings.get_api_key())
        self.base_url_var.set(settings.api_base_url or "https://api.deepseek.com")
        self.api_model_var.set(settings.api_model or "deepseek-chat")
        self.threads_var.set(str(int(settings.cpu_threads or 0)))
        self.idle_var.set(str(int(settings.idle_unload_minutes or 0)))
        self.output_var.set(
            settings.output_dir.strip()
            if settings.output_dir.strip()
            else str(default_output_dir())
        )
        if settings.glossary:
            self.glossary_text.delete("1.0", "end")
            self.glossary_text.insert("1.0", settings.glossary)

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
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning(
                "缺少 API Key",
                "请先填写 DeepSeek API Key，再测试连接。\n"
                "Key 可在 platform.deepseek.com 申请（仅 AI 整理需要）。",
                parent=self,
            )
            return
        base_url = self.base_url_var.get().strip() or "https://api.deepseek.com"
        model = self.api_model_var.get().strip() or "deepseek-chat"
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
        model = self.api_model_var.get().strip() or "deepseek-chat"

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

        api_key = self.api_key_var.get().strip()
        encrypted = self.settings.set_api_key(api_key)
        self.settings.api_base_url = base_url
        self.settings.api_model = model
        self.settings.cpu_threads = threads
        self.settings.idle_unload_minutes = idle_minutes
        self.settings.glossary = self.glossary_text.get("1.0", "end-1c").strip()
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

        if api_key and not encrypted:
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
            text="VoxScribe · 把语音内容整理成书面文字",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=theme.font(SMALL_SIZE),
            anchor="w",
        ).pack(fill="x", pady=(theme.px(2), theme.px(14)))

        tk.Frame(outer, bg=COLORS["border"], height=1).pack(fill="x", pady=(0, theme.px(14)))

        bullets = [
            ("本地转写，免费且离线", "语音识别在你自己的电脑上完成，音频不会上传，也不需要任何费用。"),
            ("AI 整理，需要 DeepSeek Key", "在 platform.deepseek.com 申请 API Key 后，可把口语整理成通顺的书面文字（按量付费，费用很低）。"),
            ("导出多种格式", "处理完成后自动导出 txt / md / docx / srt，保存在 outputs/ 目录。"),
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
            "还没有 API Key？现在可以先只做语音转写，稍后在「设置」中随时补充。"
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
            buttons, theme, "先只转写", self._choose_transcribe, width_chars=10
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
_HELP_TEXT = """三步完成转写

1. 添加音频
   点击「添加文件」选择 mp3 / wav / m4a / flac 等音频，也可以直接把文件拖进左侧列表。
   需要现场录制时，点击顶部的「● 录音」，再点一次「■ 停止录音」，
   录音会自动加入待处理列表。

2. 设置处理选项
   选择整理风格（口述成文 / 会议纪要 / 逐字稿润色 / 要点摘要）、音频语言、
   识别模型和要导出的格式，并确认输出目录。
   默认导出到程序目录下的 outputs/ 文件夹。

3. 开始处理
   点击「开始处理」，程序会依次完成：本地语音转写 → DeepSeek 整理 → 导出文件。
   处理过程可以随时点击「取消」；完成后可用「复制整理稿」「打开输出目录」查看结果。
   详细进度、失败原因都会显示在底部状态栏。

轻量说明

· 本地转写免费：语音识别完全在本机 CPU 上完成，音频不会上传到任何服务器。
· AI 整理可选：需要在「设置」中填写 DeepSeek API Key 才会启用；没有 Key 也能正常转写。
· API Key 安全：使用 Windows DPAPI 加密保存在本机，只有当前 Windows 用户能解密。
· 模型空闲自动卸载：默认空闲 5 分钟后自动释放识别模型占用的内存，可在设置中调整。
· CPU 占用可控：默认最多使用 4 个线程，避免占满电脑资源。
· 数据都在本地：配置在 data/config.json，录音在 data/recordings，日志在 data/logs/app.log。
· 首次使用会自动下载识别模型（small 约 484MB），请保持网络畅通；下载一次后会复用本地缓存。

小提示

· 首次转写会比较慢，之后模型已缓存会快很多。
· 识别效果不好时，可以在设置里换成更大的模型（medium / large-v3），但速度会更慢。
· 整理结果不满意时，可以换一种整理风格，或只做「逐字稿润色」。
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
