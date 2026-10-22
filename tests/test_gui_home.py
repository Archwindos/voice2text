"""首页「语音输入」/ 托盘 / 全局热键 GUI 测试（全部隐藏窗口运行）。

不需要真实麦克风、真实 DeepSeek API、真实系统托盘：
* ``tests/conftest.py`` 已全局设置 ``VOXSCRIBE_GUI_HIDDEN=1`` 与 ``VOXSCRIBE_DISABLE_TRAY=1``；
* 录音器替换为假实现，``Pipeline.transcribe_and_refine`` 用 monkeypatch 模拟；
* 托盘菜单逻辑用假后端（FakeIcon/FakeMenu）验证，不创建真实图标。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FAKE_REFINED = "我们今天主要讨论项目的进度问题，前端已经完成百分之七十。"
FAKE_RAW = "嗯，那个，我们今天主要讨论项目的进度问题。"


# ---------------------------------------------------------------------------
# 环境与辅助
# ---------------------------------------------------------------------------
@pytest.fixture()
def gui_env(tmp_path, monkeypatch):
    """隔离配置/数据目录，并确认有可用显示环境（隐藏窗口）。"""
    try:
        import tkinter as tk

        probe = tk.Tk()
        probe.withdraw()
        probe.destroy()
    except Exception as exc:  # pragma: no cover - 无桌面环境
        pytest.skip(f"没有可用显示环境：{exc}")

    monkeypatch.setenv("VOXSCRIBE_CONFIG", str(tmp_path / "data" / "config.json"))
    monkeypatch.setenv("VOXSCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOXSCRIBE_GUI_HIDDEN", "1")
    monkeypatch.setenv("VOXSCRIBE_DISABLE_TRAY", "1")
    monkeypatch.setenv("VOXSCRIBE_GUI_AUTOCLOSE_MS", "")
    monkeypatch.delenv("VOXSCRIBE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    from voxscribe.gui import app as gui_app

    dialogs: list = []

    def _collector(name: str):
        def fake(*args, **kwargs):
            dialogs.append((name, args))
            return True

        return fake

    for name in (
        "showinfo",
        "showerror",
        "showwarning",
        "askyesno",
        "askokcancel",
        "askquestion",
    ):
        monkeypatch.setattr(gui_app.messagebox, name, _collector(name))

    return {"gui_app": gui_app, "dialogs": dialogs, "tmp_path": tmp_path}


def _fake_job_result(refined: bool = True):
    from voxscribe.models import JobResult, RefineResult, Segment, Transcript

    transcript = Transcript(
        segments=[Segment(0.0, 2.0, FAKE_RAW)],
        language="zh",
        duration=2.0,
    )
    refined_result = RefineResult(text=FAKE_REFINED) if refined else None
    return JobResult(
        audio_path="fake_recording.wav",
        transcript=transcript,
        refined=refined_result,
        elapsed=1.2,
    )


class FakeRecorder:
    """不依赖真实麦克风的录音器替身。"""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.started = 0
        self.stopped = 0
        self.cancelled = 0
        self._started_at = 0.0

    def start(self, device=None) -> None:  # noqa: ANN001
        self.started += 1
        self._started_at = time.monotonic()

    def stop(self) -> Path:
        self.stopped += 1
        path = self._tmp_path / "fake_recording.wav"
        path.write_bytes(b"RIFF-fake-wav")
        return path

    def cancel(self) -> None:
        self.cancelled += 1

    @property
    def is_recording(self) -> bool:
        return self.started > self.stopped

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def level(self) -> float:
        return 0.4

    def recent_levels(self, count: int = 90):
        return [0.2, 0.5, 0.8, 0.3, 0.0] if self.is_recording else []


class FakeTray:
    """主窗口使用的假托盘对象（不创建真实图标）。"""

    def __init__(self) -> None:
        self.notifications: list = []
        self.stopped = False
        self.updates = 0

    def update_state(self) -> None:
        self.updates += 1

    def notify(self, title: str, message: str) -> bool:
        self.notifications.append((title, message))
        return True

    def stop(self) -> None:
        self.stopped = True


def _make_window(env, *, api_key: str | None = None, auto_copy: bool = True):
    gui_app = env["gui_app"]
    from voxscribe.config import AppSettings

    tmp_path = env["tmp_path"]
    settings = AppSettings(
        asr_model="tiny",
        cpu_threads=1,
        language="zh",
        model_dir=str(tmp_path / "models"),
        export_formats=["txt"],
        output_dir=str(tmp_path / "out"),
        refine_by_default=True,
        default_style="oral",
        auto_copy=auto_copy,
        first_run_done=True,
        idle_unload_minutes=0,
        window_geometry="",
    )
    if api_key is not None:
        settings.set_api_key(api_key)
    root = gui_app._create_root()
    window = gui_app.MainWindow(root, settings)
    window._recorder = FakeRecorder(tmp_path)
    window._recorder_note = ""
    window._refresh_action_states()
    return root, window


def _patch_pipeline(
    window,
    monkeypatch,
    *,
    refined: bool = True,
    interim_text: str | None = None,
    refine_delay: float = 0.0,
):
    """替换 Pipeline.transcribe_and_refine，可模拟「先出原始转写、再润色」。"""
    from voxscribe.models import JobEvent, Segment, Transcript

    captured: dict = {}

    def fake(audio_path, job, emit, cancel=None, on_transcript=None):  # noqa: ANN001
        captured["job"] = job
        captured["audio"] = audio_path
        captured["interim"] = on_transcript is not None
        if on_transcript is not None and interim_text is not None:
            transcript = Transcript(
                segments=[Segment(0.0, 1.0, interim_text)],
                language="zh",
                duration=1.0,
            )
            on_transcript(transcript)
            if refine_delay:
                time.sleep(refine_delay)
            emit(JobEvent(stage="refine", progress=0.5, message="DeepSeek 润色中…"))
        else:
            emit(JobEvent(stage="asr", progress=1.0, message="转写完成"))
        return _fake_job_result(refined)

    monkeypatch.setattr(window.pipeline, "transcribe_and_refine", fake)
    return captured


def _wait_idle(window, timeout: float = 15.0) -> None:
    root = window.root
    deadline = time.time() + timeout
    while window._quick_state != "idle" and time.time() < deadline:
        root.update()
        time.sleep(0.02)
    for _ in range(30):  # 消化剩余 UI 队列消息
        root.update()
        time.sleep(0.01)
    assert window._quick_state == "idle", "语音输入转换超时"


def _run_quick_flow(window, timeout: float = 15.0) -> None:
    window._quick_start()
    assert window._quick_state == "recording"
    window._quick_stop_and_convert()
    assert window._quick_state == "converting"
    _wait_idle(window, timeout)


def _close_window(window) -> None:
    try:
        window._on_close()
    except Exception:  # pragma: no cover - 清理失败不影响断言结果
        pass


def _root_destroyed(root) -> bool:
    try:
        return not root.winfo_exists()
    except Exception:
        return True


# ---------------------------------------------------------------------------
# 全局快捷键 Alt+~
# ---------------------------------------------------------------------------
def test_alt_tilde_hotkey_edge_detection(gui_env):
    from voxscribe.gui.hotkey import (
        HOTKEY_LABEL,
        VK_MENU,
        VK_OEM_3,
        HotkeyListener,
    )

    assert HOTKEY_LABEL == "Alt+~"
    assert VK_MENU == 0x12
    assert VK_OEM_3 == 0xC0

    root = gui_env["gui_app"]._create_root()
    hits: list = []
    listener = HotkeyListener(root, lambda: hits.append("hot"))
    states = iter([True, True, False, True])
    listener.is_pressed_now = lambda: next(states)  # type: ignore[method-assign]
    try:
        assert listener.poll_once() is True  # 按下沿
        assert listener.poll_once() is False  # 长按不重复
        assert listener.poll_once() is False  # 松开
        assert listener.poll_once() is True  # 再次按下
        assert hits == []

        # _tick 应触发回调（等价 after(50ms) 轮询）：先松开，再按下
        listener.is_pressed_now = lambda: False  # type: ignore[method-assign]
        assert listener.poll_once() is False
        listener.is_pressed_now = lambda: True  # type: ignore[method-assign]
        listener._enabled = True
        listener._tick()
        assert hits == ["hot"]
        listener.stop()
    finally:
        root.destroy()


def test_hotkey_starts_and_stops_recording(gui_env, monkeypatch):
    root, window = _make_window(gui_env, api_key="sk-test-home")
    _patch_pipeline(window, monkeypatch, refined=True)
    try:
        # F2 绑定已移除，改由全局 Alt+~ 轮询
        assert not window.root.bind("<F2>")
        assert window._hotkey is not None
        # 空闲时只保留一句提示（含 Alt+~），不再显示模式小字
        assert "Alt+~" in window.home_status_var.get()
        assert not hasattr(window, "home_mode_var")

        window._on_hotkey()  # 模拟按下 Alt+~
        root.update()
        assert window._quick_state == "recording"
        assert "停止" in window.home_record_btn.cget("text")

        window._on_hotkey()
        _wait_idle(window)
        assert FAKE_REFINED in window._get_home_text()
        assert "AI 已整理" in window.home_result_info_var.get()
        assert gui_env["dialogs"] == []
    finally:
        _close_window(window)


# ---------------------------------------------------------------------------
# 两段式：ASR 完成先显示原始转写，润色后替换
# ---------------------------------------------------------------------------
def test_two_stage_shows_raw_transcript_before_refine(gui_env, monkeypatch):
    root, window = _make_window(gui_env, api_key="sk-test-home")
    raw = "嗯，那个，我们今天讨论一下项目进度。"
    captured = _patch_pipeline(
        window,
        monkeypatch,
        refined=True,
        interim_text=raw,
        refine_delay=0.4,  # 模拟 DeepSeek 润色耗时
    )
    try:
        window._quick_start()
        window._quick_stop_and_convert()
        assert window._quick_state == "converting"

        interim_seen = False
        deadline = time.time() + 15
        while time.time() < deadline:
            root.update()
            time.sleep(0.02)
            if window._quick_state == "converting":
                text = window._get_home_text()
                if text.strip() == raw:
                    interim_seen = True
                    status = window.home_status_var.get()
                    assert "润色" in status or "DeepSeek" in status, status
            if window._quick_state == "idle":
                break
        for _ in range(30):
            root.update()
            time.sleep(0.01)

        assert captured["interim"] is True, "应把 on_transcript 回调传给 pipeline"
        assert interim_seen, "DeepSeek 润色完成前应先显示原始转写"
        assert window._quick_state == "idle"
        assert window._get_home_text() == FAKE_REFINED, "最终应替换为润色后的文字"
        assert window._quick_interim_text == ""
    finally:
        _close_window(window)


# ---------------------------------------------------------------------------
# 转换流程与降级
# ---------------------------------------------------------------------------
def test_conversion_passes_proofread_style(gui_env, monkeypatch):
    root, window = _make_window(gui_env, api_key="sk-test-home")
    captured = _patch_pipeline(window, monkeypatch, refined=True)
    try:
        _run_quick_flow(window)
        assert captured["job"].refine is True
        assert captured["job"].style == "proofread"
    finally:
        _close_window(window)


def test_auto_copy_disabled_keeps_clipboard(gui_env, monkeypatch):
    root, window = _make_window(
        gui_env, api_key="sk-test-home", auto_copy=False
    )
    _patch_pipeline(window, monkeypatch, refined=True)
    try:
        window.root.clipboard_clear()
        window.root.clipboard_append("KEEP-ME")
        window.root.update()
        assert window.settings.auto_copy is False

        _run_quick_flow(window)

        assert FAKE_REFINED in window._get_home_text()
        assert "已复制" not in window.home_result_info_var.get()
        assert window.root.clipboard_get() == "KEEP-ME"
    finally:
        _close_window(window)


def test_home_minimal_layout_and_half_wave(gui_env):
    """v1.3：模式小字移除、波形非对称半高、结果区元信息+按钮收敛。"""
    root, window = _make_window(gui_env)
    try:
        # 模式小字整行移除
        assert not hasattr(window, "home_mode_var")
        assert not hasattr(window, "home_mode_label")
        # 波形高度压缩到约 80~100（96dpi 设计值，随 DPI 缩放）
        wave_height = int(window.wave_canvas.cget("height"))
        assert 60 <= wave_height <= 160, wave_height
        # v1.3.1：波形高度映射保留顶部余量，正常说话不会顶到上沿
        from voxscribe.gui.app import _wave_visual

        assert _wave_visual(1.0) <= 0.8
        assert _wave_visual(0.8) < 0.8
        assert _wave_visual(0.5) < 0.6
        assert _wave_visual(0.0) == 0.0
        # 空闲波形区内不再有小字（只有基线与刻度线）
        window._draw_home_idle()
        types = {window.wave_canvas.type(item) for item in window.wave_canvas.find_all()}
        assert "text" not in types, types
        # 结果区标题行右侧元信息 + 复制/清空按钮（无文字时禁用）
        assert window.home_result_info_var.get() == ""
        assert window.home_copy_btn.cget("state") == "disabled"
        assert window.home_clear_btn.cget("state") == "disabled"
        # 底部状态栏不再显示内存（memory_var 仍保留给集成测试）
        assert "内存" in window.memory_var.get()
    finally:
        _close_window(window)


def test_settings_model_shared_with_file_page(gui_env):
    """设置里的「识别模型」是语音输入/文件转写共用的唯一设置。"""
    root, window = _make_window(gui_env)
    try:
        from voxscribe.config import ASR_MODELS
        from voxscribe.gui.dialogs import SettingsDialog

        dialog = SettingsDialog(
            root,
            settings=window.settings,
            pipeline=window.pipeline,
            theme=window.theme,
        )
        dialog.model_var.set(ASR_MODELS["base"])
        dialog._on_save()
        root.update()
        assert window.settings.asr_model == "base"
        # 文件转写页的模型下拉在设置保存后同步为同一个值
        window._apply_settings_to_ui()
        assert window._current_model_key() == "base"
    finally:
        _close_window(window)


def test_no_api_key_degrades_to_local_transcribe(gui_env, monkeypatch):
    root, window = _make_window(gui_env, api_key=None)
    captured = _patch_pipeline(window, monkeypatch, refined=False)
    try:
        assert window.settings.has_api_key is False
        _run_quick_flow(window)

        assert captured["job"].refine is False
        assert captured["job"].style == "proofread"
        assert window._get_home_text() == FAKE_RAW
        meta = window.home_result_info_var.get()
        assert "仅本地转写" in meta
        assert "已整理" not in meta
        assert "未配置" in window.home_inline_hint.cget("text")
        assert gui_env["dialogs"] == []
    finally:
        _close_window(window)


def test_page_switch_blocked_while_recording(gui_env, monkeypatch):
    root, window = _make_window(gui_env, api_key="sk-test-home")
    _patch_pipeline(window, monkeypatch, refined=True)
    try:
        window._quick_start()
        assert window._quick_state == "recording"
        window._switch_page("files")
        assert window._page == "home", "录音中切换页面应被拦截"
        assert gui_env["dialogs"], "切换被拦截时应给出提示"
        window._cancel_quick_recording()
        assert window._quick_state == "idle"
        assert window._recorder.cancelled == 1
    finally:
        _close_window(window)


def test_escape_cancels_recording(gui_env, monkeypatch):
    root, window = _make_window(gui_env, api_key="sk-test-home")
    _patch_pipeline(window, monkeypatch, refined=True)
    try:
        window._quick_start()
        window._on_escape()
        assert window._quick_state == "idle"
        assert window._recorder.cancelled == 1
    finally:
        _close_window(window)


# ---------------------------------------------------------------------------
# 隐藏模式 / 托盘
# ---------------------------------------------------------------------------
def test_hidden_mode_no_tray_and_close_destroys(gui_env):
    from voxscribe.gui import tray as tray_module

    assert tray_module.tray_disabled_by_env() is True
    assert tray_module.tray_available() is False

    root, window = _make_window(gui_env, api_key=None)
    try:
        assert window._hidden_mode is True
        assert window._tray is None
        assert window._tray_job is None
        # 隐藏模式下关闭请求必须真实销毁，不能拦截（否则测试会挂住）
        window._on_close_request()
        assert _root_destroyed(root)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_tray_menu_callbacks_and_state(gui_env, monkeypatch):
    """用假 pystray 后端验证菜单文案/回调/图标状态刷新，不创建真实图标。"""
    from types import SimpleNamespace

    from voxscribe.gui import tray as tray_module

    events: list = []
    state = {"recording": False, "visible": False}

    class FakeIcon:
        def __init__(self, name, icon=None, title=None, menu=None) -> None:
            self.name = name
            self.icon = icon
            self.title = title
            self.menu = menu
            self.notifications: list = []
            self.stopped = False
            self.updates = 0

        def run(self) -> None:
            return None

        def stop(self) -> None:
            self.stopped = True

        def notify(self, message, title=None) -> None:
            self.notifications.append((title, message))

        def update_menu(self) -> None:
            self.updates += 1

    class FakeMenuItem:
        def __init__(self, text, action, default=False) -> None:
            self.text = text
            self.action = action
            self.default = default

    class FakeMenu:
        SEPARATOR = object()

        def __init__(self, *items) -> None:
            self.items = list(items)

    backend = SimpleNamespace(
        Icon=FakeIcon, Menu=FakeMenu, MenuItem=FakeMenuItem, SEPARATOR=object()
    )
    monkeypatch.setattr(tray_module, "tray_available", lambda: True)
    monkeypatch.setattr(tray_module, "_load_backend", lambda: backend)
    monkeypatch.setattr(tray_module, "create_tray_image", lambda recording=False: object())

    root = gui_env["gui_app"]._create_root()
    icon = tray_module.TrayIcon(
        root,
        on_toggle_window=lambda: events.append("toggle_window"),
        on_toggle_recording=lambda: events.append("toggle_recording"),
        on_quit=lambda: events.append("quit"),
        is_recording=lambda: state["recording"],
        is_visible=lambda: state["visible"],
    )
    try:
        assert icon.start() is True
        fake_icon = icon._icon
        assert fake_icon is not None
        labels = [item.text for item in fake_icon.menu.items if hasattr(item, "text")]
        assert "显示主界面" in labels
        assert "开始录音（Alt+~）" in labels
        assert fake_icon.title and "Alt+~" in fake_icon.title

        # 触发「显示主界面」菜单项：回调经 root.after(0) 投递到主线程
        for item in fake_icon.menu.items:
            if getattr(item, "text", "") == "显示主界面":
                item.action()
                break
        root.update()
        assert events == ["toggle_window"]

        # 录音中：菜单文案、tooltip、图标一起刷新
        state["recording"] = True
        state["visible"] = True
        icon.update_state()
        labels = [item.text for item in fake_icon.menu.items if hasattr(item, "text")]
        assert "隐藏主界面" in labels
        assert "停止录音（Alt+~）" in labels
        assert "正在录音" in fake_icon.title
        assert fake_icon.updates >= 1

        assert icon.notify("测试标题", "测试内容") is True
        assert fake_icon.notifications
        icon.stop()
        assert fake_icon.stopped is True
    finally:
        root.destroy()


def test_close_to_tray_hides_and_shows_first_hint(gui_env):
    root, window = _make_window(gui_env)
    try:
        window._hidden_mode = False  # 模拟正常桌面模式
        fake_tray = FakeTray()
        window._tray = fake_tray
        window.settings.close_to_tray = True
        assert window._hide_to_tray(notify=True) is True
        assert window.root.state() == "withdrawn"
        assert window._tray_hint_shown is True
        assert len(fake_tray.notifications) == 1
        assert "托盘" in fake_tray.notifications[0][0]

        # 再次隐藏不重复弹气泡
        window._hide_to_tray(notify=True)
        assert len(fake_tray.notifications) == 1

        # 关闭按钮在托盘可用时 = 隐藏，而不是销毁
        window._on_close_request()
        assert root.winfo_exists() == 1
        assert fake_tray.stopped is False
    finally:
        _close_window(window)


def test_close_to_tray_disabled_exits(gui_env):
    root, window = _make_window(gui_env)
    try:
        window._hidden_mode = False
        window._tray = FakeTray()
        window.settings.close_to_tray = False
        window._on_close_request()
        assert _root_destroyed(root)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_minimize_hides_to_tray(gui_env):
    root, window = _make_window(gui_env)
    try:
        window._hidden_mode = False
        fake_tray = FakeTray()
        window._tray = fake_tray
        # 用假 state() 模拟最小化（真实 iconify 在测试机上会打扰桌面）
        original_state = root.state
        state_calls = {"count": 0}

        def fake_state(*args):
            state_calls["count"] += 1
            if state_calls["count"] == 1:
                return "iconic"
            return original_state(*args)

        root.state = fake_state  # type: ignore[method-assign]
        window._maybe_hide_iconified()
        assert state_calls["count"] >= 1
        assert original_state() == "withdrawn"
        assert len(fake_tray.notifications) == 1
    finally:
        _close_window(window)


# ---------------------------------------------------------------------------
# 模型预热（v1.2：启动后台 warm_up，首句更快）
# ---------------------------------------------------------------------------
def test_hidden_mode_skips_warmup(gui_env, monkeypatch):
    root, window = _make_window(gui_env)
    called: list = []
    monkeypatch.setattr(
        window.pipeline, "warm_up", lambda cancel=None: called.append(1) or True
    )
    try:
        assert window._hidden_mode is True
        window._start_warmup()  # 隐藏测试模式直接跳过
        assert called == []
        assert window._warming_up is False
    finally:
        _close_window(window)


def test_preload_model_disabled_skips_warmup(gui_env, monkeypatch):
    root, window = _make_window(gui_env)
    called: list = []
    monkeypatch.setattr(
        window.pipeline, "warm_up", lambda cancel=None: called.append(1) or True
    )
    try:
        window._hidden_mode = False
        window.settings.preload_model = False
        window._start_warmup()
        assert called == []
        assert window._warming_up is False
    finally:
        _close_window(window)


def test_warmup_updates_status_without_dialog(gui_env, monkeypatch):
    root, window = _make_window(gui_env)
    monkeypatch.setattr(window.pipeline, "warm_up", lambda cancel=None: True)
    try:
        window._hidden_mode = False  # 模拟正常桌面模式
        window._start_warmup()
        assert window._warming_up is True
        deadline = time.time() + 10
        while window._warming_up and time.time() < deadline:
            root.update()
            time.sleep(0.02)
        for _ in range(20):
            root.update()
            time.sleep(0.01)
        assert window._warming_up is False
        assert "预热" in window.status_var.get()
        assert gui_env["dialogs"] == [], "预热不应弹窗"

        # 预热中录音时，状态提示应提示「首次使用稍等」
        window._warming_up = True
        assert "预热" in window._listening_status()
        window._warming_up = False
        assert "预热" not in window._listening_status()
    finally:
        _close_window(window)


# ---------------------------------------------------------------------------
# 设置对话框：Key 保护 / 关闭到托盘选项
# ---------------------------------------------------------------------------
def test_settings_dialog_keeps_saved_key_on_empty_save(gui_env):
    gui_app = gui_env["gui_app"]
    from voxscribe import secure
    from voxscribe.config import AppSettings
    from voxscribe.gui.dialogs import SettingsDialog

    if not secure.is_available():
        pytest.skip("当前系统不支持 DPAPI，跳过 Key 保存测试")

    tmp_path = gui_env["tmp_path"]
    settings = AppSettings(
        first_run_done=True,
        model_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "out"),
        window_geometry="",
    )
    assert settings.set_api_key("sk-keep-me-123") is True

    root = gui_app._create_root()
    window = gui_app.MainWindow(root, settings)
    try:
        dialog = SettingsDialog(
            root,
            settings=settings,
            pipeline=window.pipeline,
            theme=window.theme,
        )
        assert dialog.api_key_var.get() == ""
        assert dialog._has_saved_key is True
        assert dialog.close_to_tray_var.get() is True

        dialog._on_save()
        root.update()

        assert settings.has_api_key is True, "留空保存不得清空已保存的 Key"
        assert settings.get_api_key() == "sk-keep-me-123"
        raw = json.loads(
            (Path(gui_env["tmp_path"]) / "data" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        assert raw["api_key_enc"].startswith("dpapi:")
        assert "sk-keep-me-123" not in json.dumps(raw, ensure_ascii=False)
        assert raw["close_to_tray"] is True
    finally:
        _close_window(window)


def test_settings_dialog_accepts_new_key_and_tray_option(gui_env):
    gui_app = gui_env["gui_app"]
    from voxscribe import secure
    from voxscribe.config import AppSettings
    from voxscribe.gui.dialogs import SettingsDialog

    if not secure.is_available():
        pytest.skip("当前系统不支持 DPAPI，跳过 Key 保存测试")

    tmp_path = gui_env["tmp_path"]
    settings = AppSettings(
        first_run_done=True,
        model_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "out"),
        window_geometry="",
    )
    settings.set_api_key("sk-old-key")

    root = gui_app._create_root()
    window = gui_app.MainWindow(root, settings)
    try:
        dialog = SettingsDialog(
            root,
            settings=settings,
            pipeline=window.pipeline,
            theme=window.theme,
        )
        dialog.api_key_var.set("sk-new-key-456")
        dialog.close_to_tray_var.set(False)
        dialog._on_save()
        root.update()

        assert settings.get_api_key() == "sk-new-key-456"
        assert settings.close_to_tray is False
        raw = json.loads(
            (Path(gui_env["tmp_path"]) / "data" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        assert raw["api_key_enc"].startswith("dpapi:")
        assert "sk-new-key-456" not in json.dumps(raw, ensure_ascii=False)
        assert raw["close_to_tray"] is False
    finally:
        _close_window(window)


def test_settings_dialog_auto_copy_option(gui_env):
    """v1.3：自动复制开关从首页移到设置对话框。"""
    gui_app = gui_env["gui_app"]
    from voxscribe.config import AppSettings
    from voxscribe.gui.dialogs import SettingsDialog

    tmp_path = gui_env["tmp_path"]
    settings = AppSettings(
        first_run_done=True,
        model_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "out"),
        window_geometry="",
        auto_copy=True,
    )
    root = gui_app._create_root()
    window = gui_app.MainWindow(root, settings)
    try:
        dialog = SettingsDialog(
            root,
            settings=settings,
            pipeline=window.pipeline,
            theme=window.theme,
        )
        assert dialog.auto_copy_var.get() is True
        # 首页不再有自动复制控件
        assert not hasattr(window, "auto_copy_var")
        dialog.auto_copy_var.set(False)
        dialog._on_save()
        root.update()
        assert settings.auto_copy is False
        raw = json.loads(
            (Path(gui_env["tmp_path"]) / "data" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        assert raw["auto_copy"] is False
    finally:
        _close_window(window)
