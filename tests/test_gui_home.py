"""首页「语音输入」GUI 测试（不需要真实麦克风 / 真实 DeepSeek API）。

通过 monkeypatch ``Pipeline.transcribe_and_refine`` 与假录音器驱动完整流程：
F2 快捷键、录音 → 转换 → 结果文本、自动复制、无 Key 降级、设置保存不清 Key。
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
    """隔离配置/数据目录，并确认有可用显示环境。"""
    try:
        import tkinter as tk

        probe = tk.Tk()
        probe.withdraw()
        probe.destroy()
    except Exception as exc:  # pragma: no cover - 无桌面环境
        pytest.skip(f"没有可用显示环境：{exc}")

    monkeypatch.setenv("VOXSCRIBE_CONFIG", str(tmp_path / "data" / "config.json"))
    monkeypatch.setenv("VOXSCRIBE_DATA_DIR", str(tmp_path / "data"))
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


def _patch_pipeline(window, monkeypatch, *, refined: bool = True):
    """替换 Pipeline.transcribe_and_refine，记录传入的 JobOptions。"""
    from voxscribe.models import JobEvent

    captured: dict = {}

    def fake(audio_path, job, emit, cancel=None):  # noqa: ANN001
        captured["job"] = job
        captured["audio"] = audio_path
        emit(JobEvent(stage="asr", progress=1.0, message="转写完成"))
        return _fake_job_result(refined)

    monkeypatch.setattr(window.pipeline, "transcribe_and_refine", fake)
    return captured


def _run_quick_flow(window, timeout: float = 15.0) -> None:
    root = window.root
    window._quick_start()
    assert window._quick_state == "recording"
    window._quick_stop_and_convert()
    assert window._quick_state == "converting"
    deadline = time.time() + timeout
    while window._quick_state != "idle" and time.time() < deadline:
        root.update()
        time.sleep(0.02)
    for _ in range(30):  # 消化剩余 UI 队列消息
        root.update()
        time.sleep(0.01)
    assert window._quick_state == "idle", "语音输入转换超时"


def _press_key(window, keysym: str) -> None:
    """模拟按键（先让窗口获得焦点，否则多窗口测试中生成的键盘事件会被丢弃）。"""
    root = window.root
    try:
        root.focus_force()
    except Exception:  # pragma: no cover - 极端环境
        pass
    root.update()
    root.event_generate(keysym)
    root.update()


def _close_window(window) -> None:
    try:
        window._on_close()
    except Exception:  # pragma: no cover - 清理失败不影响断言结果
        pass


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
def test_f2_binding_and_successful_conversion(gui_env, monkeypatch):
    root, window = _make_window(gui_env, api_key="sk-test-home")
    _patch_pipeline(window, monkeypatch, refined=True)
    try:
        assert window.root.bind("<F2>"), "F2 快捷键应绑定到主窗口"

        # F2 开始录音
        _press_key(window, "<F2>")
        assert window._quick_state == "recording"
        assert "停止" in window.home_record_btn.cget("text")
        assert window._recorder.started == 1

        # 再按 F2 结束并开始转换（fake 流程很快，状态可能瞬间进入 idle）
        _press_key(window, "<F2>")
        assert window._quick_state in ("converting", "idle")
        deadline = time.time() + 15
        while window._quick_state != "idle" and time.time() < deadline:
            window.root.update()
            time.sleep(0.02)
        for _ in range(30):
            window.root.update()
            time.sleep(0.01)

        assert window._quick_state == "idle"
        assert window._recorder.stopped == 1
        assert FAKE_REFINED in window._get_home_text()
        status = window.home_status_var.get()
        assert "AI 已整理" in status
        assert "仅本地转写" not in status
        # 成功后不应出现「未配置 Key」橙色提示
        assert "未配置" not in window.home_inline_hint.cget("text")
        # 自动复制到剪贴板
        assert window.settings.auto_copy is True
        assert window.root.clipboard_get() == window._get_home_text()
        assert gui_env["dialogs"] == []
    finally:
        _close_window(window)


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
        assert window.auto_copy_var.get() is False

        _run_quick_flow(window)

        assert FAKE_REFINED in window._get_home_text()
        assert "已复制" not in window.home_status_var.get()
        assert window.root.clipboard_get() == "KEEP-ME"
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
        status = window.home_status_var.get()
        assert "仅本地转写" in status
        assert "已整理" not in status
        assert "未配置" in window.home_inline_hint.cget("text")
        assert "未配置" in window.home_mode_var.get()
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
        assert window.root.bind("<Escape>"), "Esc 快捷键应绑定到主窗口"
        window._quick_start()
        # 隐藏窗口下合成键盘事件的焦点行为不稳定，直接触发绑定后的处理函数
        window._on_escape()
        assert window._quick_state == "idle"
        assert window._recorder.cancelled == 1
    finally:
        _close_window(window)


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
    assert settings.has_api_key is True

    root = gui_app._create_root()
    window = gui_app.MainWindow(root, settings)
    try:
        dialog = SettingsDialog(
            root,
            settings=settings,
            pipeline=window.pipeline,
            theme=window.theme,
        )
        # 不回显明文 Key；留空表示不修改
        assert dialog.api_key_var.get() == ""
        assert dialog._has_saved_key is True
        assert dialog.clear_key_button["state"] != "disabled"

        dialog._on_save()
        root.update()

        assert settings.has_api_key is True, "留空保存不得清空已保存的 Key"
        assert settings.get_api_key() == "sk-keep-me-123"
        config_path = Path(gui_env["tmp_path"]) / "data" / "config.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        assert raw["api_key_enc"].startswith("dpapi:")
        assert "sk-keep-me-123" not in json.dumps(raw, ensure_ascii=False)
    finally:
        _close_window(window)


def test_settings_dialog_accepts_new_key(gui_env):
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
        dialog._on_save()
        root.update()
        assert settings.get_api_key() == "sk-new-key-456"
        raw = json.loads(
            (Path(gui_env["tmp_path"]) / "data" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        assert raw["api_key_enc"].startswith("dpapi:")
        assert "sk-new-key-456" not in json.dumps(raw, ensure_ascii=False)
    finally:
        _close_window(window)
