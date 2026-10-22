"""GUI 端到端测试。

覆盖：
1. 文件转写页：真实 tiny 转写 + mock DeepSeek + 四格式导出；
2. 语音输入页：用假录音器喂入真实音频 -> 真实 tiny 转写 + mock DeepSeek 校对 -> 结果进文本框并自动复制；
3. 设置对话框：已保存 API Key 时留空保存不得清空 Key。

需要 ``VOXSCRIBE_RUN_ASR=1``、Windows 桌面会话、本机已缓存 tiny 模型。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from tests.mock_api import MockAPIServer
from voxscribe.config import AppSettings

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "tts_zh.wav"
RUN_ASR = os.environ.get("VOXSCRIBE_RUN_ASR") == "1"

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="GUI 测试需要 Windows 桌面"),
]

ASR_GATE = [
    pytest.mark.skipif(
        not RUN_ASR, reason="需要 VOXSCRIBE_RUN_ASR=1 且本机可用 tiny 模型"
    ),
    pytest.mark.skipif(not FIXTURE.exists(), reason="缺少中文 TTS 测试音频"),
]


class FakeRecorder:
    """替代真实麦克风的录音器：stop() 直接返回测试音频。"""

    def __init__(self, recordings_dir=None, samplerate: int = 16000) -> None:
        self.recordings_dir = recordings_dir
        self._recording = False
        self._started_at = 0.0
        self._levels = [0.02, 0.18, 0.42, 0.75, 0.51, 0.23, 0.08, 0.31, 0.66, 0.44]

    def available(self) -> bool:
        return True

    def list_input_devices(self):
        return [(0, "Fake Microphone")]

    def start(self, device=None) -> None:
        self._recording = True
        self._started_at = time.monotonic()

    def stop(self) -> Path:
        self._recording = False
        return FIXTURE

    def cancel(self) -> None:
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def level(self) -> float:
        return self._levels[-1] if self._recording else 0.0

    def recent_levels(self, count: int = 90):
        return list(self._levels[-max(1, int(count)) :]) if self._recording else []


def _env_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXSCRIBE_CONFIG", str(tmp_path / "data" / "config.json"))
    monkeypatch.setenv("VOXSCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOXSCRIBE_GUI_AUTOCLOSE_MS", "")


def _patch_dialogs(monkeypatch: pytest.MonkeyPatch, gui_app) -> list:
    dialogs = []
    for name in (
        "showinfo",
        "showerror",
        "showwarning",
        "askyesno",
        "askokcancel",
        "askquestion",
    ):

        def recorder(*args, _name=name, **kwargs):
            dialogs.append((_name, args[1] if len(args) > 1 else ""))
            return True

        monkeypatch.setattr(gui_app.messagebox, name, recorder)
    return dialogs


def test_settings_save_keeps_existing_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """打开设置后不输入新 Key 直接保存，不能清空已保存的 Key。"""
    _env_isolation(tmp_path, monkeypatch)
    from voxscribe.gui import app as gui_app
    from voxscribe.gui.dialogs import SettingsDialog

    settings = AppSettings()
    settings.set_api_key("sk-keep-me-please")
    root = gui_app._create_root()
    root.withdraw()
    theme = gui_app.theme_module.apply_theme(root)
    pipeline = gui_app.Pipeline(settings)
    dialog = SettingsDialog(
        root, settings=settings, pipeline=pipeline, theme=theme, on_saved=None
    )
    try:
        assert dialog._has_saved_key is True
        dialog.api_key_var.set("")  # 模拟“留空不改”
        dialog._on_save()
        assert settings.has_api_key is True
        assert settings.get_api_key() == "sk-keep-me-please"
    finally:
        try:
            dialog.destroy()
        except Exception:
            pass
        root.destroy()


@pytest.mark.skipif(
    not RUN_ASR, reason="需要 VOXSCRIBE_RUN_ASR=1 且本机可使用 tiny 模型"
)
@pytest.mark.skipif(not FIXTURE.exists(), reason="缺少中文 TTS 测试音频")
def test_file_page_batch_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env_isolation(tmp_path, monkeypatch)
    from voxscribe.gui import app as gui_app

    dialogs = _patch_dialogs(monkeypatch, gui_app)

    with MockAPIServer() as server:
        settings = AppSettings(
            asr_model="tiny",
            cpu_threads=4,
            language="zh",
            model_dir=str(ROOT / "data" / "models"),
            api_base_url=server.url,
            api_model="deepseek-chat",
            export_formats=["txt", "md", "docx", "srt"],
            output_dir=str(tmp_path / "out"),
            refine_by_default=True,
            default_style="oral",
            first_run_done=True,
            idle_unload_minutes=0,
            window_geometry="",
        )
        settings.set_api_key("sk-gui-e2e")

        root = gui_app._create_root()
        root.withdraw()
        window = None
        try:
            window = gui_app.MainWindow(root, settings)
            assert window.refine_var.get() is True

            window._add_paths([str(FIXTURE)])
            assert len(window.files) == 1
            assert window.files[0].status == "待处理"

            window._start_processing()
            deadline = time.time() + 300
            while window._processing and time.time() < deadline:
                root.update()
                time.sleep(0.02)
            for _ in range(30):  # 把剩余 UI 队列消息消化完
                root.update()
                time.sleep(0.01)

            assert not window._processing, f"处理超时，弹窗：{dialogs}"
            assert dialogs == [], f"正常流程出现了弹窗：{dialogs}"
            assert window.files[0].status == "已完成"
            assert len(window.results) == 1
            item, result = window.results[0]
            assert result.refined is not None and result.refined.text.strip()
            assert result.transcript.duration > 5
            assert server.request_count == 1, "文件转写没有调用 mock DeepSeek"

            refined_preview = window.refined_view.get("1.0", "end").strip()
            raw_preview = window.raw_view.get("1.0", "end").strip()
            assert any(key in refined_preview for key in ("项目", "进度", "前端", "后端"))
            assert "[" in raw_preview and "]" in raw_preview

            outputs = sorted((tmp_path / "out").glob("*"))
            suffixes = {path.suffix for path in outputs}
            assert {".txt", ".md", ".docx", ".srt"} <= suffixes
            assert any("原始转写" in path.name for path in outputs)
            assert all(path.stat().st_size > 0 for path in outputs)

            assert window.progress_var.get() == pytest.approx(100.0)
            assert "内存" in window.memory_var.get()
            assert "MB" in window.memory_var.get()
            assert "最近完成" in window.result_info_var.get()
        finally:
            if window is not None:
                window._on_close()
            else:
                root.destroy()


@pytest.mark.skipif(
    not RUN_ASR, reason="需要 VOXSCRIBE_RUN_ASR=1 且本机可使用 tiny 模型"
)
@pytest.mark.skipif(not FIXTURE.exists(), reason="缺少中文 TTS 测试音频")
def test_home_page_quick_dictation_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """语音输入页：F2 开始/停止 -> 真实转写 + mock DeepSeek 校对 -> 自动复制。"""
    _env_isolation(tmp_path, monkeypatch)
    from voxscribe.gui import app as gui_app

    monkeypatch.setattr(gui_app, "Recorder", FakeRecorder)
    dialogs = _patch_dialogs(monkeypatch, gui_app)

    with MockAPIServer() as server:
        settings = AppSettings(
            asr_model="tiny",
            cpu_threads=4,
            language="zh",
            model_dir=str(ROOT / "data" / "models"),
            api_base_url=server.url,
            api_model="deepseek-chat",
            auto_copy=True,
            first_run_done=True,
            idle_unload_minutes=0,
            window_geometry="",
        )
        settings.set_api_key("sk-gui-quick")

        root = gui_app._create_root()
        root.withdraw()
        window = None
        try:
            window = gui_app.MainWindow(root, settings)
            assert window._recorder is not None
            assert window._quick_state == "idle"

            # F2 第一次：开始录音
            window._toggle_quick()
            assert window._quick_state == "recording"
            root.update()
            time.sleep(0.15)
            root.update()

            # F2 第二次：停止并转换
            window._toggle_quick()
            assert window._quick_state == "converting"

            deadline = time.time() + 300
            while time.time() < deadline:
                root.update()
                time.sleep(0.02)
                if window._quick_state == "idle" and window._get_home_text().strip():
                    break
            root.update()

            assert not dialogs, f"正常流程出现了弹窗：{dialogs}"
            assert window._quick_state == "idle"
            text = window._get_home_text()
            assert any(key in text for key in ("项目", "进度", "前端", "后端")), text
            assert server.request_count == 1, "语音输入没有调用 mock DeepSeek 整理"
            assert "已整理" in window.home_status_var.get()
            assert "已复制" in window.home_status_var.get()

            clipboard = root.clipboard_get()
            assert text in clipboard

            assert window.progress_var.get() == pytest.approx(0.0)
            assert "内存" in window.memory_var.get()
        finally:
            if window is not None:
                window._on_close()
            else:
                root.destroy()


@pytest.mark.skipif(
    not RUN_ASR, reason="需要 VOXSCRIBE_RUN_ASR=1 且本机可使用 tiny 模型"
)
@pytest.mark.skipif(not FIXTURE.exists(), reason="缺少中文 TTS 测试音频")
def test_home_page_quick_dictation_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未配置 Key 时自动降级为仅本地转写，不调用 DeepSeek。"""
    _env_isolation(tmp_path, monkeypatch)
    monkeypatch.delenv("VOXSCRIBE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from voxscribe.gui import app as gui_app

    monkeypatch.setattr(gui_app, "Recorder", FakeRecorder)
    dialogs = _patch_dialogs(monkeypatch, gui_app)

    with MockAPIServer() as server:
        settings = AppSettings(
            asr_model="tiny",
            cpu_threads=4,
            language="zh",
            model_dir=str(ROOT / "data" / "models"),
            api_base_url=server.url,
            auto_copy=False,
            first_run_done=True,
            idle_unload_minutes=0,
            window_geometry="",
        )
        assert not settings.has_api_key

        root = gui_app._create_root()
        root.withdraw()
        window = None
        try:
            window = gui_app.MainWindow(root, settings)
            window._toggle_quick()
            assert window._quick_state == "recording"
            window._toggle_quick()
            assert window._quick_state == "converting"

            deadline = time.time() + 300
            while time.time() < deadline:
                root.update()
                time.sleep(0.02)
                if window._quick_state == "idle" and window._get_home_text().strip():
                    break
            root.update()

            assert not dialogs, f"正常流程出现了弹窗：{dialogs}"
            text = window._get_home_text()
            assert any(key in text for key in ("项目", "进度", "前端", "后端")), text
            assert server.request_count == 0, "无 Key 时不应调用 DeepSeek"
            status = window.home_status_var.get()
            assert "转写" in status
            assert "仅本地转写" in status or "未配置" in status
        finally:
            if window is not None:
                window._on_close()
            else:
                root.destroy()
