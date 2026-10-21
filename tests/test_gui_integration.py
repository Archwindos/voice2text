"""GUI 端到端测试：真实 tiny 转写 + mock DeepSeek 整理 + 四格式导出。

直接实例化 ``MainWindow`` 并驱动文件队列/后台线程，绕过人工点击；
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
    pytest.mark.skipif(not RUN_ASR, reason="需要 VOXSCRIBE_RUN_ASR=1 且本机可用 tiny 模型"),
    pytest.mark.skipif(not FIXTURE.exists(), reason="缺少中文 TTS 测试音频"),
]


def test_gui_full_flow_with_mock_deepseek(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXSCRIBE_CONFIG", str(tmp_path / "data" / "config.json"))
    monkeypatch.setenv("VOXSCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VOXSCRIBE_GUI_AUTOCLOSE_MS", "")  # 不使用自动关闭

    from voxscribe.gui import app as gui_app

    # 弹窗不应在正常流程中出现；一旦出现即记录，最后断言为空
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
            assert server.request_count == 1, "GUI 的 AI 整理没有调用 mock DeepSeek"

            # 预览区展示整理稿与原始转写
            refined_preview = window.refined_view.get("1.0", "end").strip()
            raw_preview = window.raw_view.get("1.0", "end").strip()
            assert any(key in refined_preview for key in ("项目", "进度", "前端", "后端"))
            assert "[" in raw_preview and "]" in raw_preview

            # 导出文件
            outputs = sorted((tmp_path / "out").glob("*"))
            suffixes = {path.suffix for path in outputs}
            assert {".txt", ".md", ".docx", ".srt"} <= suffixes
            assert any("原始转写" in path.name for path in outputs)
            assert all(path.stat().st_size > 0 for path in outputs)

            # 状态与资源显示（启动提示定时器可能覆盖状态栏文案，故只校验关键状态）
            assert window.progress_var.get() == pytest.approx(100.0)
            assert "内存" in window.memory_var.get()
            assert "MB" in window.memory_var.get()
            assert "最近完成" in window.result_info_var.get()
        finally:
            if window is not None:
                window._on_close()
            else:
                root.destroy()
