"""pytest 公共 fixture。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from voxscribe.config import AppSettings
from voxscribe.models import JobOptions, Segment, Transcript


@pytest.fixture(scope="session", autouse=True)
def _hide_gui_windows():
    """测试期间不弹出任何 GUI 窗口、不创建托盘图标（不在用户桌面上打扰）。"""
    os.environ["VOXSCRIBE_GUI_HIDDEN"] = "1"
    os.environ["VOXSCRIBE_DISABLE_TRAY"] = "1"
    yield


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(0.0, 2.5, "嗯，那个，我们今天呢主要讨论一下项目的进度问题。"),
            Segment(2.5, 6.0, "就是那个，我觉得前端大概完成了百分之七十左右吧。"),
            Segment(6.0, 9.5, "然后后端的话可能还需要两周时间，就是这样。"),
        ],
        language="zh",
        duration=9.5,
    )


@pytest.fixture
def settings(tmp_path) -> AppSettings:
    data = tmp_path / "data"
    data.mkdir()
    return AppSettings(output_dir=str(tmp_path / "outputs"))


@pytest.fixture
def job_options(tmp_path) -> JobOptions:
    return JobOptions(
        refine=False,
        style="oral",
        language="zh",
        export_formats=["txt", "md"],
        output_dir=str(tmp_path / "out"),
    )
