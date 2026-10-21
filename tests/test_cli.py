"""命令行入口测试（子进程方式，验证真实可运行性）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "voxscribe", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def test_check_command() -> None:
    result = _run(["--check"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "自检通过" in result.stdout


def test_help_command() -> None:
    result = _run(["--help"])
    assert result.returncode == 0
    assert "語声成文" in result.stdout or "语音" in result.stdout


def test_missing_inputs_exit_code() -> None:
    result = _run(["-i", "不存在的音频.mp3"])
    assert result.returncode == 2
    assert "没有找到可处理的音频文件" in (result.stdout + result.stderr)
