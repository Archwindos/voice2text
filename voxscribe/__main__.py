"""程序入口。

用法：
    python -m voxscribe                        # 打开图形界面
    python -m voxscribe --check                # 自检运行环境
    python -m voxscribe -i 录音.m4a 录音2.mp3   # 命令行批量处理
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import (
    APP_NAME,
    ASR_MODELS,
    LANGUAGES,
    AppSettings,
    config_path,
    data_dir,
    logs_dir,
    models_dir,
)
from .models import STYLES, JobEvent, JobOptions
from .pipeline import Pipeline


def _force_utf8_console() -> None:
    """避免 Windows 控制台 GBK 编码导致中文输出报错。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _attach_glossary_log_handler(logs: Path) -> None:
    """词语学习单独写 JSONL 日志（产品试验/后续优化用，不混在 app.log）。"""
    logger = logging.getLogger("voxscribe.glossary")
    target = (logs / "glossary_learn.jsonl").resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            try:
                if Path(handler.baseFilename).resolve() == target:
                    return
            except Exception:  # pragma: no cover
                continue
    handler = logging.handlers.RotatingFileHandler(
        target, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # 单独落盘，避免和 app.log 重复；需要看整体流程日志时仍可查 app.log
    logger.propagate = False


def setup_logging(verbose: bool = False) -> Path:
    logs = logs_dir()
    log_file = logs / "app.log"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not root.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1_500_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root.addHandler(file_handler)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG if verbose else logging.WARNING)
        root.addHandler(console)
    _attach_glossary_log_handler(logs)
    return log_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voxscribe",
        description=f"{APP_NAME} 语声成文 v{__version__}：本地语音转写 + DeepSeek 文字整理",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    parser.add_argument("-i", "--input", nargs="+", metavar="音频", help="要处理的音频文件（可多个，支持通配符）。不填则打开图形界面")
    parser.add_argument("-o", "--out", metavar="目录", help="输出目录（默认 outputs/）")
    parser.add_argument("--style", choices=list(STYLES), help="整理风格")
    parser.add_argument("--language", choices=list(LANGUAGES), help="音频语言")
    parser.add_argument("--model", choices=list(ASR_MODELS), help="识别模型大小")
    parser.add_argument("--formats", metavar="txt,md,docx,srt", help="导出格式，逗号分隔")
    parser.add_argument("--no-refine", action="store_true", help="只转写，不调用 DeepSeek 整理")
    parser.add_argument("--print", dest="print_text", action="store_true", help="在终端打印整理后的正文")
    parser.add_argument("--check", action="store_true", help="检查运行环境后退出")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser


def _expand_inputs(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if any(char in pattern for char in "*?["):
            matches = sorted(Path(".").glob(pattern))
            files.extend(matches)
        else:
            files.append(candidate)
    return [path for path in files if path.exists()]


def _parse_formats(raw: Optional[str], fallback: List[str]) -> List[str]:
    if not raw:
        return fallback
    allowed = {"txt", "md", "docx", "srt"}
    formats = [item.strip().lower() for item in raw.split(",") if item.strip()]
    formats = [item for item in formats if item in allowed]
    return formats or fallback


def _print_event(event: JobEvent) -> None:
    if event.stage in ("prepare", "asr", "refine", "export"):
        progress = ""
        if event.progress >= 0:
            progress = f" {event.progress * 100:5.1f}%"
        print(f"  [{event.stage}]{progress} {event.message}", file=sys.stderr, flush=True)
    elif event.stage == "done":
        print(f"  [完成] {event.message}", file=sys.stderr, flush=True)


def run_headless(args: argparse.Namespace, settings: AppSettings) -> int:
    inputs = _expand_inputs(args.input)
    if not inputs:
        print("错误：没有找到可处理的音频文件。", file=sys.stderr)
        return 2

    if args.model:
        settings.asr_model = args.model
    if args.language:
        settings.language = args.language
    style = args.style or settings.default_style
    formats = _parse_formats(args.formats, list(settings.export_formats))
    job = JobOptions(
        refine=not args.no_refine,
        style=style,
        language=args.language or settings.language,
        export_formats=formats,
        output_dir=args.out,
    )

    pipeline = Pipeline(settings)
    cancel = threading.Event()
    exit_code = 0
    for index, audio in enumerate(inputs, start=1):
        print(f"\n[{index}/{len(inputs)}] {audio.name}", file=sys.stderr, flush=True)
        try:
            result = pipeline.run(audio, job, _print_event, cancel)
        except KeyboardInterrupt:
            cancel.set()
            print("\n已手动中断。", file=sys.stderr)
            return 130
        except Exception as exc:  # 单个文件失败不影响后续文件
            exit_code = 1
            hint = getattr(exc, "user_hint", "") or ""
            print(f"  处理失败：{exc}", file=sys.stderr)
            if hint:
                print(f"  提示：{hint}", file=sys.stderr)
            logging.getLogger(__name__).exception("处理失败：%s", audio)
            continue
        print("  输出文件：", file=sys.stderr)
        for output in result.output_files:
            print(f"    - {output}", file=sys.stderr)
        if args.print_text:
            print("\n" + "=" * 60)
            print(result.final_text)
            print("=" * 60)
    pipeline.unload()
    return exit_code


def run_check() -> int:
    print(f"{APP_NAME} 环境自检")
    print(f"  程序版本   : {__version__}")
    print(f"  Python     : {sys.version.split()[0]}  ({sys.executable})")
    print(f"  CPU 逻辑核 : {os.cpu_count()}")
    print(f"  数据目录   : {data_dir()}")
    print(f"  配置文件   : {config_path()}")
    print(f"  模型目录   : {models_dir()}")
    print(f"  日志目录   : {logs_dir()}")

    checks = [
        ("faster-whisper", "faster_whisper"),
        ("requests", "requests"),
        ("python-docx", "docx"),
        ("sounddevice", "sounddevice"),
        ("tkinter", "tkinter"),
        ("tkinterdnd2", "tkinterdnd2"),
    ]
    failed = 0
    for label, module in checks:
        try:
            __import__(module)
            print(f"  [OK] {label} 已安装")
        except Exception as exc:
            failed += 1
            print(f"  [缺失] {label}: {exc}")
    settings = AppSettings.load()
    key = settings.get_api_key()
    print(f"  DeepSeek API Key : {'已配置' if key else '未配置（仅可转写，无法 AI 整理）'}")
    print(f"  DeepSeek 模型    : {settings.api_model} @ {settings.api_base_url}")
    print(
        f"  识别设备/精度    : {settings.effective_device()} / "
        f"{settings.effective_compute_type()}"
    )
    try:
        import ctranslate2  # noqa: PLC0415

        cuda_count = int(ctranslate2.get_cuda_device_count())
        print(f"  CUDA 设备数量    : {cuda_count}")
        if settings.effective_device() == "cuda" and cuda_count == 0:
            print(
                "  [警告] 配置为 GPU 转写，但未检测到可用 CUDA 设备；"
                "可在 data/config.json 把 asr_device 改为 cpu。"
            )
    except Exception as exc:  # pragma: no cover - 依赖缺失/无 GPU
        print(f"  CUDA 检测失败    : {exc}")

    model_dir = settings.resolve_model_dir()
    cached = []
    if model_dir.exists():
        for item in sorted(model_dir.glob("models--*")):
            name = item.name.replace("models--Systran--faster-whisper-", "")
            if item.is_dir() and "-" != name:
                cached.append(name)
    print(f"  已缓存识别模型   : {', '.join(cached) if cached else '无（首次转写时自动下载）'}")
    print()
    if failed:
        print(f"自检发现 {failed} 个依赖缺失，请运行 setup.bat 安装依赖。")
        return 1
    print("自检通过。运行 run.vbs（或 .venv\\Scripts\\pythonw.exe -m voxscribe）打开界面。")
    return 0


def _hide_console_window() -> None:
    """GUI 模式下隐藏控制台窗口（即使通过 python.exe/run.bat 启动也不显示黑框）。

    设置环境变量 ``VOXSCRIBE_KEEP_CONSOLE=1`` 可保留控制台（调试用）。
    """
    if sys.platform != "win32":
        return
    if os.environ.get("VOXSCRIBE_KEEP_CONSOLE", "").strip() == "1":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:  # pragma: no cover - 无控制台/极端环境
        pass


def launch_gui() -> int:
    _hide_console_window()
    try:
        from .gui.app import launch
    except Exception as exc:
        print(f"无法加载图形界面：{exc}", file=sys.stderr)
        print("可尝试运行 setup.bat 重新安装依赖，或用 -i 参数走命令行模式。", file=sys.stderr)
        return 1
    return int(launch() or 0)


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.check:
        return run_check()
    if args.input:
        settings = AppSettings.load()
        return run_headless(args, settings)
    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
