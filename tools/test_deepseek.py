"""手动验证 DeepSeek 真实 API 的小工具。

**不参与自动化测试**，仅在用户/开发者提供了真实 API Key 时手动运行，
会消耗少量 DeepSeek 额度。

Key 只从环境变量读取，绝不硬编码、不打印：

    set VOXSCRIBE_API_KEY=sk-...
    :: 或者
    set DEEPSEEK_API_KEY=sk-...

用法（项目根目录）::

    :: 只测连接（最省额度）
    .venv\\Scripts\\python.exe tools\\test_deepseek.py --connection-only

    :: 整理一段文字
    .venv\\Scripts\\python.exe tools\\test_deepseek.py --text "嗯，那个，我们明天上午十点开会。"

    :: 从文件读取并测试分块（长文本会被切成多块）
    .venv\\Scripts\\python.exe tools\\test_deepseek.py --text-file data/sample.txt --style meeting

Base URL / 模型可用参数或环境变量覆盖（VOXSCRIBE_API_BASE_URL / VOXSCRIBE_API_MODEL）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voxscribe.errors import VoxError  # noqa: E402
from voxscribe.models import STYLES, RefineOptions, Segment, Transcript  # noqa: E402
from voxscribe.refine import DeepSeekRefiner  # noqa: E402

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
KEY_ENV_NAMES = ("VOXSCRIBE_API_KEY", "DEEPSEEK_API_KEY")


def read_api_key() -> str:
    """只从环境变量读取 API Key；不落盘、不打印。"""
    for name in KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def build_transcript(text: str) -> Transcript:
    """把输入文本按行切成带时间戳的 segment（空行忽略）。"""
    lines = [line.strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return Transcript(segments=[], language="zh", duration=0.0)
    segments = [
        Segment(float(index), float(index + 1), line) for index, line in enumerate(lines)
    ]
    return Transcript(segments=segments, language="zh", duration=float(len(lines)))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="手动验证 VoxScribe 的 DeepSeek 文本整理模块（会消耗少量 API 额度）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VOXSCRIBE_API_BASE_URL") or DEFAULT_BASE_URL,
        help="DeepSeek / 兼容代理的 Base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VOXSCRIBE_API_MODEL") or DEFAULT_MODEL,
        help="模型名称",
    )
    parser.add_argument(
        "--style",
        default="oral",
        choices=sorted(STYLES),
        help="整理风格",
    )
    parser.add_argument("--language", default="zh", help="输出语言代码，如 zh / en")
    parser.add_argument("--glossary", default="", help="术语表（可选）")
    parser.add_argument("--extra-instruction", default="", help="额外要求（可选）")
    parser.add_argument(
        "--max-chunk-chars", type=int, default=2800, help="单块最大字符数"
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="失败重试次数")
    parser.add_argument(
        "--connection-only", action="store_true", help="只测试连接，不发送整理内容"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="要整理的文本（直接写在命令行里）")
    group.add_argument("--text-file", help="要整理的文本文件路径（UTF-8）")
    return parser.parse_args(argv)


def resolve_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.text_file:
        path = Path(args.text_file)
        if not path.exists():
            raise SystemExit(f"找不到文本文件：{path}")
        return path.read_text(encoding="utf-8")
    return (
        "嗯，那个，我们今天呢主要讨论一下项目的进度问题。\n"
        "就是那个，我觉得前端大概完成了百分之七十左右吧。\n"
        "然后后端的话可能还需要两周时间，就是这样。"
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    api_key = read_api_key()
    if not api_key:
        print(
            "未找到 API Key。请先设置环境变量（二选一）：\n"
            "  set VOXSCRIBE_API_KEY=sk-...\n"
            "  set DEEPSEEK_API_KEY=sk-...",
            file=sys.stderr,
        )
        return 2

    refiner = DeepSeekRefiner(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    try:
        print(f"Base URL：{refiner.base_url}")
        print(f"模型：{refiner.model}")
        print("正在测试连接…")
        started = time.monotonic()
        reply = refiner.test_connection()
        print(f"连接成功（{time.monotonic() - started:.1f} 秒），模型回复：{reply}")
        if args.connection_only:
            return 0

        text = resolve_text(args)
        transcript = build_transcript(text)
        options = RefineOptions(
            enabled=True,
            style=args.style,
            language=args.language,
            glossary=args.glossary,
            extra_instruction=args.extra_instruction,
            max_chunk_chars=args.max_chunk_chars,
        )

        def on_progress(ratio: float, message: str) -> None:
            print(f"  [{ratio * 100:5.1f}%] {message}")

        print(f"\n开始整理：风格={args.style}，原文 {len(transcript.text)} 字符")
        started = time.monotonic()
        result = refiner.refine(transcript, options, on_progress=on_progress)
        elapsed = time.monotonic() - started

        print("\n================ 整理结果 ================")
        print(result.text)
        print("==========================================")
        print(
            f"分块：{result.chunks}；耗时：{elapsed:.1f} 秒；"
            f"tokens：prompt={result.usage.prompt_tokens}，"
            f"completion={result.usage.completion_tokens}，"
            f"total={result.usage.total_tokens}"
        )
        return 0
    except VoxError as exc:
        print(f"\n调用失败：{exc}", file=sys.stderr)
        if exc.user_hint:
            print(f"建议：{exc.user_hint}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    finally:
        refiner.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:  # 保证 Windows 控制台能输出中文
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - 极少数环境不支持
            pass
    raise SystemExit(main())
