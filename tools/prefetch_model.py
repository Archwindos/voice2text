"""预下载 faster-whisper 识别模型，避免第一次转写时长时间等待。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe tools\\prefetch_model.py --model small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 部分网络环境 Xet 下载通道不可用；走普通 HTTP 更稳（镜像也支持）。
import os  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from voxscribe.config import ASR_MODELS, AppSettings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="预下载 VoxScribe 语音识别模型")
    parser.add_argument(
        "--model",
        default="small",
        choices=list(ASR_MODELS),
        help="模型大小（默认 small，中文推荐）",
    )
    parser.add_argument("--model-dir", default="", help="模型缓存目录（默认 data/models）")
    args = parser.parse_args()

    settings = AppSettings.load()
    model_dir = Path(args.model_dir) if args.model_dir else settings.resolve_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型目录：{model_dir}")
    print(f"正在下载/校验模型 {args.model} …")
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - 依赖缺失
        print(f"未安装 faster-whisper：{exc}", file=sys.stderr)
        print("请先运行 setup.bat 安装依赖。", file=sys.stderr)
        return 1

    try:
        # 构造即完成下载 + int8 量化校验；随后释放内存
        model = WhisperModel(
            args.model,
            device="cpu",
            compute_type="int8",
            download_root=str(model_dir),
            cpu_threads=1,
        )
        del model
    except Exception as exc:  # pragma: no cover - 网络错误
        print(f"模型下载失败：{exc}", file=sys.stderr)
        print("提示：可以稍后重试；第一次转写时程序也会自动下载。", file=sys.stderr)
        return 1

    print(f"完成：{args.model} 已缓存到 {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
