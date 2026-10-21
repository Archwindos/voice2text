"""导出模块：txt / md / docx / srt。

负责人：gui-dev。接口契约（pipeline 依赖，改动需同步）：

    paths = export_result(result, out_dir=Path(...), base_name="会议记录",
                          formats=["txt", "md", "docx", "srt"],
                          title="会议记录", include_raw=True) -> list[Path]

要求：
* 有整理稿时，txt/md/docx 输出整理稿；同时另存一份
  ``<base_name>_原始转写.txt``（include_raw=True 时）。
* 没有整理稿时，txt/md/docx 输出原始转写。
* srt 始终由原始转写的时间戳生成。
* 文件名重复时自动加 ``(2)``、``(3)`` 后缀，绝不覆盖已有文件。
* 文件开头可附少量元信息（源文件、时长、生成时间、整理风格），
  保证正文仍可直接使用。
* 所有文本以 UTF-8 写入（docx 由 python-docx 生成）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .models import JobResult


def export_result(
    result: JobResult,
    *,
    out_dir: Path,
    base_name: str,
    formats: Sequence[str],
    title: Optional[str] = None,
    include_raw: bool = True,
) -> List[Path]:
    """按 ``formats`` 导出结果文件，返回实际写出的路径列表。"""
    raise NotImplementedError("导出模块尚未实现（gui-dev）")
