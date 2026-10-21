"""DeepSeek 文本整理模块。

负责人：refine-dev。接口契约（pipeline 依赖，改动需同步）：

    refiner = DeepSeekRefiner(api_key="sk-...",
                              base_url="https://api.deepseek.com",
                              model="deepseek-chat",
                              timeout=120, max_retries=2)
    refiner.refine(transcript, RefineOptions(...),
                   on_progress=callable(ratio, message),
                   cancel=threading.Event()) -> RefineResult
    refiner.test_connection() -> str    # 返回模型回复，用于「测试连接」按钮

要求：
* 使用 requests，OpenAI 兼容的 /chat/completions 接口，不带流式。
* 长文本按段落/片段切块（默认约 2800 字），块间带少量重叠上下文，
  逐块整理后在本地拼接，避免模型跨块编造内容。
* 429 / 5xx / 网络错误按指数退避重试；鉴权失败等 4xx 直接抛出
  voxscribe.errors.RefineError 并给出中文提示。
* 取消：块之间与重试等待期间检查 cancel.is_set()，抛出
  voxscribe.errors.CancelledError。
* 记录并汇总 usage token 数。
* prompt 文案统一放在 prompts.py，便于调参。
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from .models import RefineOptions, RefineResult, Transcript

ProgressCallback = Callable[[float, str], None]


class DeepSeekRefiner:
    """DeepSeek Chat Completions 封装。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        raise NotImplementedError("DeepSeek 整理模块尚未实现（refine-dev）")

    def refine(
        self,
        transcript: Transcript,
        options: RefineOptions,
        *,
        on_progress: Optional[ProgressCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> RefineResult:
        raise NotImplementedError

    def test_connection(self) -> str:
        raise NotImplementedError
