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

实现要点：
* requests.Session 复用连接；OpenAI 兼容的 ``POST {base_url}/chat/completions``，
  非流式；``base_url`` 兼容 ``https://api.deepseek.com``、``.../v1``、
  ``.../chat/completions`` 以及第三方兼容代理。
* 长文本按 segment 累积切块（默认 2800 字），逐块整理后本地拼接；块间把
  上一块结尾的少量文字作为「上文」放进 prompt，仅供衔接、不重复输出。
* 429 / 5xx / 超时 / 连接错误按指数退避重试（尊重 Retry-After），重试等待
  使用 ``cancel.wait`` 以便及时取消。
* 401/403、402、其它 4xx、返回结构异常都转换为带中文 user_hint 的 RefineError。
* 汇总每块 usage；日志不记录 API Key 与正文内容（只记长度）。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from . import prompts
from .errors import CancelledError, RefineError
from .models import STYLES, RefineOptions, RefineResult, RefineUsage, Transcript

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

DEFAULT_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_PATH = "/chat/completions"
MAX_OUTPUT_TOKENS = 4096
DEFAULT_MAX_CHUNK_CHARS = 2800
DEFAULT_CHUNK_OVERLAP_CHARS = 120

# 重试等待：第 n 次失败后等待 RETRY_BASE_SECONDS * 2**(n-1) 秒，上限 RETRY_MAX_SECONDS。
# 测试可以 monkeypatch 这些常量来避免真实等待。
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 16.0
RETRY_AFTER_MAX_SECONDS = 60.0
CONNECT_TIMEOUT_SECONDS = 10.0

# 这些内容不能出现在日志/错误提示里
_API_KEY_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]+")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def normalize_base_url(base_url: str) -> str:
    """归一化 ``base_url``，返回不含 ``/chat/completions`` 的根地址。

    兼容以下写法（第三方 OpenAI 兼容代理同理）：

    * ``https://api.deepseek.com``
    * ``https://api.deepseek.com/v1``
    * ``https://api.deepseek.com/v1/chat/completions``
    * ``https://api.deepseek.com/chat/completions``（自动去掉末尾路径）
    """
    raw = (base_url or "").strip()
    if not raw:
        return DEFAULT_BASE_URL
    if "://" not in raw:
        raw = "https://" + raw
    raw = raw.rstrip("/")
    if raw.lower().endswith(CHAT_COMPLETIONS_PATH):
        raw = raw[: -len(CHAT_COMPLETIONS_PATH)].rstrip("/")
    return raw or DEFAULT_BASE_URL


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate(text: str, limit: int = 200) -> str:
    """压缩空白并截断，避免错误信息过长。"""
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[:limit] + "…"


def _sanitize_detail(text: str, api_key: str = "") -> str:
    """错误详情去敏：去掉可能出现的 API Key 与多余空白。"""
    detail = (text or "").strip()
    if not detail:
        return ""
    if api_key:
        detail = detail.replace(api_key, "***")
    detail = _API_KEY_PATTERN.sub(r"\1***", detail)
    return _truncate(detail)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """解析 ``Retry-After`` 头（仅支持秒数形式；HTTP 日期形式忽略）。"""
    if not value:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return 0.0
    return min(seconds, RETRY_AFTER_MAX_SECONDS)


def _parse_usage(data: Any) -> RefineUsage:
    """从响应 JSON 中解析 usage；字段缺失时尽力补全，绝不因此报错。"""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return RefineUsage()

    prompt_tokens = _safe_int(usage.get("prompt_tokens"), 0)
    completion_tokens = _safe_int(usage.get("completion_tokens"), 0)
    total_raw = usage.get("total_tokens")
    total_tokens = (
        _safe_int(total_raw, 0)
        if total_raw is not None
        else prompt_tokens + completion_tokens
    )
    return RefineUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _extract_content(data: Any) -> str:
    """从 chat.completion 响应中取出 assistant 正文。"""
    if not isinstance(data, dict):
        raise RefineError(
            "DeepSeek 返回的数据格式异常（不是 JSON 对象）",
            hint="请稍后重试；若持续出现，请检查「设置」中的 Base URL "
            "是否指向 OpenAI 兼容接口。",
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RefineError(
            "DeepSeek 返回结果缺少内容（choices 为空）",
            hint="请稍后重试；若持续出现，请检查「设置」中的模型名称是否正确"
            "（一般填 deepseek-chat）。",
        )
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):  # 少数兼容接口用 content parts
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        content = "".join(parts)
    if not isinstance(content, str):
        raise RefineError(
            "DeepSeek 返回结果结构异常（缺少 message.content）",
            hint="请稍后重试；若持续出现，请检查「设置」中的 Base URL "
            "是否指向 OpenAI 兼容接口。",
        )
    text = content.strip()
    if not text:
        raise RefineError(
            "DeepSeek 返回了空内容",
            hint="请稍后重试；也可以关闭「AI 整理」只做语音转写。",
        )
    return text


# ---------------------------------------------------------------------------
# 分块
# ---------------------------------------------------------------------------
@dataclass
class _Chunk:
    """一个待整理的文本块。"""

    text: str
    context: str = ""  # 上一块结尾，仅用于 prompt 衔接，不会拼进结果


def split_transcript(
    transcript: Transcript,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> List[_Chunk]:
    """把转写结果按 ``segment`` 累积成若干块。

    * 单块正文（含换行）不超过 ``max_chunk_chars``；
    * 超过单块上限的 ``segment`` 会先按字符硬切；
    * 块间 ``context`` 取上一块结尾 ``chunk_overlap_chars`` 个字符，
      只放进 prompt 作为衔接，不会被拼接到最终结果里。
    """
    max_chunk_chars = max(1, _safe_int(max_chunk_chars, DEFAULT_MAX_CHUNK_CHARS))
    overlap = max(0, _safe_int(chunk_overlap_chars, DEFAULT_CHUNK_OVERLAP_CHARS))
    overlap = min(overlap, max_chunk_chars // 2)

    pieces: List[str] = []
    for segment in transcript.segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        if len(text) <= max_chunk_chars:
            pieces.append(text)
        else:
            pieces.extend(
                text[start : start + max_chunk_chars]
                for start in range(0, len(text), max_chunk_chars)
            )
    if not pieces:
        return []

    chunks: List[_Chunk] = []
    current: List[str] = []

    def flush() -> None:
        if not current:
            return
        body = "\n".join(current)
        context = ""
        if chunks and overlap > 0:
            context = chunks[-1].text[-overlap:]
        chunks.append(_Chunk(text=body, context=context))
        current.clear()

    for piece in pieces:
        if current and len("\n".join(current)) + 1 + len(piece) > max_chunk_chars:
            flush()
        current.append(piece)
    flush()
    return chunks


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------
class DeepSeekRefiner:
    """DeepSeek Chat Completions 封装（单实例单工作线程使用即可）。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "deepseek-chat",
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self.base_url = normalize_base_url(base_url)
        self.chat_url = self.base_url + CHAT_COMPLETIONS_PATH
        self.model = (model or "deepseek-chat").strip() or "deepseek-chat"

        timeout_value = _safe_float(timeout, 120.0)
        if timeout_value <= 0:
            timeout_value = 120.0
        self.timeout = timeout_value
        # (连接超时, 读取超时)
        self._request_timeout = (min(CONNECT_TIMEOUT_SECONDS, timeout_value), timeout_value)

        self.max_retries = max(0, _safe_int(max_retries, 2))

        self._session = requests.Session()
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "VoxScribe/1.0 (DeepSeek refine)",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._session.headers.update(headers)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def refine(
        self,
        transcript: Transcript,
        options: RefineOptions,
        *,
        on_progress: Optional[ProgressCallback] = None,
        cancel: Optional[threading.Event] = None,
    ) -> RefineResult:
        """把转写文本分块交给 DeepSeek 整理，返回拼接后的结果。"""
        self._ensure_api_key()
        cancel = cancel or threading.Event()
        if cancel.is_set():
            raise CancelledError("任务已取消")

        options = options or RefineOptions()
        style = prompts.style_key(options.style)
        if (options.style or "").strip() not in STYLES:
            log.warning("未知的整理风格 %r，已回退为 oral", options.style)
        language = (options.language or "zh").strip() or "zh"
        temperature = max(0.0, min(2.0, _safe_float(options.temperature, 0.3)))
        max_chunk_chars = _safe_int(options.max_chunk_chars, DEFAULT_MAX_CHUNK_CHARS)
        if max_chunk_chars <= 0:
            max_chunk_chars = DEFAULT_MAX_CHUNK_CHARS
        chunk_overlap_chars = _safe_int(
            options.chunk_overlap_chars, DEFAULT_CHUNK_OVERLAP_CHARS
        )
        if chunk_overlap_chars < 0:
            chunk_overlap_chars = 0

        started = time.monotonic()
        chunks = split_transcript(transcript, max_chunk_chars, chunk_overlap_chars)
        total = len(chunks)
        if total == 0:
            log.warning("转写内容为空，跳过 DeepSeek 整理")
            return RefineResult(text="", chunks=0, elapsed=time.monotonic() - started)

        system = prompts.system_prompt(style, language)
        usage = RefineUsage()
        parts: List[str] = []
        log.info(
            "开始整理文字：风格=%s，语言=%s，共 %d 段，原文 %d 字符",
            style,
            language,
            total,
            len(transcript.text),
        )

        for index, chunk in enumerate(chunks, start=1):
            if cancel.is_set():
                raise CancelledError("任务已取消")
            if on_progress is not None:
                on_progress((index - 1) / total, f"正在整理第 {index}/{total} 段…")

            user_content = prompts.user_prompt(
                chunk.text,
                context=chunk.context,
                glossary=options.glossary or "",
                extra_instruction=options.extra_instruction or "",
            )
            content, chunk_usage = self._chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                cancel=cancel,
                purpose=f"整理第 {index}/{total} 段",
            )
            parts.append(content)
            usage.add(chunk_usage)
            log.info(
                "第 %d/%d 段整理完成：输入 %d 字符，输出 %d 字符",
                index,
                total,
                len(chunk.text),
                len(content),
            )

        if on_progress is not None:
            on_progress(1.0, f"文字整理完成（共 {total} 段）")

        elapsed = time.monotonic() - started
        text = "\n\n".join(part for part in parts if part)
        log.info(
            "文字整理完成：%d 段，用时 %.1f 秒，tokens=%d",
            total,
            elapsed,
            usage.total_tokens,
        )
        return RefineResult(text=text, chunks=total, usage=usage, elapsed=elapsed)

    def test_connection(self) -> str:
        """发一条极短消息验证 Key / Base URL / 模型是否可用。"""
        self._ensure_api_key()
        started = time.monotonic()
        content, usage = self._chat(
            messages=[
                {"role": "system", "content": prompts.CONNECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompts.CONNECTION_USER_PROMPT},
            ],
            temperature=0.0,
            max_tokens=prompts.CONNECTION_MAX_TOKENS,
            cancel=threading.Event(),
            purpose="连接测试",
        )
        log.info(
            "DeepSeek 连接测试成功：用时 %.1f 秒，回复 %d 字符，tokens=%d",
            time.monotonic() - started,
            len(content),
            usage.total_tokens,
        )
        return content

    def close(self) -> None:
        """释放底层连接池（可选；进程退出时也会自动回收）。"""
        try:
            self._session.close()
        except Exception:  # pragma: no cover - 关闭失败不影响主流程
            log.debug("关闭 requests.Session 失败", exc_info=True)

    # ------------------------------------------------------------------
    # 请求与错误处理
    # ------------------------------------------------------------------
    def _ensure_api_key(self) -> None:
        if not self._api_key:
            raise RefineError(
                "尚未配置 DeepSeek API Key，无法整理文字",
                hint="请在「设置」中填写 API Key（platform.deepseek.com 获取），"
                "或关闭「AI 整理」只做语音转写。",
            )

    def _chat(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        cancel: threading.Event,
        purpose: str,
    ) -> Tuple[str, RefineUsage]:
        """发送一次（带重试的）chat.completions 请求，返回 (正文, usage)。

        ``max_retries`` 表示首次请求失败后最多再重试的次数
        （总请求次数 = 1 + max_retries）。
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        max_attempts = self.max_retries + 1
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            if cancel.is_set():
                raise CancelledError("任务已取消")

            try:
                response = self._session.post(
                    self.chat_url, json=payload, timeout=self._request_timeout
                )
            except requests.exceptions.Timeout as exc:
                if attempt >= max_attempts:
                    raise RefineError(
                        f"请求 DeepSeek 超时（{purpose}，已重试 {attempt - 1} 次）",
                        hint="请检查网络是否稳定；网络较慢时可在「设置」中调大"
                        "超时时间，或稍后再试。",
                    ) from exc
                delay = self._retry_delay(attempt, None)
                log.warning(
                    "%s：请求超时，%.1f 秒后重试（第 %d/%d 次）",
                    purpose,
                    delay,
                    attempt,
                    max_attempts - 1,
                )
                self._wait(delay, cancel)
                continue
            except requests.exceptions.ConnectionError as exc:
                if attempt >= max_attempts:
                    raise RefineError(
                        f"无法连接 DeepSeek 服务（{purpose}，已重试 {attempt - 1} 次）",
                        hint="请检查网络、代理与防火墙设置，并确认「设置」中的 "
                        f"Base URL 正确（当前：{self.base_url}）。",
                    ) from exc
                delay = self._retry_delay(attempt, None)
                log.warning(
                    "%s：连接失败，%.1f 秒后重试（第 %d/%d 次）",
                    purpose,
                    delay,
                    attempt,
                    max_attempts - 1,
                )
                self._wait(delay, cancel)
                continue
            except requests.exceptions.RequestException as exc:
                detail = _truncate(str(exc))
                raise RefineError(
                    f"请求 DeepSeek 失败（{purpose}）：{detail}",
                    hint="请检查网络连接与「设置」中的 Base URL"
                    f"（当前：{self.base_url}）。",
                ) from exc

            try:
                status = response.status_code
                if 200 <= status < 300:
                    return self._parse_chat_response(response)
                if status in (401, 403):
                    raise self._build_auth_error(response)
                if status == 402:
                    raise self._build_balance_error(response)
                if status == 408 or status == 429 or 500 <= status < 600:
                    if attempt >= max_attempts:
                        if status == 429:
                            raise self._build_rate_limit_error(attempt)
                        raise self._build_server_error(status, attempt)
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    delay = self._retry_delay(attempt, retry_after)
                    log.warning(
                        "%s：HTTP %s，%.1f 秒后重试（第 %d/%d 次）：%s",
                        purpose,
                        status,
                        delay,
                        attempt,
                        max_attempts - 1,
                        _sanitize_detail(self._response_detail(response), self._api_key),
                    )
                    self._wait(delay, cancel)
                    continue
                raise self._build_http_error(status, response)
            finally:
                response.close()

        # 理论上不可达：循环内所有分支要么 return、要么 raise、要么 continue
        raise RefineError(  # pragma: no cover
            "DeepSeek 请求失败（未知原因）",
            hint="请稍后重试；若持续失败，可暂时关闭「AI 整理」。",
        )

    def _parse_chat_response(self, response: requests.Response) -> Tuple[str, RefineUsage]:
        try:
            data = response.json()
        except ValueError as exc:
            detail = _sanitize_detail(
                _truncate(getattr(response, "text", "") or ""), self._api_key
            )
            suffix = f"：{detail}" if detail else ""
            raise RefineError(
                f"DeepSeek 返回的数据不是合法 JSON{suffix}",
                hint="请稍后重试；若持续出现，请检查「设置」中的 Base URL "
                "是否指向 OpenAI 兼容的 /chat/completions 接口。",
            ) from exc
        return _extract_content(data), _parse_usage(data)

    def _retry_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        """计算下一次重试前的等待秒数。"""
        if retry_after is not None:
            return max(0.0, min(float(retry_after), RETRY_AFTER_MAX_SECONDS))
        delay = RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1))
        return min(delay, RETRY_MAX_SECONDS)

    def _wait(self, seconds: float, cancel: threading.Event) -> None:
        """可取消的等待；被取消时抛 CancelledError。"""
        seconds = max(0.0, float(seconds))
        if seconds <= 0:
            if cancel.is_set():
                raise CancelledError("任务已取消")
            return
        log.debug("等待 %.2f 秒后重试 DeepSeek 请求", seconds)
        if cancel.wait(seconds):
            raise CancelledError("任务已取消")

    # ------------------------------------------------------------------
    # 错误构造
    # ------------------------------------------------------------------
    def _response_detail(self, response: requests.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return _truncate(getattr(response, "text", "") or "")
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("code")
                if isinstance(message, str) and message.strip():
                    return _truncate(message)
            message = data.get("message")
            if isinstance(message, str) and message.strip():
                return _truncate(message)
        return ""

    def _build_auth_error(self, response: requests.Response) -> RefineError:
        detail = _sanitize_detail(self._response_detail(response), self._api_key)
        suffix = f"：{detail}" if detail else ""
        return RefineError(
            f"DeepSeek 鉴权失败：API Key 无效或无权限（HTTP {response.status_code}）{suffix}",
            hint="请打开「设置」检查 DeepSeek API Key 是否正确"
            "（platform.deepseek.com → API Keys 创建），确认没有多余空格，"
            "或更换有效 Key 后重试。",
        )

    def _build_balance_error(self, response: requests.Response) -> RefineError:
        detail = _sanitize_detail(self._response_detail(response), self._api_key)
        suffix = f"：{detail}" if detail else ""
        return RefineError(
            f"DeepSeek 账户余额不足，无法继续整理{suffix}",
            hint="请到 platform.deepseek.com 充值后重试；也可以暂时关闭"
            "「AI 整理」只做语音转写。",
        )

    def _build_rate_limit_error(self, attempts: int) -> RefineError:
        return RefineError(
            f"DeepSeek 接口请求过于频繁（HTTP 429，已重试 {attempts - 1} 次）",
            hint="请求频率或额度受限，请稍等一会儿再试；长音频也可以分成"
            "多个文件处理。",
        )

    def _build_server_error(self, status: int, attempts: int) -> RefineError:
        return RefineError(
            f"DeepSeek 服务暂时不可用（HTTP {status}，已重试 {attempts - 1} 次）",
            hint="这是 DeepSeek 服务端的问题，通常稍等片刻再试即可；"
            "若持续失败，可暂时关闭「AI 整理」。",
        )

    def _build_http_error(self, status: int, response: requests.Response) -> RefineError:
        detail = _sanitize_detail(self._response_detail(response), self._api_key)
        suffix = f"：{detail}" if detail else ""
        if status == 400:
            return RefineError(
                f"DeepSeek 拒绝了请求（HTTP 400）{suffix}",
                hint="请检查「设置」中的模型名称与 Base URL 是否正确；"
                "恢复默认参数后也可再试。",
            )
        if status == 404:
            return RefineError(
                f"找不到 DeepSeek 接口（HTTP 404）{suffix}",
                hint="请检查「设置」中的 Base URL"
                f"（当前：{self.base_url}），通常应填写 "
                "https://api.deepseek.com 或 https://api.deepseek.com/v1。",
            )
        if status == 413:
            return RefineError(
                f"单次请求内容过长（HTTP 413）{suffix}",
                hint="请在「设置」中调小「单块最大字符数」后重试，"
                "或分段处理长音频。",
            )
        return RefineError(
            f"DeepSeek 请求失败（HTTP {status}）{suffix}",
            hint="请检查网络与 API 设置后重试；若持续失败，"
            "可暂时关闭「AI 整理」。",
        )


__all__ = [
    "DeepSeekRefiner",
    "ProgressCallback",
    "normalize_base_url",
    "split_transcript",
    "DEFAULT_BASE_URL",
    "CHAT_COMPLETIONS_PATH",
    "MAX_OUTPUT_TOKENS",
]
