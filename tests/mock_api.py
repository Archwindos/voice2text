"""测试用的本地假 DeepSeek ``/chat/completions`` 服务。

只依赖标准库（``http.server.ThreadingHTTPServer`` + ``threading``），
在 127.0.0.1 的随机端口上监听，不会访问外网。

典型用法::

    with MockAPIServer() as server:
        refiner = DeepSeekRefiner(api_key="test-key", base_url=server.url)
        result = refiner.refine(transcript, RefineOptions())
    assert len(server.requests) == 1

默认行为：把 user prompt 中「【待整理正文】」之后的内容原样返回（echo），
并附带固定 usage，方便断言分块与拼接结果。

也可以自定义响应：

* ``server.handler = fixed_handler("固定的整理结果")`` —— 固定回复
* ``server.push(MockResponse(401, error_response_body("invalid key")))``
  —— 按顺序返回预设响应（用完后回到默认 handler）
* ``server.fail_times(2, status=429, retry_after="0")`` —— 前 N 次返回 429
* ``server.delay = 0.2`` —— 每次响应前延迟
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

from voxscribe import prompts

# 默认 usage：单块成功整理时预期值固定，便于测试汇总逻辑
DEFAULT_USAGE: Dict[str, int] = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
}


# ---------------------------------------------------------------------------
# 响应构造
# ---------------------------------------------------------------------------
def chat_response(
    content: str,
    *,
    usage: Optional[Dict[str, int]] = None,
    model: str = "deepseek-flash",
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """构造一份 OpenAI 兼容的 chat.completion 响应体。"""
    if usage is None:
        response_usage = dict(DEFAULT_USAGE)
    else:
        response_usage = dict(usage)
        if response_usage.get("total_tokens") is None:
            response_usage["total_tokens"] = int(
                response_usage.get("prompt_tokens", 0)
            ) + int(response_usage.get("completion_tokens", 0))
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": response_usage,
    }


def error_response_body(
    message: str,
    *,
    type_: str = "invalid_request_error",
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """构造 OpenAI 风格的错误响应体。"""
    error: Dict[str, Any] = {"message": message, "type": type_}
    if code is not None:
        error["code"] = code
    return {"error": error}


# ---------------------------------------------------------------------------
# 请求/响应数据结构
# ---------------------------------------------------------------------------
@dataclass
class MockRequest:
    """假服务收到的请求记录。"""

    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    json: Any = None

    @property
    def bearer_token(self) -> str:
        value = self.headers.get("authorization", "")
        if value.lower().startswith("bearer "):
            return value[7:].strip()
        return ""

    @property
    def body(self) -> str:
        """user prompt 中「待整理正文」部分（没有标记时返回全文）。"""
        return extract_body(last_user_content(self.json))


@dataclass
class MockResponse:
    """一次预设响应。``delay=None`` 表示使用服务级的默认延迟。"""

    status: int = 200
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    delay: Optional[float] = None


ChatHandler = Callable[[MockRequest], MockResponse]


# ---------------------------------------------------------------------------
# prompt 解析辅助
# ---------------------------------------------------------------------------
def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def last_user_content(payload: Any) -> str:
    """取出请求里最后一条 user 消息的文本。"""
    if not isinstance(payload, dict):
        return ""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message.get("content"))
    return ""


def system_content(payload: Any) -> str:
    """取出请求里 system 消息的文本。"""
    if not isinstance(payload, dict):
        return ""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            return _message_text(message.get("content"))
    return ""


def extract_body(user_content: str) -> str:
    """从 user prompt 中取出「待整理正文」；没有标记时返回全文。"""
    if not user_content:
        return ""
    marker = prompts.BODY_MARKER
    if marker in user_content:
        return user_content.rsplit(marker, 1)[1].strip()
    return user_content.strip()


# ---------------------------------------------------------------------------
# 默认 handler
# ---------------------------------------------------------------------------
def echo_handler(request: MockRequest) -> MockResponse:
    """把「待整理正文」原样返回，用于验证分块拼接。"""
    return MockResponse(200, chat_response(request.body))


def fixed_handler(
    text: str,
    *,
    usage: Optional[Dict[str, int]] = None,
    delay: Optional[float] = None,
) -> ChatHandler:
    """返回一个固定文本回复的 handler。"""

    def handler(request: MockRequest) -> MockResponse:
        return MockResponse(200, chat_response(text, usage=usage), delay=delay)

    return handler


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------
class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockDeepSeek/1.0"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 接口
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        payload: Any = None
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None

        record = MockRequest(
            method="POST",
            path=self.path,
            headers={key.lower(): value for key, value in self.headers.items()},
            json=payload,
        )
        mock: "MockAPIServer" = self.server.mock  # type: ignore[attr-defined]
        response = mock.dispatch(record)

        delay = response.delay if response.delay is not None else mock.delay
        if delay and delay > 0:
            time.sleep(delay)

        data = response.body
        if data is None:
            data = b""
        elif isinstance(data, bytes):
            pass
        elif isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, str):
            data = data.encode("utf-8")
        else:  # pragma: no cover - 防御性分支
            data = str(data).encode("utf-8")

        try:
            self.send_response(response.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            if data:
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):  # 客户端提前断开
            pass

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(404)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """静默 HTTP 日志，保持 pytest 输出干净。"""


class MockAPIServer:
    """线程化的本地假 DeepSeek 服务。"""

    def __init__(
        self,
        handler: Optional[ChatHandler] = None,
        *,
        delay: float = 0.0,
    ) -> None:
        self._handler: ChatHandler = handler or echo_handler
        self._delay = float(delay or 0.0)
        self._queue: List[MockResponse] = []
        self._requests: List[MockRequest] = []
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- 生命周期 ------------------------------------------------------
    def start(self) -> "MockAPIServer":
        if self._httpd is not None:
            return self
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        httpd.daemon_threads = True
        httpd.mock = self  # type: ignore[attr-defined]
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="mock-deepseek",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd = None
        self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> "MockAPIServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    # -- 地址与记录 ----------------------------------------------------
    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("MockAPIServer 尚未启动，请先调用 start() 或使用 with")
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def requests(self) -> List[MockRequest]:
        with self._lock:
            return list(self._requests)

    @property
    def request_count(self) -> int:
        with self._lock:
            return len(self._requests)

    @property
    def last_request(self) -> Optional[MockRequest]:
        with self._lock:
            return self._requests[-1] if self._requests else None

    # -- 行为配置 ------------------------------------------------------
    @property
    def handler(self) -> ChatHandler:
        return self._handler

    @handler.setter
    def handler(self, value: Optional[ChatHandler]) -> None:
        self._handler = value or echo_handler

    @property
    def delay(self) -> float:
        return self._delay

    @delay.setter
    def delay(self, value: float) -> None:
        self._delay = float(value or 0.0)

    def push(self, response: MockResponse) -> None:
        """追加一条按顺序消费的预设响应。"""
        with self._lock:
            self._queue.append(response)

    def push_chat(
        self,
        content: str,
        *,
        usage: Optional[Dict[str, int]] = None,
        delay: Optional[float] = None,
    ) -> None:
        self.push(MockResponse(200, chat_response(content, usage=usage), delay=delay))

    def push_error(
        self,
        status: int,
        message: str,
        *,
        retry_after: Optional[str] = None,
        type_: str = "invalid_request_error",
    ) -> None:
        headers: Dict[str, str] = {}
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        self.push(MockResponse(status, error_response_body(message, type_=type_), headers))

    def fail_times(
        self,
        times: int,
        *,
        status: int = 429,
        message: str = "Too Many Requests",
        retry_after: Optional[str] = "0",
    ) -> None:
        """让接下来的 ``times`` 次请求返回指定错误。"""
        for _ in range(max(0, int(times))):
            self.push_error(status, message, retry_after=retry_after)

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self._requests.clear()

    # -- 分发 ----------------------------------------------------------
    def dispatch(self, record: MockRequest) -> MockResponse:
        """记录请求并决定返回哪条响应（供 HTTP handler 调用）。"""
        with self._lock:
            self._requests.append(record)
            queued = self._queue.pop(0) if self._queue else None
        if queued is not None:
            return queued
        try:
            response = self._handler(record)
        except Exception as exc:  # 测试 handler 写错时给出可读的 500
            response = MockResponse(
                500, error_response_body(f"mock handler error: {exc!r}")
            )
        if response is None:  # pragma: no cover - 防御性分支
            response = MockResponse(200, chat_response(record.body))
        return response


__all__ = [
    "DEFAULT_USAGE",
    "MockAPIServer",
    "MockRequest",
    "MockResponse",
    "chat_response",
    "error_response_body",
    "echo_handler",
    "extract_body",
    "fixed_handler",
    "last_user_content",
    "system_content",
]
