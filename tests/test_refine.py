"""DeepSeek 整理模块测试。

全部使用 ``tests/mock_api.MockAPIServer``（本地假 HTTP 服务），
不访问真实 DeepSeek API、不消耗额度。
"""

from __future__ import annotations

import logging
import socket
import threading
import time

import pytest

from tests.mock_api import (
    DEFAULT_USAGE,
    MockAPIServer,
    MockResponse,
    error_response_body,
    extract_body,
    last_user_content,
    system_content,
)
from voxscribe import prompts
from voxscribe.errors import CancelledError, RefineError
from voxscribe.models import (
    STYLES,
    RefineOptions,
    RefineResult,
    Segment,
    Transcript,
)
from voxscribe.refine import (
    DeepSeekRefiner,
    MAX_OUTPUT_TOKENS,
    normalize_base_url,
    split_transcript,
)

TEST_KEY = "sk-test-1234567890"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def server():
    srv = MockAPIServer()
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def make_refiner(server: MockAPIServer, **overrides) -> DeepSeekRefiner:
    kwargs = dict(
        api_key=TEST_KEY,
        base_url=server.url,
        model="deepseek-flash",
        timeout=10.0,
        max_retries=2,
    )
    kwargs.update(overrides)
    return DeepSeekRefiner(**kwargs)


@pytest.fixture()
def refiner(server):
    client = make_refiner(server)
    try:
        yield client
    finally:
        client.close()


def run_refine(refiner, transcript, options=None, **kwargs):
    """执行整理并收集进度回调。"""
    progress = []
    result = refiner.refine(
        transcript,
        options or RefineOptions(),
        on_progress=lambda ratio, message: progress.append((ratio, message)),
        **kwargs,
    )
    return result, progress


def make_long_transcript(segments: int = 12, segment_chars: int = 180) -> Transcript:
    """构造长度可预测的长转写：每段恰好 ``segment_chars`` 个字符。"""
    result = []
    for index in range(segments):
        prefix = f"第{index:03d}段"
        filler = "分块测试内容。" * ((segment_chars - len(prefix)) // 7 + 2)
        text = (prefix + filler)[:segment_chars]
        result.append(Segment(float(index), float(index + 1), text))
    return Transcript(
        segments=result, language="zh", duration=float(segments)
    )


# ---------------------------------------------------------------------------
# base_url 归一化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", "https://api.deepseek.com"),
        ("https://api.deepseek.com", "https://api.deepseek.com"),
        ("https://api.deepseek.com/", "https://api.deepseek.com"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/v1/", "https://api.deepseek.com/v1"),
        (
            "https://api.deepseek.com/v1/chat/completions",
            "https://api.deepseek.com/v1",
        ),
        (
            "https://api.deepseek.com/chat/completions/",
            "https://api.deepseek.com",
        ),
        ("api.deepseek.com/v1", "https://api.deepseek.com/v1"),
        ("  https://proxy.example.com/v1  ", "https://proxy.example.com/v1"),
    ],
)
def test_normalize_base_url(raw, expected):
    assert normalize_base_url(raw) == expected


@pytest.mark.parametrize(
    "suffix, expected_path",
    [
        ("", "/chat/completions"),
        ("/v1", "/v1/chat/completions"),
        ("/chat/completions", "/chat/completions"),
        ("/v1/", "/v1/chat/completions"),
        ("/v1/chat/completions", "/v1/chat/completions"),
    ],
)
def test_base_url_variants_hit_expected_path(
    server, sample_transcript, suffix, expected_path
):
    client = DeepSeekRefiner(api_key=TEST_KEY, base_url=server.url + suffix)
    try:
        result = client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()
    assert result.text == sample_transcript.text
    assert server.last_request is not None
    assert server.last_request.path == expected_path
    assert server.request_count == 1


# ---------------------------------------------------------------------------
# 短文本 / 请求结构
# ---------------------------------------------------------------------------
def test_single_chunk_success(server, refiner, sample_transcript):
    result, progress = run_refine(refiner, sample_transcript)

    assert isinstance(result, RefineResult)
    assert result.text == sample_transcript.text
    assert result.chunks == 1
    assert result.usage.prompt_tokens == DEFAULT_USAGE["prompt_tokens"]
    assert result.usage.completion_tokens == DEFAULT_USAGE["completion_tokens"]
    assert result.usage.total_tokens == DEFAULT_USAGE["total_tokens"]
    assert result.elapsed >= 0.0

    assert server.request_count == 1
    request = server.last_request
    assert request.method == "POST"
    assert request.path == "/chat/completions"
    assert request.bearer_token == TEST_KEY
    assert request.headers.get("authorization", "").startswith("Bearer ")

    payload = request.json
    assert payload["model"] == "deepseek-flash"
    assert payload["reasoning_effort"] == "none"
    assert payload["stream"] is False
    assert payload["max_tokens"] == MAX_OUTPUT_TOKENS
    assert payload["max_tokens"] == 8192
    assert payload["temperature"] == pytest.approx(0.3)
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert system_content(payload) == prompts.system_prompt("oral", "zh")
    assert extract_body(last_user_content(payload)) == sample_transcript.text

    # 进度：开始 0.0，结束 1.0，且带「第 i/n 段」提示
    assert [ratio for ratio, _ in progress] == [0.0, 1.0]
    assert "正在整理第 1/1 段" in progress[0][1]
    assert progress[-1][0] == 1.0


def test_fixed_response(server, sample_transcript):
    server.handler = lambda request: MockResponse(
        200,
        {
            "choices": [
                {"message": {"role": "assistant", "content": "固定的整理结果"}}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        },
    )
    client = make_refiner(server)
    try:
        result = client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()
    assert result.text == "固定的整理结果"
    assert result.usage.total_tokens == 11


def test_all_styles_use_their_system_prompt(server, sample_transcript):
    # 文件转写页仍只有 5 种风格；语音输入另有 direct/person/ai 三种 prompt
    assert set(STYLES) == {"oral", "proofread", "meeting", "verbatim", "summary"}
    assert "proofread" in STYLES
    assert len(prompts.STYLE_PROMPTS) == 8
    assert set(prompts.STYLE_PROMPTS) == set(STYLES) | {"direct", "person", "ai"}
    for style in STYLES:
        client = make_refiner(server)
        try:
            result = client.refine(sample_transcript, RefineOptions(style=style))
        finally:
            client.close()
        assert result.text == sample_transcript.text
        assert system_content(server.last_request.json) == prompts.system_prompt(
            style, "zh"
        )


def test_quick_modes_prompt_and_reasoning_effort(server, sample_transcript):
    """语音输入三模式：直出无思考 / 对人 low / 对AI high。"""
    cases = (
        ("direct", "none", "原汁原味"),
        ("person", "low", "即时消息"),
        ("ai", "high", "提示词"),
    )
    for style, effort, keyword in cases:
        client = make_refiner(server)
        try:
            result = client.refine(
                sample_transcript,
                RefineOptions(style=style, reasoning_effort=effort),
            )
        finally:
            client.close()
        payload = server.last_request.json
        assert payload["reasoning_effort"] == effort
        assert system_content(payload) == prompts.system_prompt(style, "zh")
        assert keyword in system_content(payload)
        assert result.text == sample_transcript.text


def test_proofread_style_prompt_and_result(server, refiner, sample_transcript):
    """「口述校对」：system prompt 强调常用词/上下文纠错/错别字/术语表，结果与 usage 正常。"""
    result, progress = run_refine(
        refiner, sample_transcript, RefineOptions(style="proofread")
    )

    payload = server.last_request.json
    system = system_content(payload)
    for keyword in (
        "校对",
        "错别字",
        "同音字",
        "同音",
        "上下文",
        "常见用语",
        "常用",
        "语义不通",
        "明显识别错误",
        "音轨",
        "术语表",
        "口头禅",
        "不要翻译",
        "英文",
    ):
        assert keyword in system, f"proofread system prompt 缺少关键词：{keyword}"
    # 整词误识别示例：阴轨→音轨、以下→一下、在确认→再确认、帐目→账目
    assert "阴轨" in system and "音轨" in system
    assert "讨论以下项目" in system and "讨论一下项目" in system
    assert "在确认" in system and "再确认" in system
    assert "帐目" in system and "账目" in system
    # 防止过度纠错：专有名词/人名/数字不硬改，拿不准保留原文
    assert "不要硬改" in system
    assert "不要猜" in system or "拿不准" in system
    assert "输出语言：简体中文" in system
    assert system == prompts.system_prompt("proofread", "zh")

    # 待整理正文照常传入，结果正常返回
    assert extract_body(last_user_content(payload)) == sample_transcript.text
    assert result.text == sample_transcript.text
    assert result.chunks == 1
    assert result.usage.prompt_tokens == DEFAULT_USAGE["prompt_tokens"]
    assert result.usage.completion_tokens == DEFAULT_USAGE["completion_tokens"]
    assert result.usage.total_tokens == DEFAULT_USAGE["total_tokens"]
    assert result.elapsed >= 0.0
    assert progress[-1][0] == 1.0


def test_proofread_whole_word_guidance_only_for_proofread():
    """「按常用语/上下文纠正整词误识别」的要求只加在 proofread，不影响其他风格。"""
    proofread = prompts.system_prompt("proofread", "zh")
    assert "上下文" in proofread
    assert "明显识别错误" in proofread
    assert "常见用语" in proofread
    assert "常用" in proofread
    assert "语义不通" in proofread
    assert "阴轨" in proofread and "音轨" in proofread
    assert "讨论以下项目" in proofread
    assert "讨论一下项目" in proofread
    for style in ("oral", "meeting", "verbatim", "summary"):
        system = prompts.system_prompt(style, "zh")
        assert "阴轨" not in system
        assert "音轨" not in system
        assert "讨论以下项目" not in system
        assert "明显识别错误" not in system
        assert "语义不通" not in system


def test_proofread_style_with_glossary(server, refiner, sample_transcript):
    options = RefineOptions(
        style="proofread",
        glossary="VoxScribe（语声成文）：项目名称\nDeepSeek：模型服务商",
    )
    run_refine(refiner, sample_transcript, options)
    user_content = last_user_content(server.last_request.json)
    assert prompts.GLOSSARY_MARKER in user_content
    assert "VoxScribe（语声成文）" in user_content
    assert "DeepSeek" in user_content
    # 术语表优先级最高的要求写在 proofread 的 system prompt 里
    assert "术语表" in system_content(server.last_request.json)


def test_unknown_style_falls_back_to_oral(server, refiner, sample_transcript):
    run_refine(refiner, sample_transcript, RefineOptions(style="不存在的风格"))
    assert system_content(server.last_request.json) == prompts.system_prompt(
        "oral", "zh"
    )


def test_glossary_and_extra_instruction_in_user_prompt(
    server, refiner, sample_transcript
):
    options = RefineOptions(
        style="meeting",
        glossary="VoxScribe（语声成文）：项目名称",
        extra_instruction="保留 emoji，不要输出时间轴。",
    )
    run_refine(refiner, sample_transcript, options)
    user_content = last_user_content(server.last_request.json)

    assert prompts.GLOSSARY_MARKER in user_content
    assert "VoxScribe（语声成文）" in user_content
    assert prompts.EXTRA_MARKER in user_content
    assert "保留 emoji" in user_content
    # 正文仍然在 user prompt 里，且位于标记之后
    assert prompts.BODY_MARKER in user_content
    assert extract_body(user_content) == sample_transcript.text
    assert system_content(server.last_request.json) == prompts.system_prompt(
        "meeting", "zh"
    )


# ---------------------------------------------------------------------------
# 长文本分块
# ---------------------------------------------------------------------------
def test_long_text_chunks_merge_and_progress(server, sample_transcript):
    # 每段 180 字，单块上限 361 字 => 每块恰好 2 段，共 6 块（> 3）
    transcript = make_long_transcript(segments=12, segment_chars=180)
    options = RefineOptions(max_chunk_chars=361, chunk_overlap_chars=40)
    client = make_refiner(server)
    try:
        result, progress = run_refine(client, transcript, options)
    finally:
        client.close()

    chunks = result.chunks
    assert chunks == 6
    assert server.request_count == chunks

    bodies = [
        extract_body(last_user_content(request.json)) for request in server.requests
    ]
    # 分块调用次数、拼接结果完整且不重复
    assert len(bodies) == chunks
    assert all(0 < len(body) <= 361 for body in bodies)
    expected = "\n\n".join(bodies)
    assert result.text == expected
    assert "\n".join(bodies).replace("\n", "") == transcript.text.replace("\n", "")

    # 块间上文：第 2 块起带「上文」，且包含上一块结尾的 overlap 字符
    first_user = last_user_content(server.requests[0].json)
    assert prompts.CONTEXT_MARKER not in first_user
    for index in range(1, chunks):
        user_content = last_user_content(server.requests[index].json)
        assert prompts.CONTEXT_MARKER in user_content
        context_part = user_content.split(prompts.CONTEXT_MARKER, 1)[1]
        context_part = context_part.split(prompts.BODY_MARKER, 1)[0]
        assert "仅供衔接" in context_part
        assert bodies[index - 1][-40:] in context_part
        # 上下文没有混进待整理正文
        assert extract_body(user_content) == bodies[index]

    # 进度：0, 1/6, 2/6, ..., 5/6, 1.0
    ratios = [ratio for ratio, _ in progress]
    assert ratios == pytest.approx([index / chunks for index in range(chunks)] + [1.0])
    assert ratios[-1] == 1.0
    assert "正在整理第 6/6 段" in progress[chunks - 1][1]

    # usage 是每块累加
    assert result.usage.prompt_tokens == DEFAULT_USAGE["prompt_tokens"] * chunks
    assert result.usage.completion_tokens == DEFAULT_USAGE["completion_tokens"] * chunks
    assert result.usage.total_tokens == DEFAULT_USAGE["total_tokens"] * chunks


def test_split_transcript_unit():
    transcript = Transcript(
        segments=[
            Segment(0.0, 1.0, "甲" * 100),
            Segment(1.0, 2.0, "乙" * 100),
            Segment(2.0, 3.0, "丙" * 100),
        ]
    )
    chunks = split_transcript(transcript, max_chunk_chars=150, chunk_overlap_chars=20)
    assert [chunk.text for chunk in chunks] == ["甲" * 100, "乙" * 100, "丙" * 100]
    assert chunks[0].context == ""
    assert chunks[1].context == "甲" * 20
    assert chunks[2].context == "乙" * 20


def test_split_transcript_splits_oversized_segment():
    transcript = Transcript(segments=[Segment(0.0, 1.0, "字" * 250)])
    chunks = split_transcript(transcript, max_chunk_chars=100, chunk_overlap_chars=10)
    assert [len(chunk.text) for chunk in chunks] == [100, 100, 50]
    assert chunks[0].context == ""
    assert chunks[1].context == "字" * 10


def test_empty_transcript(server, refiner):
    result = refiner.refine(Transcript(segments=[]), RefineOptions())
    assert result.text == ""
    assert result.chunks == 0
    assert result.usage.total_tokens == 0
    assert server.request_count == 0


# ---------------------------------------------------------------------------
# 重试 / 错误处理
# ---------------------------------------------------------------------------
def test_retry_on_429_then_success(server, sample_transcript):
    server.fail_times(2, status=429, retry_after="0")
    client = make_refiner(server, max_retries=3)
    try:
        result = client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()

    assert result.text == sample_transcript.text
    assert result.chunks == 1
    assert server.request_count == 3
    # 失败响应不累计 usage，只统计成功的那一次
    assert result.usage.total_tokens == DEFAULT_USAGE["total_tokens"]


def test_retry_exhausted_5xx(server, sample_transcript):
    server.fail_times(5, status=503, retry_after="0")
    client = make_refiner(server, max_retries=2)
    try:
        with pytest.raises(RefineError) as excinfo:
            client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()

    message = str(excinfo.value)
    assert "503" in message
    assert "已重试 2 次" in message
    assert excinfo.value.user_hint
    assert server.request_count == 3


def test_retry_after_is_respected(server, sample_transcript):
    server.push_error(429, "Too Many Requests", retry_after="0.2")
    client = make_refiner(server, max_retries=1)
    started = time.monotonic()
    try:
        result = client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()
    elapsed = time.monotonic() - started
    assert result.text == sample_transcript.text
    assert server.request_count == 2
    assert elapsed >= 0.18


def test_exponential_backoff_without_retry_after(
    server, sample_transcript, monkeypatch
):
    monkeypatch.setattr("voxscribe.refine.RETRY_BASE_SECONDS", 0.05)
    server.push(MockResponse(429, error_response_body("Too Many Requests")))
    client = make_refiner(server, max_retries=1)
    started = time.monotonic()
    try:
        result = client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()
    elapsed = time.monotonic() - started
    assert result.text == sample_transcript.text
    assert server.request_count == 2
    assert elapsed >= 0.04


def test_auth_failure_raises_refine_error(server, refiner, sample_transcript):
    server.push_error(
        401,
        "Authentication Fails (invalid api key)",
        type_="authentication_error",
    )
    with pytest.raises(RefineError) as excinfo:
        refiner.refine(sample_transcript, RefineOptions())

    message = str(excinfo.value)
    hint = excinfo.value.user_hint
    assert "API Key" in message
    assert "无效" in message or "鉴权" in message
    assert "设置" in hint
    # 401 不重试
    assert server.request_count == 1


def test_insufficient_balance_402(server, refiner, sample_transcript):
    server.push_error(402, "Insufficient Balance", type_="insufficient_balance_error")
    with pytest.raises(RefineError) as excinfo:
        refiner.refine(sample_transcript, RefineOptions())
    assert "余额不足" in str(excinfo.value)
    assert excinfo.value.user_hint
    assert server.request_count == 1


@pytest.mark.parametrize(
    "response",
    [
        MockResponse(200, {"choices": []}),
        MockResponse(200, {"choices": [{"message": {"role": "assistant"}}]}),
        MockResponse(200, {"choices": [{"message": {"content": "   "}}]}),
        MockResponse(200, {"unexpected": True}),
        MockResponse(200, "not-a-json-body"),
    ],
    ids=["empty_choices", "missing_content", "blank_content", "missing_choices", "not_json"],
)
def test_malformed_response_raises_refine_error(server, sample_transcript, response):
    server.push(response)
    client = make_refiner(server)
    try:
        with pytest.raises(RefineError) as excinfo:
            client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()
    assert excinfo.value.user_hint
    assert server.request_count == 1


def test_connection_error_raises_refine_error(sample_transcript):
    # 取一个刚释放的本地端口，基本可以保证没有服务监听
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    finally:
        sock.close()

    client = DeepSeekRefiner(
        api_key=TEST_KEY,
        base_url=f"http://127.0.0.1:{port}",
        timeout=0.5,
        max_retries=0,
    )
    try:
        with pytest.raises(RefineError) as excinfo:
            client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()
    # Windows 上连接被拒绝有时会表现为 ConnectTimeout，两种提示都接受
    message = str(excinfo.value)
    assert "超时" in message or "无法连接" in message or "请求 DeepSeek 失败" in message
    assert excinfo.value.user_hint


# ---------------------------------------------------------------------------
# 取消
# ---------------------------------------------------------------------------
def test_cancel_before_start(server, refiner, sample_transcript):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        refiner.refine(sample_transcript, RefineOptions(), cancel=cancel)
    assert server.request_count == 0


def test_cancel_interrupts_retry_wait(server, sample_transcript):
    server.push_error(429, "Too Many Requests", retry_after="30")
    client = make_refiner(server, max_retries=2)
    cancel = threading.Event()
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(CancelledError):
            client.refine(sample_transcript, RefineOptions(), cancel=cancel)
    finally:
        timer.cancel()
        timer.join(timeout=1.0)
        elapsed = time.monotonic() - started
        client.close()
    # 如果没有用 cancel.wait 而是 time.sleep(30)，这里会超时
    assert elapsed < 10.0
    assert server.request_count == 1


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------
def test_test_connection_success(server, refiner):
    reply = refiner.test_connection()
    assert reply == prompts.CONNECTION_USER_PROMPT

    payload = server.last_request.json
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == prompts.CONNECTION_MAX_TOKENS
    assert last_user_content(payload) == prompts.CONNECTION_USER_PROMPT
    assert system_content(payload) == prompts.CONNECTION_SYSTEM_PROMPT


def test_test_connection_failure(server, refiner):
    server.push_error(401, "Authentication Fails", type_="authentication_error")
    with pytest.raises(RefineError) as excinfo:
        refiner.test_connection()
    assert "API Key" in str(excinfo.value)
    assert excinfo.value.user_hint
    assert server.request_count == 1


# ---------------------------------------------------------------------------
# 其它
# ---------------------------------------------------------------------------
def test_delayed_response_is_handled(server, sample_transcript):
    server.delay = 0.15
    client = make_refiner(server, timeout=5.0)
    started = time.monotonic()
    try:
        result = client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()
    elapsed = time.monotonic() - started
    assert result.text == sample_transcript.text
    assert elapsed >= 0.12


def test_api_key_never_logged_or_leaked(server, sample_transcript, caplog):
    secret = "sk-super-secret-abcdef123456"
    caplog.set_level(logging.DEBUG)
    client = DeepSeekRefiner(api_key=secret, base_url=server.url)
    try:
        client.refine(sample_transcript, RefineOptions())
        # 即使服务端在错误信息里回显 key，也不能出现在异常/日志中
        server.push_error(401, f"invalid key: {secret}")
        with pytest.raises(RefineError) as excinfo:
            client.refine(sample_transcript, RefineOptions())
    finally:
        client.close()
    assert secret not in str(excinfo.value)
    assert secret not in caplog.text


def test_missing_api_key_raises(server, sample_transcript):
    client = DeepSeekRefiner(api_key="", base_url=server.url)
    try:
        with pytest.raises(RefineError) as excinfo:
            client.refine(sample_transcript, RefineOptions())
        with pytest.raises(RefineError):
            client.test_connection()
    finally:
        client.close()
    assert "API Key" in str(excinfo.value)
    assert server.request_count == 0
