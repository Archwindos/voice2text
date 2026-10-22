"""词语自动学习核心模块 :mod:`voxscribe.glossary` 的单元测试。

全部是纯函数测试：不访问网络、不启动 GUI、不写项目 ``data/`` 目录
（存储相关用例一律使用 ``tmp_path`` 并通过 ``VOXSCRIBE_CONFIG`` 隔离）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from voxscribe import glossary
from voxscribe.glossary import (
    ExtractionResult,
    GlossaryEntry,
    LearnCandidate,
    apply_replacements,
    build_prompt_glossary,
    extract_candidates,
    load_store,
    merge_candidates,
    parse_manual_glossary,
    save_store,
    store_path,
    undo_candidates,
)

LOGGER_NAME = "voxscribe.glossary"


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def _json_events(caplog: pytest.LogCaptureFixture) -> list[dict]:
    """把 caplog 里的 glossary 日志逐行解析成 dict，并校验「单行 JSON」。"""
    events = []
    for record in caplog.records:
        if record.name != LOGGER_NAME:
            continue
        message = record.getMessage()
        assert "\n" not in message, "日志必须是单行 JSON"
        events.append(json.loads(message))
    return events


def _event(events: list[dict], kind: str) -> dict:
    matched = [item for item in events if item.get("event") == kind]
    assert matched, f"缺少 {kind} 事件：{[e.get('event') for e in events]}"
    return matched[-1]


def _reasons(result: ExtractionResult) -> list[str]:
    return [item.get("reason", "") for item in result.filtered]


def _candidate(wrong: str, correct: str, confidence: float = 0.75) -> LearnCandidate:
    return LearnCandidate(
        wrong=wrong,
        correct=correct,
        kind="replace",
        confidence=confidence,
        context="",
    )


# ---------------------------------------------------------------------------
# 1. 抽取：替换 / 删除
# ---------------------------------------------------------------------------
def test_extract_replace_word():
    before = "我们打开阴轨设置"
    after = "我们打开音轨设置"

    result = extract_candidates(before, after)

    assert isinstance(result, ExtractionResult)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.wrong == "阴轨"
    assert candidate.correct == "音轨"
    assert candidate.kind == "replace"
    assert candidate.confidence == pytest.approx(0.75)
    assert candidate.context == "我们打开…设置"

    # 统计字段
    assert result.baseline_chars == len(before)
    assert result.edited_chars == len(after)
    assert result.changed_chars == 1
    assert 0 < result.change_ratio <= 0.3
    assert result.opcode_count >= 1
    assert result.elapsed_ms >= 0.0
    assert result.filtered == []


def test_extract_english_word_is_expanded_to_whole_word():
    """英文按空白分词取整，且多个字符级片段扩张成同一对词语时只保留一条。"""
    result = extract_candidates("I use the audio track", "I use the video track")

    assert [(c.wrong, c.correct) for c in result.candidates] == [("audio", "video")]


def test_extract_expands_single_char_on_left_side():
    """单字差异也能按常见词扩张到左侧（登入→登录），不是只认识「阴轨」。"""
    result = extract_candidates("请先登入账号再继续", "请先登录账号再继续")

    assert [(c.wrong, c.correct) for c in result.candidates] == [("登入", "登录")]


def test_extract_expands_single_char_on_left_for_common_word():
    result = extract_candidates("这个网络连接有问题", "这个网路连接有问题")

    assert [(c.wrong, c.correct) for c in result.candidates] == [("网络", "网路")]


def test_extract_delete_word():
    result = extract_candidates("请打开阴轨设置", "请打开设置")

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.wrong == "阴轨"
    assert candidate.correct == ""
    assert candidate.kind == "delete"
    assert candidate.confidence == pytest.approx(0.5)
    assert candidate.context == "请打开…设置"
    assert result.changed_chars == 2
    # 删除类不会出现在「替换」候选里
    assert [c.kind for c in result.candidates] == ["delete"]


# ---------------------------------------------------------------------------
# 2. 过滤：标点 / 繁简 / 大改 / 单字 / 停用词 / 超长
# ---------------------------------------------------------------------------
def test_pure_punctuation_is_not_learned():
    # 只追加标点（insert）
    inserted = extract_candidates("我们今天开始吧", "我们今天开始吧！")
    assert inserted.candidates == []
    assert any("插入" in reason for reason in _reasons(inserted))

    # 只替换标点（replace）
    replaced = extract_candidates("今天很好。", "今天很好！")
    assert replaced.candidates == []
    assert any("标点" in reason for reason in _reasons(replaced))


def test_traditional_simplified_difference_is_filtered():
    before = "我们今天讨论的问题就是這個項目非常好用而且簡單"
    after = "我们今天讨论的问题就是这个项目非常好用而且简单"

    result = extract_candidates(before, after)

    assert result.candidates == []
    assert _reasons(result)
    assert all("繁简" in reason for reason in _reasons(result))


def test_big_rewrite_is_rejected():
    before = "我们今天下午三点在公司开会讨论项目进度"
    after = "明天早上九点去健身房锻炼身体然后吃饭"

    result = extract_candidates(before, after)

    assert result.candidates == []
    assert any("大改" in reason for reason in _reasons(result))
    assert result.change_ratio > 0.3
    assert result.filtered[0]["snippet"] == ""  # 大改不记录正文片段


def test_single_char_replace_is_rejected():
    result = extract_candidates("这个东西的很好用", "这个东西得很好用")

    assert result.candidates == []
    assert any("单字" in reason for reason in _reasons(result))


def test_stopword_replace_is_rejected():
    result = extract_candidates("然后我们就开始讨论项目进度", "接着我们就开始讨论项目进度")

    assert result.candidates == []
    assert any("停用词" in reason for reason in _reasons(result))


def test_oversized_candidate_is_rejected():
    prefix = "开头的说明文字" * 4
    suffix = "结尾的说明文字" * 3
    old_block = "甲乙丙丁戊己庚辛壬癸子丑寅"  # 13 字，超过 max_len=12
    new_block = "春夏秋冬东南西北上下左右前后"
    before = prefix + old_block + suffix
    after = prefix + new_block + suffix

    result = extract_candidates(before, after)

    assert result.changed_chars >= 13
    assert result.change_ratio <= 0.3
    assert result.candidates == []
    assert any("超长" in reason for reason in _reasons(result))


def test_candidate_limit():
    pairs = [
        ("甲乙", "春夏"),
        ("丙丁", "秋冬"),
        ("戊己", "东南"),
        ("庚辛", "西北"),
        ("壬癸", "上下"),
        ("子丑", "左右"),
        ("寅卯", "前后"),
    ]
    filler = "，我们继续处理项目内容并检查"
    before = "开头内容是" + filler.join(old for old, _ in pairs) + "，结束"
    after = "开头内容是" + filler.join(new for _, new in pairs) + "，结束"

    result = extract_candidates(before, after)

    assert result.changed_chars <= 40
    assert result.change_ratio <= 0.3
    assert len(result.candidates) == 5
    assert [c.correct for c in result.candidates] == [new for _, new in pairs][:5]
    assert any("上限" in reason for reason in _reasons(result))


def test_identity_and_empty_texts_are_safe():
    same = extract_candidates("同样的文本", "同样的文本")
    assert same.candidates == [] and same.filtered == []
    assert same.changed_chars == 0 and same.opcode_count == 0

    empty_after = extract_candidates("用户清空了全部内容", "")
    assert empty_after.candidates == []
    assert any("为空" in reason for reason in _reasons(empty_after))

    empty_before = extract_candidates("", "凭空多出来的内容")
    assert empty_before.candidates == []
    assert any("为空" in reason for reason in _reasons(empty_before))


# ---------------------------------------------------------------------------
# 3. 合并：手动优先 / 冲突 / 重复累加 / 删除规则
# ---------------------------------------------------------------------------
def test_new_replace_candidate_first_time_goes_pending():
    entries, pending, conflicts = merge_candidates([], [_candidate("阴轨", "音轨")])

    assert len(entries) == 1
    entry = entries[0]
    assert entry.source == "learned"
    assert entry.enabled is False
    assert entry.count == 1
    assert entry.confidence == pytest.approx(0.75)
    assert entry.created_at and entry.updated_at
    assert [item.wrong for item in pending] == ["阴轨"]
    assert conflicts == []


def test_repeat_accumulates_and_auto_enables():
    candidate = _candidate("阴轨", "音轨")

    entries, pending, _ = merge_candidates([], [candidate])
    assert entries[0].enabled is False and len(pending) == 1

    entries, pending, conflicts = merge_candidates(entries, [candidate])
    entry = entries[0]
    assert entry.count == 2
    assert entry.confidence == pytest.approx(0.85)
    assert entry.enabled is True  # 第二次出现达到 0.85 自动应用
    assert pending == []
    assert conflicts == []


def test_auto_apply_threshold_is_configurable():
    high = LearnCandidate("阴轨", "音轨", "replace", 0.9, "")
    entries, pending, _ = merge_candidates([], [high])
    assert entries[0].enabled is True
    assert pending == []

    entries, pending, _ = merge_candidates(
        [], [high], auto_apply_threshold=0.95
    )
    assert entries[0].enabled is False
    assert len(pending) == 1


def test_manual_entries_have_priority():
    manual = parse_manual_glossary("音轨：阴轨")

    entries, pending, conflicts = merge_candidates(
        [],
        [_candidate("阴轨", "银轨")],
        manual_entries=manual,
    )

    assert len(entries) == 1
    assert entries[0].source == "manual"
    assert entries[0].correct == "音轨"
    assert entries[0].enabled is True
    # 学习候选没有被静默覆盖，而是进入冲突区
    assert [item.correct for item in conflicts] == ["银轨"]
    assert pending == []


def test_manual_same_pair_accumulates_count():
    manual = parse_manual_glossary("音轨：阴轨")

    entries, pending, conflicts = merge_candidates(
        [], [_candidate("阴轨", "音轨")], manual_entries=manual
    )

    assert len(entries) == 1
    assert entries[0].source == "manual"
    assert entries[0].count == 2
    assert entries[0].confidence == pytest.approx(1.0)
    assert pending == [] and conflicts == []


def test_provided_manual_entries_are_authoritative():
    stored = [
        GlossaryEntry(wrong="旧词", correct="旧写法", source="manual", enabled=True)
    ]

    # 传了（哪怕是空列表）本次手动术语表，就以它为准：设置页删掉的术语不会复活
    entries, pending, conflicts = merge_candidates(stored, [], manual_entries=[])
    assert entries == [] and pending == [] and conflicts == []

    # 不传 manual_entries 时沿用存储里的手动条目
    entries, _, _ = merge_candidates(stored, [])
    assert [item.wrong for item in entries] == ["旧词"]


def test_same_wrong_different_correct_conflicts():
    first = _candidate("阴轨", "音轨")
    entries, _, _ = merge_candidates([], [first])

    entries, pending, conflicts = merge_candidates(
        entries, [_candidate("阴轨", "银轨")]
    )

    # 保留原条目，不静默覆盖
    assert len(entries) == 1
    assert entries[0].correct == "音轨"
    assert entries[0].count == 1
    assert len(conflicts) == 1
    assert conflicts[0].correct == "银轨"
    assert conflicts[0].enabled is False
    # 原待确认条目仍然待确认
    assert [item.correct for item in pending] == ["音轨"]


def test_delete_candidate_never_auto_enabled():
    candidate = LearnCandidate("阴轨", "", "delete", 0.5, "")

    entries, pending, _ = merge_candidates([], [candidate])
    assert entries[0].kind == "delete"
    assert entries[0].enabled is False

    # 重复出现 + 很低的阈值也不会自动启用删除规则
    entries, pending, _ = merge_candidates(
        entries, [candidate], auto_apply_threshold=0.01
    )
    assert entries[0].count == 2
    assert entries[0].enabled is False
    assert [item.wrong for item in pending] == ["阴轨"]


def test_replace_wins_over_delete_rule():
    first = LearnCandidate("阴轨", "", "delete", 0.5, "")
    entries, pending, _ = merge_candidates([], [first])
    assert entries[0].kind == "delete"

    entries, pending, conflicts = merge_candidates(entries, [_candidate("阴轨", "音轨")])

    assert len(entries) == 1
    assert entries[0].kind == "replace"
    assert entries[0].correct == "音轨"
    assert [item.kind for item in conflicts] == ["delete"]
    assert [item.wrong for item in pending] == ["阴轨"]


def test_merge_does_not_mutate_inputs():
    manual = parse_manual_glossary("音轨：阴轨")
    learned = GlossaryEntry(
        wrong="登入", correct="登录", source="learned", confidence=0.75, count=1
    )
    candidate = _candidate("阴轨", "音轨")

    merge_candidates([learned], [candidate], manual_entries=manual)

    assert manual[0].count == 1
    assert learned.count == 1
    assert candidate.wrong == "阴轨"


def test_merge_with_empty_input_is_safe():
    entries, pending, conflicts = merge_candidates([], [])
    assert entries == [] and pending == [] and conflicts == []


def test_merge_respects_max_entries():
    candidates = [_candidate(f"错{i}", f"对{i}", confidence=0.9) for i in range(3)]

    entries, pending, conflicts = merge_candidates(
        [], candidates, max_entries=2
    )

    # 只保留 2 条启用条目，被淘汰的那条进入冲突区
    assert len(entries) == 2
    assert sum(1 for item in entries if item.enabled) == 2
    assert pending == []
    assert len(conflicts) == 1
    assert conflicts[0].wrong == "错0"
    assert conflicts[0].enabled is False


# ---------------------------------------------------------------------------
# 4. 手动术语表解析 / prompt 术语表
# ---------------------------------------------------------------------------
def test_parse_manual_glossary_formats():
    text = """
    # 这是一行注释
    VoxScribe（语声成文）：项目名称
    音轨：阴轨、银轨
    DeepSeek
    音轨：阴轨
    // 另一种注释
    ：没有正确写法
    """

    entries = parse_manual_glossary(text)

    assert [(item.correct, item.wrong) for item in entries] == [
        ("VoxScribe（语声成文）", "项目名称"),
        ("音轨", "阴轨"),
        ("音轨", "银轨"),
        ("DeepSeek", ""),
    ]
    for item in entries:
        assert item.source == "manual"
        assert item.enabled is True
        assert item.kind == "replace"
        assert item.confidence == pytest.approx(1.0)
        assert item.count == 1
        assert item.created_at and item.updated_at


def test_parse_manual_glossary_empty():
    assert parse_manual_glossary("") == []
    assert parse_manual_glossary("   \n# 只有注释\n") == []


def test_build_prompt_glossary():
    entries = [
        GlossaryEntry(wrong="阴轨", correct="音轨", source="manual", enabled=True),
        GlossaryEntry(wrong="登入", correct="登录", source="learned", enabled=True),
        GlossaryEntry(wrong="那个", correct="", kind="delete", enabled=True),
        GlossaryEntry(wrong="测试", correct="检验", source="learned", enabled=False),
        GlossaryEntry(wrong="", correct="VoxScribe", source="manual", enabled=True),
        GlossaryEntry(wrong="阴轨", correct="银轨", source="learned", enabled=True),
    ]

    text = build_prompt_glossary(entries)

    # 只含 enabled 的 replace 条目，按 wrong 去重，格式「正确：错误」
    assert text.splitlines() == ["音轨：阴轨", "登录：登入", "VoxScribe"]
    assert build_prompt_glossary([]) == ""


# ---------------------------------------------------------------------------
# 5. 本地应用替换：链式映射 / 环 / 删除 / 英文词边界
# ---------------------------------------------------------------------------
def test_apply_replacements_basic():
    entries = [
        GlossaryEntry(wrong="阴轨", correct="音轨", enabled=True),
        GlossaryEntry(wrong="阴鬼", correct="", kind="delete", enabled=True),
        GlossaryEntry(wrong="测试", correct="检验", enabled=False),
    ]

    assert (
        apply_replacements("打开阴轨和阴鬼，测试没问题", entries)
        == "打开音轨和，测试没问题"
    )
    assert apply_replacements("", entries) == ""
    assert apply_replacements("没有任何规则命中", []) == "没有任何规则命中"


def test_apply_replacements_chain_mapping():
    entries = [
        GlossaryEntry(wrong="甲", correct="乙", enabled=True),
        GlossaryEntry(wrong="乙", correct="丙", enabled=True),
    ]

    # A→B、B→C 做一次传递闭包，最终都变成 C
    assert apply_replacements("甲和乙", entries) == "丙和丙"


def test_apply_replacements_cycle_is_skipped():
    entries = [
        GlossaryEntry(wrong="甲", correct="乙", enabled=True),
        GlossaryEntry(wrong="乙", correct="甲", enabled=True),
    ]

    # 环上的规则全部跳过，也不会死循环
    assert apply_replacements("甲和乙", entries) == "甲和乙"


def test_apply_replacements_ascii_word_boundary():
    entries = [GlossaryEntry(wrong="cat", correct="dog", enabled=True)]

    assert apply_replacements("category and cat", entries) == "category and dog"


# ---------------------------------------------------------------------------
# 6. 存储：原子读写 / 损坏恢复 / 默认路径
# ---------------------------------------------------------------------------
def test_store_roundtrip_and_atomic_write(tmp_path: Path):
    target = tmp_path / "data" / "glossary.json"
    entries = [
        GlossaryEntry(
            wrong="阴轨",
            correct="音轨",
            source="learned",
            enabled=True,
            confidence=0.9,
            count=2,
            created_at="2026-10-22T21:00:00+08:00",
            updated_at="2026-10-23T00:00:00+08:00",
        ),
        GlossaryEntry(
            wrong="那个",
            correct="",
            kind="delete",
            source="learned",
            enabled=False,
            confidence=0.4,
        ),
    ]

    saved = save_store(entries, target)
    assert saved == target
    assert target.exists()

    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["updated_at"]
    assert len(raw["entries"]) == 2

    loaded = load_store(target)
    assert [(item.wrong, item.correct, item.kind, item.source) for item in loaded] == [
        ("阴轨", "音轨", "replace", "learned"),
        ("那个", "", "delete", "learned"),
    ]
    assert loaded[0].count == 2
    assert loaded[0].confidence == pytest.approx(0.9)
    assert loaded[1].enabled is False

    # 原子写：目录里不能残留临时文件
    leftovers = [p.name for p in target.parent.iterdir() if p.name.startswith(".glossary-")]
    assert leftovers == []

    # 第二次保存保留 .bak
    save_store(loaded[:1], target)
    assert target.with_suffix(".json.bak").exists()
    assert [item.wrong for item in load_store(target)] == ["阴轨"]

    # 不存在时返回空列表，不创建文件
    missing = tmp_path / "no-such" / "glossary.json"
    assert load_store(missing) == []
    assert not missing.exists()


def test_store_corrupt_file_is_backed_up(tmp_path: Path):
    target = tmp_path / "glossary.json"
    target.write_text("{这不是合法 JSON", encoding="utf-8")

    assert load_store(target) == []
    assert not target.exists()
    assert target.with_suffix(".json.broken").exists()


def test_store_path_follows_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config_file = tmp_path / "appdata" / "config.json"
    monkeypatch.setenv("VOXSCRIBE_CONFIG", str(config_file))

    assert store_path() == tmp_path / "appdata" / "glossary.json"
    # 纯路径计算，不应创建目录/文件
    assert not config_file.parent.exists()
    assert load_store() == []
    assert not config_file.parent.exists()


# ---------------------------------------------------------------------------
# 7. 详细日志（caplog）：事件存在、字段齐全、单行 JSON、隐私
# ---------------------------------------------------------------------------
def test_extract_log_event_schema_and_privacy(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    before = "我们打开阴轨设置"
    after = "我们打开音轨设置"

    extract_candidates(before, after)

    events = _json_events(caplog)
    event = _event(events, "extract")

    for key in (
        "event",
        "timestamp",
        "baseline_chars",
        "edited_chars",
        "changed_chars",
        "change_ratio",
        "opcode_count",
        "candidates",
        "filtered",
        "accepted",
        "pending",
        "conflicts",
        "elapsed_ms",
    ):
        assert key in event, f"extract 事件缺少字段：{key}"
    assert datetime.fromisoformat(event["timestamp"]).tzinfo is not None
    assert event["baseline_chars"] == len(before)
    assert event["edited_chars"] == len(after)
    assert event["change_ratio"] == pytest.approx(0.125)
    assert event["candidates"][0] == {
        "wrong": "阴轨",
        "correct": "音轨",
        "kind": "replace",
        "confidence": 0.75,
        "context": "我们打开…设置",
    }
    assert event["filtered"] == []
    assert event["accepted"] == 0
    assert event["elapsed_ms"] >= 0.0

    # 隐私：不记录完整 before/after 原文
    assert before not in caplog.text
    assert after not in caplog.text


def test_filter_debug_log_has_reason(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)

    extract_candidates("这个东西的很好用", "这个东西得很好用")

    events = _json_events(caplog)
    filters = [item for item in events if item["event"] == "filter"]
    assert any("单字" in item["reason"] for item in filters)
    assert any(item["snippet"] for item in filters)
    # 片段长度受控，且不含被比较的整段原文
    assert all(len(item["snippet"]) <= 20 for item in filters)
    assert "这个东西的很好用" not in caplog.text


def test_merge_and_apply_log_events(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    candidate = _candidate("阴轨", "音轨")
    merge_candidates([], [candidate])
    apply_replacements(
        "请打开阴轨和音轨设置。",
        [GlossaryEntry(wrong="阴轨", correct="音轨", enabled=True)],
    )

    events = _json_events(caplog)
    merge_event = _event(events, "merge")
    assert merge_event["candidates"][0]["wrong"] == "阴轨"
    assert merge_event["accepted"] == 0
    assert merge_event["pending"] == 1
    assert merge_event["conflicts"] == 0
    assert merge_event["entries_after"] == 1

    apply_event = _event(events, "apply")
    assert apply_event["baseline_chars"] == len("请打开阴轨和音轨设置。")
    assert apply_event["edited_chars"] == len("请打开音轨和音轨设置。")
    assert apply_event["replacements"] == 1
    assert apply_event["rules_used"] == 1
    # 隐私：应用日志同样不记录正文
    assert "请打开阴轨和音轨设置。" not in caplog.text


def test_conflict_log_event(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    entries, _, _ = merge_candidates([], [_candidate("阴轨", "音轨")])

    merge_candidates(entries, [_candidate("阴轨", "银轨")])

    events = _json_events(caplog)
    conflicts = [item for item in events if item["event"] == "conflict"]
    assert any(
        item.get("wrong") == "阴轨"
        and item.get("kept_correct") == "音轨"
        and item.get("rejected_correct") == "银轨"
        for item in conflicts
    )


def test_undo_candidates(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    entries = [
        GlossaryEntry(wrong="阴轨", correct="音轨", source="learned", enabled=True),
        GlossaryEntry(wrong="登入", correct="登录", source="learned", enabled=True),
        GlossaryEntry(wrong="阴轨", correct="音轨", source="manual", enabled=True),
    ]

    kept = undo_candidates(entries, [_candidate("阴轨", "音轨")])

    assert {(item.wrong, item.source) for item in kept} == {
        ("登入", "learned"),
        ("阴轨", "manual"),  # 手动条目不会被撤销
    }
    undo_event = _event(_json_events(caplog), "undo")
    assert undo_event["undone"] == 1
    assert undo_event["entries_before"] == 3
    assert undo_event["entries_after"] == 2


# ---------------------------------------------------------------------------
# 8. 数据结构与端到端（抽取 → 合并 → 应用）
# ---------------------------------------------------------------------------
def test_entry_serialization_roundtrip():
    entry = GlossaryEntry(
        wrong="阴轨",
        correct="音轨",
        kind="replace",
        source="learned",
        enabled=True,
        confidence=0.85,
        count=3,
        created_at="2026-10-22T21:00:00+08:00",
        updated_at="2026-10-23T00:00:00+08:00",
    )

    restored = GlossaryEntry.from_dict(entry.to_dict())

    assert restored == entry
    # 脏数据降级：未知 kind/source 回退到安全默认值
    dirty = GlossaryEntry.from_dict(
        {"wrong": " 阴轨 ", "correct": "音轨", "kind": "???", "source": "???", "count": 0}
    )
    assert dirty.kind == "replace"
    assert dirty.source == "learned"
    assert dirty.count == 1
    assert dirty.wrong == "阴轨"
    with pytest.raises(ValueError):
        GlossaryEntry.from_dict("不是字典")  # type: ignore[arg-type]


def test_end_to_end_extract_merge_apply():
    before = "我们打开阴轨设置"
    after = "我们打开音轨设置"

    result = extract_candidates(before, after)
    assert result.candidates

    entries, pending, _ = merge_candidates([], result.candidates)
    assert len(pending) == 1

    # 用户点「采纳」后启用，再走一次真实应用
    pending[0].enabled = True
    assert apply_replacements("请打开阴轨设置", entries) == "请打开音轨设置"
    assert build_prompt_glossary(entries) == "音轨：阴轨"
