"""单实例控制（Windows 命名互斥量 + 激活事件）测试。"""

from __future__ import annotations

import sys
import uuid

import pytest

from voxscribe.instance import SingleInstance

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="单实例互斥量测试仅适用于 Windows"
)


def _names():
    token = uuid.uuid4().hex
    return (
        rf"Local\VoxScribe.Test.Mutex.{token}",
        rf"Local\VoxScribe.Test.Event.{token}",
    )


def test_first_instance_acquires_and_duplicate_is_rejected() -> None:
    mutex_name, event_name = _names()
    primary = SingleInstance(mutex_name=mutex_name, event_name=event_name)
    try:
        assert primary.acquire() is True
        assert primary.is_primary is True

        secondary = SingleInstance(mutex_name=mutex_name, event_name=event_name)
        try:
            assert secondary.acquire() is False
            assert secondary.is_primary is False
        finally:
            secondary.close()
    finally:
        primary.close()


def test_second_instance_requests_activation() -> None:
    mutex_name, event_name = _names()
    primary = SingleInstance(mutex_name=mutex_name, event_name=event_name)
    try:
        assert primary.acquire() is True
        # 没有请求时不应触发
        assert primary.consume_activation_request() is False

        secondary = SingleInstance(mutex_name=mutex_name, event_name=event_name)
        try:
            assert secondary.acquire() is False
            assert secondary.request_activation() is True
        finally:
            secondary.close()

        assert primary.consume_activation_request() is True
        # auto-reset 事件：消费一次后恢复未触发
        assert primary.consume_activation_request() is False
    finally:
        primary.close()


def test_mutex_released_after_close() -> None:
    mutex_name, event_name = _names()
    primary = SingleInstance(mutex_name=mutex_name, event_name=event_name)
    assert primary.acquire() is True
    primary.close()

    again = SingleInstance(mutex_name=mutex_name, event_name=event_name)
    try:
        assert again.acquire() is True
    finally:
        again.close()


def test_request_activation_without_primary_is_safe() -> None:
    mutex_name, event_name = _names()
    instance = SingleInstance(
        mutex_name=mutex_name,
        event_name=event_name,
        window_title="VoxScribe-Test-NoSuchWindow",
    )
    try:
        assert instance.request_activation() is False
    finally:
        instance.close()
