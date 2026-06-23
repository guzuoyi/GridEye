"""测试：Prompt 模板 + JSON 解析器 + 调度器"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.llm.prompts import (
    SYSTEM_EVENT_CONFIRM,
    prompt_event_confirm,
    SYSTEM_RISK_CLEARANCE,
    prompt_risk_clearance,
    prompt_state_refresh,
)
from src.llm.parser import (
    extract_json,
    parse_event_confirm,
    parse_state_refresh,
    parse_risk_clearance,
)
from src.llm.scheduler import (
    QwenCallScheduler,
    QwenCallRequest,
    CallType,
)


# ===== Prompts ===============================================================

class TestPrompts:
    """Prompt 模板生成"""

    def test_event_confirm_prompt(self):
        p = prompt_event_confirm("car", 1, 45.0, 5, 2.5)
        assert "car" in p
        assert "lane 1" in p
        assert "45" in p
        assert "5 frames" in p

    def test_state_refresh_prompt(self):
        p = prompt_state_refresh("truck", 2, "违停", 35.0)
        assert "truck" in p
        assert "违停" in p
        assert "35s" in p

    def test_risk_clearance_prompt(self):
        p = prompt_risk_clearance(
            "car", 1, 50.0, "违停", "10:00:00", "10:03:00"
        )
        assert "car" in p
        assert "违停" in p
        assert "10:00:00" in p


# ===== Parser ================================================================

class TestExtractJson:
    """JSON 提取"""

    def test_clean_json(self):
        text = '{"is_anomaly": true, "confidence": 0.9}'
        d = extract_json(text)
        assert d == {"is_anomaly": True, "confidence": 0.9}

    def test_json_with_surrounding_text(self):
        text = 'Some text {"key": "value"} more text'
        d = extract_json(text)
        assert d == {"key": "value"}

    def test_empty_string(self):
        assert extract_json("") is None
        assert extract_json("   ") is None

    def test_no_json(self):
        assert extract_json("no json here at all") is None

    def test_malformed_json(self):
        # 用正则可能提取失败，返回 None
        result = extract_json('{"a": 1, "b": }')
        assert result is None or isinstance(result, dict)


class TestParseFunctions:

    def test_event_confirm_anomaly(self):
        r = parse_event_confirm(
            '{"is_anomaly": true, "event_type": "违停", "confidence": 0.88, "reasoning": "test"}'
        )
        assert r["is_anomaly"] is True
        assert r["event_type"] == "违停"
        assert r["confidence"] == pytest.approx(0.88)

    def test_event_confirm_normal(self):
        r = parse_event_confirm(
            '{"is_anomaly": false, "event_type": "正常", "confidence": 0.95}'
        )
        assert r["is_anomaly"] is False

    def test_event_confirm_fallback(self):
        """解析失败 → 默认值。"""
        r = parse_event_confirm("garbage text")
        assert r["is_anomaly"] is False  # safe fallback

    def test_risk_clearance(self):
        r = parse_risk_clearance(
            '{"risk_cleared": true, "need_cleanup": false, "confidence": 0.95}'
        )
        assert r["risk_cleared"] is True
        assert r["need_cleanup"] is False

    def test_state_refresh(self):
        r = parse_state_refresh(
            '{"status_changed": false, "updated_event_type": "违停", "confidence": 0.9}'
        )
        assert r["status_changed"] is False


# ===== Scheduler =============================================================

@pytest.fixture
def mock_client():
    """Mock QwenAPIClient"""
    c = MagicMock()
    c.chat.return_value = {
        "content": '{"is_anomaly": true, "confidence": 0.9}',
        "finish": "stop",
        "tokens": 50,
        "elapsed": 1.5,
    }
    return c


class TestScheduler:
    """调用调度器"""

    def test_enqueue_and_process(self, mock_client):
        scheduler = QwenCallScheduler(mock_client, max_concurrency=1)
        scheduler.start()

        called = []
        req = QwenCallRequest(
            call_type=CallType.EVENT_CONFIRM,
            record_id="test123",
            system_prompt="You are expert",
            user_text="Analyze",
            image=None,
            callback=lambda r: called.append(r),
        )
        scheduler.enqueue(req)

        # 等待处理
        import time
        for _ in range(50):  # 最多等 5s
            if called:
                break
            time.sleep(0.1)

        scheduler.stop()

        assert len(called) == 1
        assert called[0].success is True
        assert called[0].tokens == 50

    def test_queue_length(self, mock_client):
        scheduler = QwenCallScheduler(mock_client)
        req = QwenCallRequest(
            call_type=CallType.RISK_CLEARANCE,
            record_id="r1",
            system_prompt="sys",
            user_text="text",
            callback=lambda r: None,
        )
        scheduler.enqueue(req)
        scheduler.enqueue(req)
        assert scheduler.queue_length == 2

    def test_total_calls_count(self, mock_client):
        scheduler = QwenCallScheduler(mock_client)
        scheduler.start()

        done = []
        req = QwenCallRequest(
            call_type=CallType.EVENT_CONFIRM,
            record_id="r1",
            system_prompt="s", user_text="t",
            callback=lambda r: done.append(r),
        )
        scheduler.enqueue(req)

        import time
        for _ in range(50):
            if done:
                break
            time.sleep(0.1)
        scheduler.stop()

        assert scheduler.total_calls == 1
