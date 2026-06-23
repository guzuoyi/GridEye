"""测试：PositionCache + StateMachine —— 位置匹配、状态迁移、过期清理"""

import pytest
import time
import math
import numpy as np

from src.cache.cache import (
    State,
    QwenResult,
    PositionRecord,
    PositionCache,
    StateMachine,
)


# ---- helpers ----------------------------------------------------------------

def make_det(cls_name, conf, bbox):
    """构造模拟 Detection。"""
    from src.detector.models import Detection
    return Detection(cls_name, conf, bbox)


def make_record(state=State.IDLE, dwell=1, last_seen=0.0):
    """构造 PositionRecord，填充 dwell_history。"""
    rec = PositionRecord(
        class_name="car",
        center_x=150.0,
        center_y=130.0,
        state=state,
        consecutive_frames=dwell,
        last_seen_time=last_seen,
        state_timestamps={state: 0.0},
    )
    # 填充 dwell_history：dwell 个 True
    from collections import deque
    rec.dwell_history = deque([True] * min(dwell, 10), maxlen=10)
    # 填充 position_history：模拟静止（同一坐标）
    rec.position_history = deque(
        [(150.0, 130.0)] * min(dwell, 10), maxlen=10
    )
    return rec


# ===== PositionCache =========================================================

class TestPositionCache:
    """位置缓存匹配测试"""

    @pytest.fixture
    def cache(self):
        return PositionCache(match_distance_px=40)

    def test_create_new_record(self, cache):
        """首次检测 → 创建新记录。"""
        dets = [make_det("car", 0.9, (100, 100, 200, 160))]
        records = cache.match_or_create(dets, now=0.0)
        assert len(records) == 1
        assert records[0].class_name == "car"
        assert records[0].consecutive_frames == 1

    def test_match_existing(self, cache):
        """同位置同类别 → 匹配并更新。"""
        dets = [make_det("car", 0.9, (100, 100, 200, 160))]
        cache.match_or_create(dets, now=0.0)
        records = cache.match_or_create(dets, now=0.5)
        assert len(records) == 1
        assert records[0].consecutive_frames == 2

    def test_no_match_different_class(self, cache):
        """不同类别 → 不匹配。"""
        cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=0)
        records = cache.match_or_create([make_det("person", 0.9, (100, 100, 200, 160))], now=0.5)
        assert len(records) == 2  # 新建 person + 保留 car

    def test_no_match_distant(self, cache):
        """距离超过 40px → 不匹配。"""
        cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=0)
        # 中心点偏移 50px
        records = cache.match_or_create([make_det("car", 0.9, (150, 100, 250, 160))], now=0.5)
        assert len(records) == 2  # 新建记录

    def test_choose_closest_on_multiple(self, cache):
        """多个候选 → 选距离最近的。"""
        cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=0)
        cache.match_or_create([make_det("car", 0.9, (500, 100, 600, 160))], now=0)
        assert len(cache._records) == 2
        # 新检测靠近第一个
        records = cache.match_or_create([make_det("car", 0.9, (105, 100, 205, 160))], now=0.5)
        assert len(records) == 2
        assert records[0].consecutive_frames == 2  # 匹配到第一个

    def test_cleanup_idle_expired(self, cache):
        """IDLE 超时 → 删除。"""
        cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=0)
        cache.cleanup_expired(now=10.0)  # 10s 后
        assert len(cache._records) == 0

    def test_cleanup_warning_expired(self, cache):
        """WARNING 超时 → 删除。"""
        recs = cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=0)
        recs[0].state = State.WARNING
        cache.cleanup_expired(now=10.0)
        assert len(cache._records) == 0

    def test_cleanup_confirmed_not_deleted(self, cache):
        """CONFIRMED 超时 → 转 CLEARED，不删除。"""
        recs = cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=0)
        recs[0].state = State.CONFIRMED
        stale = cache.cleanup_expired(now=10.0)
        assert len(stale) == 1
        assert len(cache._records) == 1  # 仍在缓存
        assert cache._records[stale[0].id].state == State.CLEARED

    def test_dwell_increments_only_on_match(self, cache):
        """仅匹配成功时驻留帧数累加。"""
        cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=0)
        cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=0.5)
        cache.match_or_create([make_det("car", 0.9, (100, 100, 200, 160))], now=1.0)
        records = list(cache._records.values())
        assert records[0].consecutive_frames == 3


# ===== StateMachine ==========================================================

class TestStateMachine:
    """状态机 9 条迁移规则"""

    @pytest.fixture
    def sm(self):
        return StateMachine(dwell_frames_to_warning=5, disappear_timeout_sec=3.0)

    # ---- 规则 1 -------------------------------------------------------------
    def test_idle_to_warning(self, sm):
        rec = make_record(State.IDLE, dwell=5)
        new_state, needs_qwen, _ = sm.evaluate(rec)
        assert new_state == State.WARNING
        assert needs_qwen is True

    def test_idle_stays_idle(self, sm):
        rec = make_record(State.IDLE, dwell=3)
        new_state, _, _ = sm.evaluate(rec)
        assert new_state == State.IDLE

    def test_moving_car_not_warning(self, sm):
        """正常行驶（位置大幅移动）不应触发 WARNING。"""
        from collections import deque
        rec = make_record(State.IDLE, dwell=8)
        rec.position_history = deque(
            [(100.0 + i * 100, 200.0) for i in range(8)], maxlen=10
        )
        new_state, needs_qwen, reason = sm.evaluate(rec)
        assert new_state == State.IDLE, f"Moving car triggered: {reason}"
        assert "正常行驶" in reason or "spread" in reason.lower()

    # ---- 规则 2,3 -----------------------------------------------------------
    def test_warning_to_confirmed_on_anomaly(self, sm):
        rec = make_record(State.WARNING, dwell=6)
        qr = QwenResult(is_anomaly=True, event_type="违停", confidence=0.9)
        new_state, needs_qwen, _ = sm.evaluate(rec, qwen_result=qr)
        assert new_state == State.CONFIRMED

    def test_warning_to_idle_on_normal(self, sm):
        rec = make_record(State.WARNING, dwell=6)
        qr = QwenResult(is_anomaly=False, event_type="正常")
        new_state, _, _ = sm.evaluate(rec, qwen_result=qr)
        assert new_state == State.IDLE

    # ---- 规则 4,5 -----------------------------------------------------------
    def test_confirmed_to_cleared(self, sm):
        rec = make_record(State.CONFIRMED, dwell=6, last_seen=0.0)
        new_state, needs_qwen, _ = sm.evaluate(rec, current_time=10.0)
        assert new_state == State.CLEARED
        assert needs_qwen is True

    def test_confirmed_refresh(self, sm):
        rec = make_record(State.CONFIRMED, dwell=6, last_seen=35.0)
        rec.state_timestamps[State.CONFIRMED] = 5.0
        new_state, needs_qwen, _ = sm.evaluate(rec, current_time=36.0)
        assert new_state == State.CONFIRMED
        assert needs_qwen is True  # 刷新触发 Qwen

    # ---- 规则 6,7 -----------------------------------------------------------
    def test_cleared_to_idle(self, sm):
        rec = make_record(State.CLEARED, dwell=6)
        qr = QwenResult(risk_cleared=True)
        new_state, _, _ = sm.evaluate(rec, qwen_result=qr)
        assert new_state == State.IDLE

    def test_cleared_to_manual(self, sm):
        rec = make_record(State.CLEARED, dwell=6)
        qr = QwenResult(risk_cleared=False)
        new_state, _, _ = sm.evaluate(rec, qwen_result=qr)
        assert new_state == State.MANUAL_REQUIRED

    # ---- 规则 8,9 -----------------------------------------------------------
    def test_manual_to_idle(self, sm):
        rec = make_record(State.MANUAL_REQUIRED, dwell=6)
        new_state, _, _ = sm.evaluate(rec, manual_action="clear")
        assert new_state == State.IDLE

    def test_manual_to_confirmed(self, sm):
        rec = make_record(State.MANUAL_REQUIRED, dwell=6)
        new_state, _, _ = sm.evaluate(rec, manual_action="confirm")
        assert new_state == State.CONFIRMED

    # ---- apply --------------------------------------------------------------
    def test_apply_transition_updates_record(self, sm):
        rec = make_record(State.IDLE, dwell=5)
        sm.apply_transition(rec, State.WARNING, current_time=10.0)
        assert rec.state == State.WARNING
        assert rec.state_timestamps[State.WARNING] == pytest.approx(10.0)

    def test_apply_to_idle_resets_dwell(self, sm):
        rec = make_record(State.WARNING, dwell=8)
        sm.apply_transition(rec, State.IDLE)
        assert rec.consecutive_frames == 0


# ===== Integration ===========================================================

class TestIntegration:
    """端到端：位置缓存 + 状态机"""

    def test_full_pipeline_idle_to_warning(self):
        cache = PositionCache(match_distance_px=40)
        sm = StateMachine(dwell_frames_to_warning=5, disappear_timeout_sec=3.0)

        records = []
        for i in range(5):
            dets = [make_det("car", 0.9, (100, 100, 200, 160))]
            recs = cache.match_or_create(dets, now=float(i) * 0.5)
            records = recs

        assert len(records) == 1
        assert records[0].consecutive_frames == 5
        new_state, needs_qwen, _ = sm.evaluate(records[0])
        assert new_state == State.WARNING
        assert needs_qwen is True
