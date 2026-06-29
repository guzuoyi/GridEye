"""位置缓存与状态机

核心设计：
- 不依赖 YOLO 跟踪 ID，纯靠空间位置匹配
- 5 状态机：IDLE → WARNING → CONFIRMED → CLEARED → MANUAL_REQUIRED
- 嫌疑锁定：3帧位移<200px → 锁坐标 → 5帧中≥3框体稳定 → WARNING
"""

import uuid, time, math
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Dict, Tuple
from logging import getLogger

from src.utils.coords import euclidean_distance

logger = getLogger(__name__)


class State(Enum):
    IDLE = "idle"
    WARNING = "warning"
    CONFIRMED = "confirmed"
    CLEARED = "cleared"
    MANUAL_REQUIRED = "manual_required"


@dataclass
class QwenResult:
    is_anomaly: bool = False
    event_type: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    risk_cleared: bool = False
    need_cleanup: bool = False


@dataclass
class PositionRecord:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    class_name: str = ""
    center_x: float = 0.0
    center_y: float = 0.0
    consecutive_frames: int = 0
    last_seen_time: float = field(default_factory=time.time)
    state: State = State.IDLE
    state_timestamps: Dict[State, float] = field(default_factory=dict)
    qwen_result: Optional[QwenResult] = None
    lane_number: int = -1
    longitudinal_distance: float = -1.0
    size_valid_history: deque = field(default_factory=lambda: deque(maxlen=10))
    dwell_history: deque = field(default_factory=lambda: deque(maxlen=10))
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    bbox_area_history: deque = field(default_factory=lambda: deque(maxlen=10))
    _last_bbox_area: float = 0.0
    _miss_count: int = 0
    # 新逻辑字段
    _locked: bool = False           # 是否锁定中心点
    _locked_cx: float = 0.0         # 锁定时的中心 x
    _locked_cy: float = 0.0         # 锁定时的中心 y
    _lock_frame_count: int = 0      # 锁定后经过的帧数
    _lock_stable_count: int = 0     # 锁定后框体稳定帧数
    _disappear_window: deque = field(default_factory=lambda: deque(maxlen=30))
    _qwen_pending: bool = False
    _last_rejected_time: float = 0.0

    def __post_init__(self):
        if not self.state_timestamps:
            self.state_timestamps = {s: 0.0 for s in State}
        self.state_timestamps[self.state] = self.last_seen_time

    # ── 位置 ────────────────────────────────────────────────────────────

    @property
    def position_spread_px(self) -> float:
        pts = list(self.position_history)
        if len(pts) < 2:
            return 0.0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)

    def dist_to_locked(self) -> float:
        """当前中心点到锁定位置的距离"""
        if not self._locked:
            return 0.0
        return math.sqrt((self.center_x - self._locked_cx)**2 +
                         (self.center_y - self._locked_cy)**2)

    def bbox_stable_now(self) -> bool:
        """最新一帧的框体是否稳定（变化<20%）"""
        hist = list(self.bbox_area_history)
        return hist[-1] if hist else True

    # ── 状态 ────────────────────────────────────────────────────────────

    def seconds_since_seen(self, now=None):
        return (now or time.time()) - self.last_seen_time

    def elapsed_in_state(self, s: State, now=None):
        return (now or time.time()) - self.state_timestamps.get(s, 0)

    def transition_to(self, new_state: State, now=None):
        now = now or time.time()
        self.state = new_state
        self.state_timestamps[new_state] = now

    # ── 特征更新 ────────────────────────────────────────────────────────

    def update_from_detection(self, detection, now=None):
        """用匹配到的检测更新。锁定状态下仅记录存在性，不移坐标。"""
        now = now or time.time()
        cur_area = getattr(detection, 'area', 0) or 0

        # 框体稳定性（始终计算）
        if self._last_bbox_area > 0 and cur_area > 0:
            change = abs(cur_area - self._last_bbox_area) / self._last_bbox_area
            self.bbox_area_history.append(change < 0.20)
        else:
            self.bbox_area_history.append(True)
        self._last_bbox_area = cur_area

        # 坐标锁定状态：不移坐标，不增加驻留，仅记录时间
        if self._locked:
            self.last_seen_time = now
            self._miss_count = 0
            return

        # 正常情况下更新坐标
        self.center_x = detection.center[0]
        self.center_y = detection.center[1]
        self.consecutive_frames += 1
        self.last_seen_time = now
        self._miss_count = 0
        self.dwell_history.append(True)
        self.position_history.append((self.center_x, self.center_y))
        if hasattr(detection, "is_size_valid") and detection.is_size_valid is not None:
            self.size_valid_history.append(detection.is_size_valid)

    def mark_missed(self, grace_frames=3):
        self._miss_count = getattr(self, '_miss_count', 0) + 1
        if self._miss_count <= grace_frames:
            return True
        self.dwell_history.append(False)
        return False

    # ── 计算属性 ────────────────────────────────────────────────────────

    @property
    def is_stable_size(self) -> bool:
        if len(self.size_valid_history) < 7:
            return False
        return sum(self.size_valid_history) >= 7

    @property
    def dwell_ratio(self) -> float:
        if len(self.dwell_history) == 0:
            return 0.0
        return sum(self.dwell_history) / len(self.dwell_history)


# ==========================================================================
# PositionCache
# ==========================================================================

class PositionCache:
    def __init__(self, match_distance_px=40.0, disappear_timeout_sec=3.0,
                 max_miss_frames=3):
        self.match_distance = match_distance_px
        self.disappear_timeout = disappear_timeout_sec
        self.max_miss_frames = max_miss_frames
        self._records: Dict[str, PositionRecord] = {}

    def match_or_create(self, detections, now=None):
        now = now or time.time()
        matched_ids = set()

        for det in detections:
            best_id, best_dist = None, float("inf")
            for rid, rec in self._records.items():
                if rid in matched_ids:
                    continue
                if rec.class_name != det.class_name:
                    continue
                d = euclidean_distance(det.center, (rec.center_x, rec.center_y))
                if d < self.match_distance and d < best_dist:
                    best_dist = d
                    best_id = rid

            if best_id is not None:
                self._records[best_id].update_from_detection(det, now)
                matched_ids.add(best_id)
            else:
                rec = PositionRecord(class_name=det.class_name,
                    center_x=det.center[0], center_y=det.center[1],
                    consecutive_frames=1, last_seen_time=now)
                rec.dwell_history.append(True)
                rec.position_history.append((det.center[0], det.center[1]))
                self._records[rec.id] = rec
                matched_ids.add(rec.id)

        for rid in self._records:
            if rid not in matched_ids:
                self._records[rid].mark_missed(self.max_miss_frames)

        return list(self._records.values())

    def get(self, rid):
        return self._records.get(rid)

    def get_active(self):
        return list(self._records.values())

    def get_by_state(self, s):
        return [r for r in self._records.values() if r.state == s]

    def cleanup_expired(self, now=None):
        now = now or time.time()
        removed = []
        for rid, rec in list(self._records.items()):
            miss = getattr(rec, '_miss_count', 0)
            if rec.state == State.CONFIRMED:
                elapsed = rec.seconds_since_seen(now)
                if elapsed >= self.disappear_timeout:
                    rec.transition_to(State.CLEARED, now)
                    removed.append(rec)
            elif rec.state in (State.IDLE, State.WARNING):
                if miss > self.max_miss_frames:
                    removed.append(rec)
                    del self._records[rid]
        return removed

    def __len__(self):
        return len(self._records)


# ==========================================================================
# StateMachine
# ==========================================================================

class StateMachine:
    """新判定逻辑:
    1. IDLE: 连续3帧位移<200px → 锁坐标
    2. 锁后5帧中≥3帧框体稳定 → WARNING → Qwen
    3. Qwen判异常 → CONFIRMED; Qwen判正常 → IDLE+冷却
    4. CONFIRMED: 30帧中≥60%在锁坐标200px内无检测 → CLEARED → Qwen
    """

    def __init__(self, dwell_frames_to_warning=7,
                 disappear_timeout_sec=3.0, refresh_interval_sec=30.0,
                 stationary_threshold_px=200, cooldown_frames=30):
        self.stationary_threshold = stationary_threshold_px
        self.cooldown_frames = cooldown_frames
        self.disappear_timeout = disappear_timeout_sec
        self.refresh_interval = refresh_interval_sec

    def evaluate(self, record, qwen_result=None, manual_action=None,
                 current_time=None):
        now = current_time or time.time()
        cur = record.state
        sp = record.position_spread_px

        # ═══════════ 锁定判定（优先，锁后 IDLE 走此路径） ═══════════
        if record._locked and cur == State.IDLE:
            record._lock_frame_count += 1
            if record.bbox_stable_now():
                record._lock_stable_count += 1

            sc = record._lock_stable_count
            fc = record._lock_frame_count

            if fc < 10:
                return (State.IDLE, False, f"判定中({fc}/10帧) 稳定={sc}")

            if sc >= 6:
                return (State.WARNING, True, f"报警: {sc}/10框体稳定")
            else:
                record._locked = False
                record._locked_cx = 0
                record._locked_cy = 0
                return (State.IDLE, False, f"解锁(稳定不足 {sc}/{fc})")

        # ═══════════ IDLE: 嫌疑检测 ═══════════
        if cur == State.IDLE:
            # 冷却
            if record._last_rejected_time > 0:
                since = int((now - record._last_rejected_time) * 2)
                if since < self.cooldown_frames:
                    return (State.IDLE, False, f"冷却({since}/{self.cooldown_frames})")

            # 连续 ≥3 帧且位移 < 200px → 锁坐标
            if record.consecutive_frames >= 3 and sp < self.stationary_threshold and sp >= 0:
                record._locked = True
                record._locked_cx = record.center_x
                record._locked_cy = record.center_y
                record._lock_frame_count = 0
                record._lock_stable_count = 0
                return (State.IDLE, False,
                    f"嫌疑锁定({record._locked_cx:.0f},{record._locked_cy:.0f}) sp={sp:.0f}")

            return (State.IDLE, False, f"d={record.consecutive_frames} sp={sp:.0f}")

        # ═══════════ WARNING ═══════════
        if cur == State.WARNING:
            if qwen_result is not None:
                if qwen_result.is_anomaly:
                    record._disappear_window.clear()
                    return (State.CONFIRMED, False, f"异常: {qwen_result.event_type}")
                else:
                    record._last_rejected_time = now
                    record._locked = False
                    return (State.IDLE, False, "Qwen判正常")
            return (State.WARNING, False, "等待Qwen")

        # ═══════════ CONFIRMED: 消失判定 ═══════════
        if cur == State.CONFIRMED:
            # 检测当前帧是否有目标在锁定坐标 200px 内
            dist = record.dist_to_locked()
            detected = dist < self.stationary_threshold

            record._disappear_window.append(not detected)  # True=无检测
            dw = list(record._disappear_window)

            if len(dw) >= 30:
                missed = sum(dw)
                if missed >= 18:  # ≥ 60%
                    return (State.CLEARED, True,
                        f"消失 {missed}/{len(dw)}帧")

            if len(dw) % 10 == 0:
                missed = sum(dw)
                return (State.CONFIRMED, False,
                    f"监控中(消失{missed}/{len(dw)})")
            return (State.CONFIRMED, False, "持续监控")

        # ═══════════ CLEARED ═══════════
        if cur == State.CLEARED:
            if qwen_result is not None:
                if qwen_result.risk_cleared:
                    record._locked = False
                    record._disappear_window.clear()
                    return (State.IDLE, False, "风险消除")
                else:
                    return (State.MANUAL_REQUIRED, False, "风险未消除")
            return (State.CLEARED, False, "等待Qwen")

        # ═══════════ MANUAL_REQUIRED ═══════════
        if cur == State.MANUAL_REQUIRED:
            if manual_action == "clear":
                record._locked = False
                return (State.IDLE, False, "人工消除")
            if manual_action == "confirm":
                return (State.CONFIRMED, False, "人工确认存在")
            return (State.MANUAL_REQUIRED, False, "等待人工")

        return (cur, False, "无规则匹配")

    def apply_transition(self, record, new_state, qwen_result=None,
                         current_time=None):
        now = current_time or time.time()
        old = record.state
        record.state = new_state
        record.state_timestamps[new_state] = now
        if qwen_result is not None:
            record.qwen_result = qwen_result
        if new_state != old and new_state == State.IDLE:
            record.consecutive_frames = 0
            record._locked = False
