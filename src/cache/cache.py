"""位置缓存与状态机

核心设计：
- 不依赖 YOLO 跟踪 ID，纯靠空间位置匹配（类别 + 欧氏距离 < 40px）
- 5 状态机：IDLE → WARNING → CONFIRMED → CLEARED → MANUAL_REQUIRED
- 全字段 PositionRecord，单一真实数据源
"""

import uuid
import time
import math
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Dict, Tuple
from logging import getLogger

from src.utils.coords import euclidean_distance

logger = getLogger(__name__)


# ==========================================================================
# 状态枚举
# ==========================================================================

class State(Enum):
    IDLE = "idle"
    WARNING = "warning"
    CONFIRMED = "confirmed"
    CLEARED = "cleared"
    MANUAL_REQUIRED = "manual_required"


# ==========================================================================
# PositionRecord
# ==========================================================================

@dataclass
class QwenResult:
    """大模型判定结果"""
    is_anomaly: bool = False
    event_type: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    risk_cleared: bool = False
    need_cleanup: bool = False


@dataclass
class PositionRecord:
    """位置缓存中的一条目标记录。

    坐标约定: 所有 x,y 基于 1920x1080 原始分辨率。
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    class_name: str = ""               # car / truck / person / two-wheeler
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
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))  # 最近N帧中心点 (cx,cy)  # 最近N帧是否匹配成功

    def __post_init__(self):
        if not self.state_timestamps:
            self.state_timestamps = {s: 0.0 for s in State}
        self.state_timestamps[self.state] = self.last_seen_time

    @property
    def is_stable_size(self) -> bool:
        """最近 10 帧中 ≥ 7 帧尺寸有效。"""
        if len(self.size_valid_history) < 7:
            return False
        return sum(self.size_valid_history) >= 7

    def update_from_detection(self, detection, now: Optional[float] = None) -> None:
        """用匹配到的检测更新此记录。"""
        if now is None:
            now = time.time()
        self.center_x = detection.center[0]
        self.center_y = detection.center[1]
        self.consecutive_frames += 1
        self.last_seen_time = now
        self.dwell_history.append(True)
        self.position_history.append((self.center_x, self.center_y))
        if hasattr(detection, "is_size_valid") and detection.is_size_valid is not None:
            self.size_valid_history.append(detection.is_size_valid)

    def mark_missed(self) -> None:
        """标记本帧未匹配到目标。"""
        self.dwell_history.append(False)

    @property
    def dwell_ratio(self) -> float:
        """最近 N 帧中匹配成功的比例。"""
        if len(self.dwell_history) == 0:
            return 0.0
        return sum(self.dwell_history) / len(self.dwell_history)

    @property
    def position_spread_px(self) -> float:
        """最近 N 帧中心点的最大位移 (px)。越小越静止。"""
        pts = list(self.position_history)
        if len(pts) < 2:
            return 0.0
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        max_dx = max(xs) - min(xs)
        max_dy = max(ys) - min(ys)
        return (max_dx ** 2 + max_dy ** 2) ** 0.5

    def seconds_since_seen(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        return now - self.last_seen_time

    def elapsed_in_state(self, state: State, now: Optional[float] = None) -> float:
        if now is None:
            now = time.time()
        entered = self.state_timestamps.get(state, 0.0)
        return now - entered if entered > 0 else 0.0

    def transition_to(self, new_state: State, now: Optional[float] = None) -> None:
        """执行状态迁移。"""
        if now is None:
            now = time.time()
        self.state = new_state
        self.state_timestamps[new_state] = now
        logger.info(
            f"[{self.id[:8]}] {self.class_name} → {new_state.value} "
            f"(frames={self.consecutive_frames}, lane={self.lane_number})"
        )

    def requires_qwen_call(self) -> bool:
        """当前状态是否需要调用 Qwen。"""
        return self.state in (State.WARNING, State.CLEARED)


# ==========================================================================
# PositionCache
# ==========================================================================

class PositionCache:
    """位置缓存管理器。

    核心逻辑：
    - match_or_create: 类别相同 + 中心点欧氏距离 < match_distance → 匹配，否则新建
    - cleanup_expired: IDLE/WARNING 3s 未命中 → 删除；CONFIRMED 3s → 转 CLEARED
    """

    def __init__(
        self,
        match_distance_px: float = 40.0,
        disappear_timeout_sec: float = 3.0,
    ):
        self.match_distance = match_distance_px
        self.default_distance = match_distance_px
        self.disappear_timeout = disappear_timeout_sec
        self._records: Dict[str, PositionRecord] = {}

    # -- 匹配 ----------------------------------------------------------------

    def match_or_create(
        self, detections: list, now: Optional[float] = None
    ) -> List[PositionRecord]:
        """将检测结果与缓存匹配，返回匹配后的记录列表。

        参数:
            detections: Detection 列表
            now: 当前时间戳

        返回:
            活跃位置记录列表（含新建的）
        """
        if now is None:
            now = time.time()

        matched_record_ids = set()

        for det in detections:
            best_id, best_dist = None, float("inf")

            for rid, rec in self._records.items():
                if rid in matched_record_ids:
                    continue
                if rec.class_name != det.class_name:
                    continue
                dist = euclidean_distance(
                    det.center, (rec.center_x, rec.center_y)
                )
                if dist < self.match_distance and dist < best_dist:
                    best_dist = dist
                    best_id = rid

            if best_id is not None:
                # 匹配成功
                rec = self._records[best_id]
                rec.update_from_detection(det, now)
                matched_record_ids.add(best_id)
            else:
                # 新建记录
                rec = PositionRecord(
                    class_name=det.class_name,
                    center_x=det.center[0],
                    center_y=det.center[1],
                    consecutive_frames=1,
                    last_seen_time=now,
                    state=State.IDLE,
                )
                rec.dwell_history.append(True)
                rec.position_history.append((det.center[0], det.center[1]))
                if hasattr(det, "is_size_valid") and det.is_size_valid is not None:
                    rec.size_valid_history.append(det.is_size_valid)
                self._records[rec.id] = rec
                matched_record_ids.add(rec.id)

        # 标记未匹配的记录
        for rid in self._records:
            if rid not in matched_record_ids:
                self._records[rid].mark_missed()

        # 返回所有活跃记录
        active = [self._records[rid] for rid in self._records]
        return active

    # -- 查询 ----------------------------------------------------------------

    def get_active(self) -> List[PositionRecord]:
        return list(self._records.values())

    def get(self, record_id: str) -> Optional[PositionRecord]:
        return self._records.get(record_id)

    def get_by_state(self, state: State) -> List[PositionRecord]:
        return [r for r in self._records.values() if r.state == state]

    # -- 清理 ----------------------------------------------------------------

    def cleanup_expired(self, now: Optional[float] = None) -> List[PositionRecord]:
        """清理过期记录，返回被清理的记录列表。

        IDLE/WARNING: 3s 未命中 → 直接删除
        CONFIRMED: 3s 未命中 → 转 CLEARED（不删除）
        CLEARED/MANUAL_REQUIRED: 由状态机处理，此处不删除
        """
        if now is None:
            now = time.time()

        removed = []
        for rid, rec in list(self._records.items()):
            elapsed = rec.seconds_since_seen(now)
            if elapsed < self.disappear_timeout:
                continue

            if rec.state == State.CONFIRMED:
                # 不删除，转为 CLEARED → 由调用者触发 Qwen 风险消除
                rec.transition_to(State.CLEARED, now)
                removed.append(rec)  # 返回给调用者，用于触发 Qwen
                logger.info(
                    f"[{rid[:8]}] CONFIRMED 消失 {elapsed:.1f}s → CLEARED"
                )
            elif rec.state in (State.IDLE, State.WARNING):
                # 短时停留，直接删除
                removed.append(rec)
                del self._records[rid]
                logger.debug(
                    f"[{rid[:8]}] {rec.state.value} 消失 {elapsed:.1f}s → 删除"
                )

        return removed

    def manual_cleanup_stale(self, now: Optional[float] = None) -> List[PositionRecord]:
        """清理 MANUAL_REQUIRED 超时未确认的记录（仅记录日志，不删除）。"""
        if now is None:
            now = time.time()
        stale = []
        for rec in self._records.values():
            if rec.state == State.MANUAL_REQUIRED:
                # 由状态机处理超时，缓存只记录
                elapsed = rec.elapsed_in_state(State.MANUAL_REQUIRED, now)
                if elapsed > 300:  # 5 分钟
                    stale.append(rec)
                    logger.warning(
                        f"[{rec.id[:8]}] MANUAL_REQUIRED 超时 {elapsed:.0f}s"
                    )
        return stale

    def remove(self, record_id: str) -> Optional[PositionRecord]:
        return self._records.pop(record_id, None)

    def __len__(self) -> int:
        return len(self._records)


# =============================================================================
# StateMachine —— 5 状态 9 条迁移规则
# =============================================================================

class StateMachine:
    """交通事件状态机。

    五个状态:
        IDLE            — 空闲（目标首次出现）
        WARNING         — 预警（连续驻留 ≥ N 帧）
        CONFIRMED       — 事件确认（Qwen 判异常）
        CLEARED         — 风险消除待确认（目标消失 ≥ T 秒）
        MANUAL_REQUIRED — 需人工介入（Qwen 判未消除）

    九条迁移规则:
        1. IDLE → WARNING         驻留 ≥ dwell_frames
        2. WARNING → CONFIRMED     Qwen 返回异常
        3. WARNING → IDLE          Qwen 返回正常 或 目标消失
        4. CONFIRMED → CLEARED     目标消失 ≥ timeout
        5. CONFIRMED → CONFIRMED   周期性刷新（> refresh_interval）
        6. CLEARED → IDLE          Qwen 确认消除
        7. CLEARED → MANUAL_REQUIRED  Qwen 判定未消除
        8. MANUAL_REQUIRED → IDLE  人工确认消除
        9. MANUAL_REQUIRED → CONFIRMED  人工确认仍存在

    用法:
        sm = StateMachine(dwell_frames=5, disappear_timeout=3.0, refresh_interval=30.0)
        new_state, needs_qwen = sm.transition(record, qwen_result=None)
    """

    def __init__(
        self,
        dwell_frames_to_warning: int = 5,
        disappear_timeout_sec: float = 3.0,
        refresh_interval_sec: float = 30.0,
        stationary_threshold_px: float = 150.0,
    ):
        self.dwell_frames = dwell_frames_to_warning
        self.disappear_timeout = disappear_timeout_sec
        self.refresh_interval = refresh_interval_sec
        self.stationary_threshold_px = stationary_threshold_px

    def evaluate(
        self,
        record: PositionRecord,
        qwen_result: Optional[QwenResult] = None,
        manual_action: Optional[str] = None,  # "clear" | "confirm"
        current_time: Optional[float] = None,
    ) -> Tuple[State, bool, str]:
        """评估状态迁移。

        返回:
            (new_state, needs_qwen_call, reason)
        """
        import time as _time
        now = current_time or _time.time()

        current = record.state

        # ---- 规则 1: IDLE → WARNING (滑动窗口 + 静止判定) ----
        if current == State.IDLE:
            need_match = self.dwell_frames
            window = list(record.dwell_history)
            matched = sum(window[-10:]) if len(window) >= need_match else 0

            # 静止判定：中心点最大位移 < 阈值
            spread = record.position_spread_px
            stationary = spread < self.stationary_threshold_px and len(record.position_history) >= need_match

            if matched >= need_match and stationary:
                return (State.WARNING, True,
                    f"匹配{matched}/10帧 + 静止(spread={spread:.0f}px)")
            if matched >= need_match and not stationary:
                return (State.IDLE, False,
                    f"正常行驶 (spread={spread:.0f}px > {self.stationary_threshold_px}px)")
            return (State.IDLE, False,
                f"匹配不足 {matched}/{need_match}")

        # ---- 规则 2,3: WARNING 状态 ----
        if current == State.WARNING:
            if qwen_result is not None:
                if qwen_result.is_anomaly:
                    return (State.CONFIRMED, False, f"Qwen 确认异常: {qwen_result.event_type}")
                else:
                    return (State.IDLE, False, f"Qwen 判定正常")
            # Qwen 尚未返回 → 保持 WARNING
            return (State.WARNING, False, "等待 Qwen 响应")

        # ---- 规则 4,5: CONFIRMED 状态 ----
        if current == State.CONFIRMED:
            elapsed_since_seen = now - record.last_seen_time
            if elapsed_since_seen >= self.disappear_timeout:
                return (State.CLEARED, True, f"目标消失 {elapsed_since_seen:.1f}s")

            # 周期性刷新
            ts_confirmed = record.state_timestamps.get(State.CONFIRMED)
            if ts_confirmed and (now - ts_confirmed) >= self.refresh_interval:
                return (State.CONFIRMED, True, "周期性状态刷新")

            return (State.CONFIRMED, False, "持续监控")

        # ---- 规则 6,7: CLEARED 状态 ----
        if current == State.CLEARED:
            if qwen_result is not None:
                if qwen_result.risk_cleared:
                    return (State.IDLE, False, "Qwen 确认风险消除")
                else:
                    return (State.MANUAL_REQUIRED, False, "Qwen 判定风险未消除")
            return (State.CLEARED, False, "等待 Qwen 风险消除确认")

        # ---- 规则 8,9: MANUAL_REQUIRED 状态 ----
        if current == State.MANUAL_REQUIRED:
            if manual_action == "clear":
                return (State.IDLE, False, "人工确认消除")
            if manual_action == "confirm":
                return (State.CONFIRMED, False, "人工确认仍存在")
            return (State.MANUAL_REQUIRED, False, "等待人工介入")

        return (current, False, "未知状态")

    def apply_transition(
        self,
        record: PositionRecord,
        new_state: State,
        qwen_result: Optional[QwenResult] = None,
        current_time: Optional[float] = None,
    ) -> None:
        """将评估结果应用到记录上。"""
        import time as _time
        now = current_time or _time.time()

        old_state = record.state
        record.state = new_state
        record.state_timestamps[new_state] = now

        if qwen_result is not None:
            record.qwen_result = qwen_result

        if new_state != old_state and new_state == State.IDLE:
            # 回到 IDLE → 重置驻留
            record.consecutive_frames = 0
