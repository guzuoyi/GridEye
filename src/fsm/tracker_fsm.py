"""TrafficEventFSM v2 — 空间冷却 + 容错漏桶 + 合并 OBSERVING"""

import time, math
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple


class State(Enum):
    IDLE = "idle"
    OBSERVING = "observing"    # 合并 SUSPECT + WARNING
    CONFIRMED = "confirmed"
    CLEARED = "cleared"
    MANUAL_REQUIRED = "manual_required"


@dataclass
class TrafficEventFSM:
    track_id: int
    cls_name: str = ""
    ref_cx: float = 0.0
    ref_cy: float = 0.0
    ref_w: float = 0.0
    ref_h: float = 0.0

    state: State = State.IDLE
    state_since: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    idle_still_frames: int = 0
    observing_frames: int = 0
    observing_ok_frames: int = 0     # 容错漏桶：满足条件的帧数
    missing_frames: int = 0
    pos_history: deque = field(default_factory=lambda: deque(maxlen=5))

    # 时间胶囊：SUSPECT 锁定时保存的全局帧路径
    t0_snapshot: Optional[str] = None
    _qwen_pending: bool = False      # 有 Qwen 请求待处理，不清理

    IDLE_TO_LOCK = 3               # 3帧位移<10px → 锁定
    OBSERVING_WINDOW = 100         # 100帧观察窗口
    OBSERVING_OK_NEED = 80         # 100帧中≥80帧满足条件 → 触发
    LOCK_DIST_THRESH = 30          # 距锁定中心 < 30px
    CLEARED_MISSING = 150

    def update(self, cx: float, cy: float, w: float, h: float,
               spatial_cooldown_check=None) -> Tuple[State, bool, str]:
        """返回 (state, need_qwen, reason)。spatial_cooldown_check(cx,cy)→bool"""
        now = time.time()
        self.last_seen = now
        self.missing_frames = 0
        self.pos_history.append((cx, cy))
        cur = self.state

        # ── IDLE ──
        if cur == State.IDLE:
            still = self._centroid_shift() < 10.0
            if still and len(self.pos_history) >= 2:
                self.idle_still_frames += 1
            else:
                self.idle_still_frames = 0

            if self.idle_still_frames >= self.IDLE_TO_LOCK:
                # 空间冷却检查
                if spatial_cooldown_check and spatial_cooldown_check(cx, cy):
                    self.idle_still_frames = 0
                    return (State.IDLE, False, "spatial cooldown")

                self.state = State.OBSERVING
                self.state_since = now
                self.ref_cx = cx; self.ref_cy = cy
                self.ref_w = w; self.ref_h = h
                self.observing_frames = 0
                self.observing_ok_frames = 0
                return (State.OBSERVING, False, f"locked ({cx:.0f},{cy:.0f})")
            return (State.IDLE, False, f"still={self.idle_still_frames}")

        # ── OBSERVING (合并 SUSPECT+WARNING，容错漏桶) ──
        if cur == State.OBSERVING:
            self.observing_frames += 1
            dist = self._dist_to_ref(cx, cy)
            aspect_ok = self._aspect_stable(w, h)

            if dist > self.LOCK_DIST_THRESH or not aspect_ok:
                # 不满足但不立即打断，只不增加 ok 计数
                pass
            else:
                self.observing_ok_frames += 1

            if self.observing_frames >= self.OBSERVING_WINDOW:
                if self.observing_ok_frames >= self.OBSERVING_OK_NEED:
                    # 空间冷却二次检查
                    if spatial_cooldown_check and spatial_cooldown_check(self.ref_cx, self.ref_cy):
                        self._reset()
                        return (State.IDLE, False, "spatial cooldown at trigger")
                    return (State.OBSERVING, True,
                        f"trigger ({self.observing_ok_frames}/{self.observing_frames})")
                else:
                    self._reset()
                    return (State.IDLE, False,
                        f"unstable ({self.observing_ok_frames}/{self.observing_frames})")

            return (State.OBSERVING, False,
                f"obs {self.observing_frames}/{self.OBSERVING_WINDOW} ok={self.observing_ok_frames}")

        # ── CONFIRMED ──
        if cur == State.CONFIRMED:
            return (State.CONFIRMED, False, "confirmed")

        return (cur, False, "")

    def mark_missing(self) -> Tuple[State, bool, str]:
        self.missing_frames += 1
        if self.state == State.CONFIRMED and self.missing_frames >= self.CLEARED_MISSING:
            self.state = State.CLEARED
            self.state_since = time.time()
            return (State.CLEARED, True, f"cleared ({self.missing_frames}f)")
        if self.state in (State.IDLE, State.OBSERVING) and self.missing_frames > 10:
            return (State.IDLE, False, "expired")
        return (self.state, False, "miss")

    def on_qwen_result(self, is_anomaly: bool, event_type: str):
        if self.state == State.OBSERVING:
            if is_anomaly:
                self.state = State.CONFIRMED
                self.state_since = time.time()
                self.missing_frames = 0
            else:
                self._reset()
        elif self.state == State.CLEARED:
            self.state = State.MANUAL_REQUIRED if is_anomaly else State.IDLE
            if not is_anomaly:
                self._reset()

    # ── helpers ──
    def _centroid_shift(self):
        pts = list(self.pos_history)
        if len(pts) < 2: return 0.0
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)

    def _dist_to_ref(self, cx, cy):
        return math.sqrt((cx-self.ref_cx)**2 + (cy-self.ref_cy)**2)

    def _aspect_stable(self, w, h):
        if self.ref_w<=0 or self.ref_h<=0 or w<=0 or h<=0: return True
        return abs((w/h)/(self.ref_w/self.ref_h)-1) < 0.15

    def _reset(self):
        self.state = State.IDLE
        self.state_since = time.time()
        self.idle_still_frames = 0
        self.observing_frames = 0
        self.observing_ok_frames = 0
        self.pos_history.clear()
