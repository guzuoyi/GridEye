"""TrackerManager v2 — 空间冷却 + 幽灵池强化"""

import time, math
from typing import Dict, Optional, List
from src.fsm.tracker_fsm import TrafficEventFSM, State


class CooldownEntry:
    __slots__ = ("cx", "cy", "until")
    def __init__(self, cx, cy, ttl=60):
        self.cx = cx; self.cy = cy; self.until = time.time() + ttl

class GhostEntry:
    def __init__(self, fsm):
        self.ref_cx = fsm.ref_cx; self.ref_cy = fsm.ref_cy
        self.ref_w = fsm.ref_w; self.ref_h = fsm.ref_h
        self.state = fsm.state
        self.observing_frames = fsm.observing_frames
        self.observing_ok_frames = fsm.observing_ok_frames
        self.missing_frames = fsm.missing_frames
        self.cls_name = fsm.cls_name
        self.t0_snapshot = fsm.t0_snapshot
        self.ghost_since = time.time()


class TrackerManager:
    GHOST_TTL = 300
    GHOST_MATCH_DIST = 30
    COOLDOWN_RADIUS = 50
    COOLDOWN_TTL = 60

    def __init__(self):
        self._fsms: Dict[int, TrafficEventFSM] = {}
        self._ghosts: List[GhostEntry] = []
        self._cooldowns: List[CooldownEntry] = []
        self._active_tracks: set = set()

    # ── 空间冷却 ──
    def add_cooldown(self, cx, cy):
        self._cooldowns.append(CooldownEntry(cx, cy, self.COOLDOWN_TTL))

    def _check_cooldown(self, cx, cy) -> bool:
        """返回 True 表示在冷却区内"""
        now = time.time()
        self._cooldowns = [c for c in self._cooldowns if now < c.until]
        for c in self._cooldowns:
            if math.sqrt((cx-c.cx)**2 + (cy-c.cy)**2) < self.COOLDOWN_RADIUS:
                return True
        return False

    # ── 主更新 ──
    def update(self, track_id, cls_name, cx, cy, w, h):
        fsm = self._fsms.get(track_id)
        if fsm is None:
            ghost = self._find_ghost(cx, cy, cls_name, w, h)
            if ghost:
                fsm = self._spawn_from_ghost(track_id, ghost)
                self._ghosts.remove(ghost)
            else:
                fsm = TrafficEventFSM(track_id=track_id, cls_name=cls_name)
            self._fsms[track_id] = fsm

        fsm.cls_name = cls_name
        self._active_tracks.add(track_id)
        return fsm.update(cx, cy, w, h, spatial_cooldown_check=self._check_cooldown)

    # ── 丢失 ──
    def mark_missing(self) -> List[int]:
        cleared = []
        for tid in list(self._fsms.keys()):
            if tid not in self._active_tracks:
                fsm = self._fsms[tid]
                state, need_qwen, _ = fsm.mark_missing()
                if need_qwen and state == State.CLEARED:
                    cleared.append(tid)
                elif fsm.missing_frames > 10:
                    # 有 pending Qwen 的不清理
                    if fsm._qwen_pending:
                        continue
                    if fsm.state in (State.OBSERVING, State.CONFIRMED):
                        self._ghosts.append(GhostEntry(fsm))
                    del self._fsms[tid]
        self._active_tracks.clear()
        self._reap_ghosts()
        return cleared

    # ── Qwen ──
    def on_qwen(self, track_id, is_anomaly, event_type=""):
        fsm = self._fsms.get(track_id)
        if fsm is None:
            # 尝试幽灵池
            for g in self._ghosts:
                if g.original_track_id if hasattr(g,'original_track_id') else False:
                    continue  # skip
            import logging
            logging.getLogger(__name__).debug(f"Qwen cb: track {track_id} not found")
            return  # 目标已销毁，放弃结果

        fsm.on_qwen_result(is_anomaly, event_type)
        if not is_anomaly:
            self.add_cooldown(fsm.ref_cx, fsm.ref_cy)
        if fsm.state == State.IDLE:
            del self._fsms[track_id]

    # ── 幽灵池（强化） ──
    def _find_ghost(self, cx, cy, cls_name, w, h):
        for g in self._ghosts:
            dist = math.sqrt((cx-g.ref_cx)**2 + (cy-g.ref_cy)**2)
            if dist >= self.GHOST_MATCH_DIST:
                continue
            if g.cls_name != cls_name:
                continue
            # 尺寸校验 ±30%
            if g.ref_w > 0 and g.ref_h > 0:
                g_area = g.ref_w * g.ref_h
                c_area = w * h
                if abs(c_area - g_area) / g_area > 0.30:
                    continue
            return g
        return None

    def _spawn_from_ghost(self, tid, ghost):
        fsm = TrafficEventFSM(track_id=tid, cls_name=ghost.cls_name)
        fsm.ref_cx = ghost.ref_cx; fsm.ref_cy = ghost.ref_cy
        fsm.ref_w = ghost.ref_w; fsm.ref_h = ghost.ref_h
        fsm.state = ghost.state
        fsm.observing_frames = ghost.observing_frames
        fsm.observing_ok_frames = ghost.observing_ok_frames
        fsm.missing_frames = 0
        fsm.t0_snapshot = ghost.t0_snapshot
        return fsm

    def _reap_ghosts(self):
        now = time.time()
        self._ghosts = [g for g in self._ghosts if now - g.ghost_since < self.GHOST_TTL]

    # ── 遮挡检测 ──
    def is_roi_occluded(self, cx, cy, radius=60):
        for fsm in self._fsms.values():
            if fsm.track_id in self._active_tracks and fsm.state == State.IDLE:
                if fsm.pos_history:
                    last = fsm.pos_history[-1]
                    if math.sqrt((last[0]-cx)**2+(last[1]-cy)**2) < radius:
                        if fsm.ref_w * fsm.ref_h > 5000:
                            return True
        return False

    def get(self, tid): return self._fsms.get(tid)
    def get_all(self): return list(self._fsms.values())
    def __len__(self): return len(self._fsms)
