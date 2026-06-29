"""SpatialTemporalGrid v2 — 线性衰减 + 高斯加权"""

import time, math, cv2, numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GridCell:
    count: int = 0
    inactive_frames: int = 0
    was_hot: bool = False  # 曾达到120，允许消警


@dataclass
class AlertCandidate:
    cx: float; cy: float; w: float; h: float
    first_seen: float; last_updated: float
    alerted: bool = False
    confirmed: bool = False
    qwen_pending: bool = False
    locked_frames: int = 0        # 锁定帧数计数 (50帧冷却)
    require_drop_below_20: bool = False


class SpatialTemporalGrid:
    CELL_SIZE = 50
    HOLD_THRESHOLD = 120
    VELOCITY_THRESH = 8
    COOLDOWN_TTL = 60
    CLUSTER_DIST = 60
    MAX_COUNT = 400
    DECAY_RATE = 1.0

    GAUSS_3x3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=np.int32)

    def __init__(self, img_w, img_h):
        self.img_w, self.img_h = img_w, img_h
        self.cols = img_w // self.CELL_SIZE + 1
        self.rows = img_h // self.CELL_SIZE + 1
        self.grid = [[GridCell() for _ in range(self.cols)] for _ in range(self.rows)]
        self.candidates: List[AlertCandidate] = []
        self.cooldowns: List[Tuple[float,float,float]] = []
        self._alerted_zones: List[Tuple[float,float,float]] = []  # (cx, cy, until) 已报警区域
        self._pending_clear: List[Tuple[int,int]] = []
        self._prev: dict = {}

    def update(self, tracks: List[dict]) -> List[AlertCandidate]:
        now = time.time()
        new_alerts = []

        # 1. 标记活跃网格
        active = np.zeros((self.rows, self.cols), dtype=bool)

        for t in tracks:
            tid, cx, cy, w, h = t['tid'], t['cx'], t['cy'], t['w'], t['h']
            v = 999.0
            if tid in self._prev:
                px, py = self._prev[tid]
                v = math.sqrt((cx-px)**2 + (cy-py)**2)
            self._prev[tid] = (cx, cy)

            gx, gy = int(cx/self.CELL_SIZE), int(cy/self.CELL_SIZE)
            if not (0 <= gx < self.cols and 0 <= gy < self.rows):
                continue

            r = int(max(w, h) / 2 / self.CELL_SIZE) + 1

            if v > self.VELOCITY_THRESH:
                # 快速通行 → 衰减覆盖区域
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        ny, nx = gy+dy, gx+dx
                        if 0 <= ny < self.rows and 0 <= nx < self.cols:
                            self.grid[ny][nx].count = max(0, self.grid[ny][nx].count - 3)
            else:
                # 静止 → 高斯加权累加
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = gy+dy, gx+dx
                        if 0 <= ny < self.rows and 0 <= nx < self.cols:
                            self.grid[ny][nx].count = min(
                                self.MAX_COUNT, self.grid[ny][nx].count + self.GAUSS_3x3[dy+1, dx+1])
                            if self.grid[ny][nx].count >= self.HOLD_THRESHOLD:
                                self.grid[ny][nx].was_hot = True
                            active[ny, nx] = True

                # 按目标大小扩展
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if abs(dy) <= 1 and abs(dx) <= 1:
                            continue
                        ny, nx = gy+dy, gx+dx
                        if 0 <= ny < self.rows and 0 <= nx < self.cols:
                            d = math.sqrt(dx*dx + dy*dy)
                            wgt = int(1 / (1 + d))
                            self.grid[ny][nx].count = min(
                                self.MAX_COUNT, self.grid[ny][nx].count + wgt)
                            if self.grid[ny][nx].count >= self.HOLD_THRESHOLD:
                                self.grid[ny][nx].was_hot = True
                            active[ny, nx] = True

        # 2. 梯度衰减 + 消警检测（只设 was_hot=True，不触发 CLEARED）
        for y in range(self.rows):
            for x in range(self.cols):
                cell = self.grid[y][x]
                if not active[y, x] and cell.count > 0:
                    cell.inactive_frames += 1
                    f = cell.inactive_frames
                    if f <= 50:        rate = 1
                    elif f <= 75:      rate = 2
                    elif f <= 100:     rate = 4
                    else:              rate = 10
                    cell.count = max(0, cell.count - rate)
                elif active[y, x]:
                    cell.inactive_frames = 0

        # 3. 收集热点
        hot_cells = [(x, y) for y in range(self.rows) for x in range(self.cols)
                     if self.grid[y][x].count >= self.HOLD_THRESHOLD]

        # 4. 聚类
        merged = self._cluster(hot_cells)

        # 5. 候选框管理 (含锁定冷却 + 消警检测)
        for (mcx, mcy, mw, mh) in merged:
            existing = self._find_candidate(mcx, mcy)
            if existing:
                existing.last_updated = now
                existing.w, existing.h = mw, mh
                # 锁定中 → 递减
                if existing.locked_frames > 0:
                    existing.locked_frames -= 1
                # 可触发条件：未锁定 且 未报警 且 未被拦截
                if existing.locked_frames <= 0 and not existing.alerted and not existing.require_drop_below_20:
                    if not existing.qwen_pending:
                        new_alerts.append(existing)
            elif not self._is_cooldown(mcx, mcy) and not self._is_alerted_zone(mcx, mcy):
                cand = AlertCandidate(cx=mcx, cy=mcy, w=mw, h=mh,
                                      first_seen=now, last_updated=now)
                self.candidates.append(cand)
                # 标记覆盖网格为 hot，允许后续 CLEARED
                for gx, gy in [(int(xx/self.CELL_SIZE), int(yy/self.CELL_SIZE))
                               for xx in range(int(mcx-mw/2), int(mcx+mw/2), self.CELL_SIZE)
                               for yy in range(int(mcy-mh/2), int(mcy+mh/2), self.CELL_SIZE)]:
                    if 0 <= gy < self.rows and 0 <= gx < self.cols:
                        self.grid[gy][gx].was_hot = True
                new_alerts.append(cand)

        self._reap_candidates(now)

        # 6. 基于候选框的消警检测
        for cand in self.candidates:
            if cand.alerted and not cand.qwen_pending and cand.locked_frames <= 0:
                gx1 = max(0, int((cand.cx - cand.w/2) / self.CELL_SIZE))
                gy1 = max(0, int((cand.cy - cand.h/2) / self.CELL_SIZE))
                gx2 = min(self.cols, int((cand.cx + cand.w/2) / self.CELL_SIZE) + 1)
                gy2 = min(self.rows, int((cand.cy + cand.h/2) / self.CELL_SIZE) + 1)
                mc = max(self.grid[gy][gx].count
                         for gy in range(gy1, gy2) for gx in range(gx1, gx2))
                if mc < 30:
                    new_alerts.append(cand)

        return new_alerts

    def _process_clears(self) -> List[AlertCandidate]:
        result = []
        for gx, gy in self._pending_clear:
            cx, cy = gx * self.CELL_SIZE, gy * self.CELL_SIZE
            cand = self._find_candidate(cx, cy)
            if cand and not cand.qwen_pending and cand.locked_frames <= 0:
                result.append(cand)
            elif cand:
                print(f"  [CLEAR skip] candidate at ({cx:.0f},{cy:.0f}) qwen_pending={cand.qwen_pending}")
            else:
                print(f"  [CLEAR miss] no candidate at ({cx:.0f},{cy:.0f})")
        self._pending_clear.clear()
        return result

    def _cluster(self, hot):
        if not hot: return []
        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for x, y in hot: mask[y, x] = 255
        mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=1)
        ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in ctrs:
            if cv2.contourArea(c) < 2: continue
            x, y, cw, ch = cv2.boundingRect(c)
            out.append(((x+cw/2)*self.CELL_SIZE, (y+ch/2)*self.CELL_SIZE,
                       cw*self.CELL_SIZE, ch*self.CELL_SIZE))
        return out

    def _find_candidate(self, cx, cy):
        for c in self.candidates:
            if abs(c.cx-cx) < self.CLUSTER_DIST and abs(c.cy-cy) < self.CLUSTER_DIST:
                return c
        return None

    def _reap_candidates(self, now):
        self.candidates = [c for c in self.candidates if now - c.last_updated < 300]

    def _is_cooldown(self, cx, cy):
        now = time.time()
        self.cooldowns = [(cx, cy, u) for cx, cy, u in self.cooldowns if now < u]
        return any(abs(cx-px) < self.CLUSTER_DIST and abs(cy-py) < self.CLUSTER_DIST
                   for px, py, _ in self.cooldowns)

    def _is_alerted_zone(self, cx, cy):
        now = time.time()
        self._alerted_zones = [(cx, cy, u) for cx, cy, u in self._alerted_zones if now < u]
        return any(abs(cx-px) < self.CLUSTER_DIST and abs(cy-py) < self.CLUSTER_DIST
                   for px, py, _ in self._alerted_zones)

    def add_cooldown(self, cx, cy):
        self.cooldowns.append((cx, cy, time.time() + self.COOLDOWN_TTL))

    def get_hot_mask(self):
        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for y in range(self.rows):
            for x in range(self.cols):
                if self.grid[y][x].count >= self.HOLD_THRESHOLD:
                    mask[y, x] = 255
        return mask
