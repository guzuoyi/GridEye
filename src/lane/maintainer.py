"""查找表日常维护 —— 每日定时自动检测车道线并更新查找表"""

import time
import threading
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
from logging import getLogger

import numpy as np

from src.lane.detector import LaneDetector
from src.lane.ipm import LUTGenerator, LUTLookup, IPMResult, LUT_DTYPE

logger = getLogger(__name__)


class DailyMaintainer:
    """查找表日常维护器。

    负责：抓取当前帧 → 车道线检测 → 生成新 LUT → 验证 → 原子替换

    用法:
        maintainer = DailyMaintainer(lut_generator, lane_detector)
        maintainer.set_on_update(callback)  # 可选：替换后的回调
        success = maintainer.run_maintenance(frame)
    """

    def __init__(
        self,
        lut_generator: LUTGenerator,
        lane_detector: LaneDetector,
        min_lanes: int = 2,  # 最少需要的车道线数量
    ):
        self.lut_generator = lut_generator
        self.lane_detector = lane_detector
        self.min_lanes = min_lanes
        self._active_lut: Optional[np.ndarray] = None
        self._on_update_callbacks: list[Callable] = []
        self._lock = threading.Lock()

    # ---- 主动 LUT ------------------------------------------------------------

    @property
    def active_lut(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._active_lut

    def set_on_update(self, callback: Callable[[np.ndarray], None]) -> None:
        """注册 LUT 更新后的回调。"""
        self._on_update_callbacks.append(callback)

    # ---- 维护流程 ------------------------------------------------------------

    def run_maintenance(self, frame: np.ndarray) -> bool:
        """执行一次维护：检测车道线 → 生成新 LUT → 原子替换。

        返回:
            True 表示维护成功并替换了 LUT
        """
        logger.info("开始查找表维护...")

        # Step 1: 检测车道线
        lanes = self.lane_detector.detect(frame)
        if len(lanes) < self.min_lanes:
            logger.warning(
                f"车道线检测不足: 需要 ≥ {self.min_lanes}，实际 {len(lanes)}，跳过更新"
            )
            return False

        # Step 2: 生成新 LUT
        new_lut = self.lut_generator.generate(frame, lanes)

        # Step 3: 验证
        if not self._validate(new_lut):
            logger.error("新 LUT 验证失败，保留旧表")
            return False

        # Step 4: 原子替换 (Immutable Snapshot)
        old_lut = None
        with self._lock:
            old_lut = self._active_lut
            self._active_lut = new_lut

        # Step 5: 通知回调
        for cb in self._on_update_callbacks:
            try:
                cb(new_lut)
            except Exception as e:
                logger.error(f"LUT 更新回调异常: {e}")

        logger.info(f"查找表已更新 (size={new_lut.shape})")
        return True

    def set_initial_lut(self, lut: np.ndarray) -> None:
        """设置初始 LUT（首日标定后调用）。"""
        with self._lock:
            self._active_lut = lut
        logger.info(f"初始查找表已设置 (size={lut.shape})")

    # ---- 查表（线程安全）------------------------------------------------------

    def lookup(self, x: int, y: int) -> tuple:
        """查表：像素坐标 → (车道号, 纵向距离)，线程安全。

        参数:
            x: 列索引 (0~1919)
            y: 行索引 (0~1079)

        返回:
            (lane_number, distance_meters)
        """
        lut = self.active_lut
        if lut is None:
            return (-1, -1.0)
        return LUTLookup.lookup_point(lut, x, y)

    # ---- 验证 ----------------------------------------------------------------

    def _validate(self, lut: np.ndarray) -> bool:
        """验证 LUT 的基本有效性。"""
        if lut is None or lut.size == 0:
            return False
        if lut.shape[0] != 1080 or lut.shape[1] != 1920:
            logger.error(f"LUT 尺寸错误: {lut.shape}")
            return False
        # 检查是否有有效数据（非全零）
        lane_data = lut["lane"]
        if np.all(lane_data == 0):
            logger.error("LUT 车道数据全为零")
            return False
        # 检查距离是否合理
        valid_dist = lut["dist"][lane_data >= 0]
        if len(valid_dist) > 0 and np.max(valid_dist) < 1.0:
            logger.warning("LUT 所有距离 < 1m，可能标定异常")
        return True


class LUTScheduler:
    """查找表每日自动维护调度器。

    在指定时间（如 03:00）触发维护。
    """

    def __init__(
        self,
        maintainer: DailyMaintainer,
        daily_time: str = "03:00",
        frame_provider: Optional[Callable[[], np.ndarray]] = None,
    ):
        """
        参数:
            maintainer: DailyMaintainer 实例
            daily_time: 每日触发时间 "HH:MM"
            frame_provider: 获取当前帧的回调函数
        """
        self.maintainer = maintainer
        self.daily_time = daily_time
        self._hour, self._minute = map(int, daily_time.split(":"))
        self.frame_provider = frame_provider
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动后台调度线程。"""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"LUT 调度器已启动，每日 {self.daily_time} 触发")

    def stop(self) -> None:
        """停止调度线程。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        """调度主循环。"""
        last_check_date = None
        while not self._stop_event.is_set():
            now = time.localtime()
            current_date = (now.tm_year, now.tm_mon, now.tm_mday)

            # 到达触发时间且今天未执行
            if (
                now.tm_hour == self._hour
                and now.tm_min == self._minute
                and current_date != last_check_date
            ):
                if self.frame_provider is not None:
                    frame = self.frame_provider()
                    self.maintainer.run_maintenance(frame)
                last_check_date = current_date

            time.sleep(30)  # 每 30 秒检查一次
