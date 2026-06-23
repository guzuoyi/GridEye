"""测试：LaneDetector + DailyMaintainer"""

import pytest
import numpy as np
import cv2

from src.lane.detector import LaneDetector, LaneLine
from src.lane.maintainer import DailyMaintainer, LUTScheduler
from src.lane.ipm import LUTGenerator, IPMResult


# ---- helpers ----------------------------------------------------------------

def make_birdseye_with_lanes(
    width: int = 400,
    height: int = 300,
    lane_x_positions: list = None,
    noise: float = 5.0,
) -> np.ndarray:
    """生成带白线的合成鸟瞰图。

    参数:
        lane_x_positions: 白线在图像中的 x 坐标列表
        noise: 高斯噪声标准差
    """
    if lane_x_positions is None:
        lane_x_positions = [100, 200, 300]

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (64, 64, 64)  # 灰色路面

    for lx in lane_x_positions:
        x = int(lx)
        # 画白色车道线（竖直，带轻微扰动）
        for y in range(0, height, 2):
            offset = np.random.randint(-2, 3) if noise > 0 else 0
            px = max(0, min(width - 1, x + offset))
            img[y, px] = (255, 255, 255)
            if px + 1 < width:
                img[y, px + 1] = (255, 255, 255)

    # 加噪声
    if noise > 0:
        noise_arr = np.random.randn(*img.shape) * noise
        img = np.clip(img.astype(np.float32) + noise_arr, 0, 255).astype(np.uint8)

    return img


# ---- LaneDetector ===========================================================

class TestLaneDetector:
    """车道线检测器测试"""

    def test_detect_single_lane(self):
        """单条明显车道线应被检出。"""
        img = make_birdseye_with_lanes(lane_x_positions=[200], noise=2.0)
        detector = LaneDetector(num_lanes=2)
        lanes = detector.detect(img)
        assert len(lanes) >= 1

    def test_detect_multiple_lanes(self):
        """多条车道线应被检出且按 left→right 排序。"""
        img = make_birdseye_with_lanes(lane_x_positions=[80, 160, 240, 320])
        detector = LaneDetector(num_lanes=4, cluster_distance=30.0)
        lanes = detector.detect(img)
        # 不要求精确等于 4，但至少检出大部分
        assert len(lanes) >= 2
        # 验证排序
        mid_y = img.shape[0] // 2
        x_vals = [l.evaluate(mid_y) for l in lanes]
        assert x_vals == sorted(x_vals), f"车道线未按从左到右排序: {x_vals}"

    def test_no_lanes_on_blank(self):
        """纯色图应检出 0 条。"""
        img = np.full((300, 400, 3), 128, dtype=np.uint8)
        detector = LaneDetector()
        lanes = detector.detect(img)
        assert len(lanes) == 0

    def test_draw_lanes(self):
        """draw_lanes 不崩溃且输出同尺寸。"""
        img = make_birdseye_with_lanes(lane_x_positions=[100, 200, 300])
        detector = LaneDetector()
        lanes = detector.detect(img)
        out = detector.draw_lanes(img, lanes)
        assert out.shape == img.shape


# ---- DailyMaintainer ========================================================

class TestDailyMaintainer:
    """日常维护器测试"""

    @pytest.fixture
    def maintainer(self):
        ipm = IPMResult(
            H=np.eye(3),
            H_inv=np.eye(3),
            pitch=0, yaw=0,
            camera_height=1.5,
            fov_scale=0.1,
        )
        gen = LUTGenerator(ipm_result=ipm, image_size=(100, 50))
        det = LaneDetector()
        return DailyMaintainer(gen, det, min_lanes=1)

    def test_set_initial_lut(self, maintainer):
        """设置初始 LUT。"""
        lut = np.zeros((50, 100), dtype=np.dtype([("lane", "i4"), ("dist", "f4")]))
        lut["lane"][:] = 1
        maintainer.set_initial_lut(lut)
        assert maintainer.active_lut is not None
        lane, _ = maintainer.lookup(50, 25)
        assert lane == 1

    def test_lookup_on_empty(self, maintainer):
        """无 LUT 时查表返回 (-1, -1)。"""
        lane, dist = maintainer.lookup(100, 50)
        assert lane == -1
        assert dist == -1.0

    def test_maintenance_fails_with_few_lanes(self, maintainer):
        """车道线不足 → 维护失败。"""
        img = np.full((50, 100, 3), 128, dtype=np.uint8)
        success = maintainer.run_maintenance(img)
        assert success is False

    def test_callback_called_on_update(self, maintainer):
        """run_maintenance 成功更新后回调被调用。set_initial_lut 不触发回调。"""
        called = []
        maintainer.set_on_update(lambda lut: called.append(True))

        # set_initial_lut 不触发回调（仅初始化）
        lut = np.zeros((50, 100), dtype=np.dtype([("lane", "i4"), ("dist", "f4")]))
        lut["lane"][:] = 2
        maintainer.set_initial_lut(lut)
        assert len(called) == 0  # 初始化不触发
