"""测试：CameraCalibrator + IPM/LUT —— save/load、查表、并发安全"""

import pytest
import numpy as np
import tempfile
import os

from src.lane.calibration import CameraCalibrator, CalibrationResult
from src.lane.ipm import (
    LUTGenerator,
    LUTLookup,
    IPMResult,
    LUT_DTYPE,
)


# ===== CameraCalibrator =======================================================

class TestCalibrationSaveLoad:
    """标定结果 save/load 循环"""

    def test_save_load_roundtrip(self):
        r = CalibrationResult(
            K=np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]], dtype=np.float64),
            dist=np.array([[-0.1, 0.05, 0, 0, 0]], dtype=np.float64),
            rms_error=0.35,
            image_size=(1920, 1080),
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            CameraCalibrator.save(r, path)
            r2 = CameraCalibrator.load(path)
            assert np.allclose(r.K, r2.K)
            assert np.allclose(r.dist, r2.dist)
            assert r.rms_error == pytest.approx(r2.rms_error, abs=0.01)
            assert r.image_size == r2.image_size
        finally:
            os.unlink(path)

    def test_not_enough_images(self):
        """不足 3 张图像应报错。"""
        calib = CameraCalibrator(pattern=(9, 6))
        # 创建假帧：纯色图，无棋盘格
        fake = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            calib.calibrate_from_frames([fake, fake])  # 只 2 帧

    def test_no_chessboard_detected(self):
        """无棋盘格 → 报错。"""
        calib = CameraCalibrator(pattern=(9, 6))
        fakes = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(5)]
        with pytest.raises(RuntimeError, match="角点"):
            calib.calibrate_from_frames(fakes)


# ===== LUTGenerator & LUTLookup ===============================================

@pytest.fixture
def mock_ipm_result():
    """构造一个简单的 IPM 结果，用于测试 LUT 生成。"""
    # 恒等映射 H = I (每像素对应自身，便于验证)
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return IPMResult(
        H=H,
        H_inv=np.linalg.inv(H),
        pitch=0.0,
        yaw=0.0,
        camera_height=1.5,
        fov_scale=0.1,
    )


class TestLUTGenerator:
    """LUT 生成测试"""

    def test_generate_small_lut(self, mock_ipm_result):
        """生成小尺寸 LUT 并验证结构。"""
        gen = LUTGenerator(
            ipm_result=mock_ipm_result,
            lane_width_m=3.75,
            image_size=(100, 50),  # 小尺寸测试
            max_distance_m=100,
        )

        # 简单的车道线：x = 25 和 x = 75 作为车道线
        from src.lane.detector import LaneLine
        lanes = [
            LaneLine(lane_id=0, polynomial=np.array([0, 0, 25]), points=np.array([])),
            LaneLine(lane_id=1, polynomial=np.array([0, 0, 75]), points=np.array([])),
        ]

        lut = gen.generate(np.zeros((50, 100, 3), dtype=np.uint8), lanes)

        assert lut.shape == (50, 100)
        assert lut.dtype == LUT_DTYPE
        assert "lane" in lut.dtype.names
        assert "dist" in lut.dtype.names

    def test_lut_write_read(self, mock_ipm_result, tmp_path):
        """LUT 写入磁盘再读回。"""
        gen = LUTGenerator(
            ipm_result=mock_ipm_result,
            lane_width_m=3.75,
            image_size=(100, 50),
            max_distance_m=100,
        )

        from src.lane.detector import LaneLine
        lanes = [
            LaneLine(lane_id=0, polynomial=np.array([0, 0, 30]), points=np.array([])),
        ]
        lut = gen.generate(np.zeros((50, 100, 3), dtype=np.uint8), lanes)

        path = str(tmp_path / "test_lut.npy")
        gen.save_lut(lut, path)
        loaded = gen.load_lut(path)

        assert np.array_equal(lut["lane"], loaded["lane"])
        assert np.allclose(lut["dist"], loaded["dist"])


class TestLUTLookup:
    """LUT 查表测试"""

    def test_lookup_static(self):
        """用预构造 LUT 验证查表正确。"""
        lut = np.zeros((10, 20), dtype=LUT_DTYPE)
        lut["lane"][3, 5] = 1
        lut["dist"][3, 5] = 42.5

        lane, dist = LUTLookup.lookup_point(lut, 5, 3)
        assert lane == 1
        assert dist == pytest.approx(42.5)

    def test_lookup_out_of_bounds(self):
        """越界查询应返回 (-1, -1)。"""
        lut = np.zeros((10, 20), dtype=LUT_DTYPE)
        lane, dist = LUTLookup.lookup_point(lut, 999, 999)
        assert lane == -1
        assert dist == -1.0

    def test_lookup_center_point(self):
        """查目标中心点。"""
        lut = np.zeros((100, 200), dtype=LUT_DTYPE)
        lut["lane"][50, 100] = 2
        lut["dist"][50, 100] = 30.0

        lane, dist = LUTLookup.lookup(lut, (100.0, 50.0))
        assert lane == 2
        assert dist == pytest.approx(30.0)

    def test_lookup_uses_int_coords(self):
        """中心点 float → int 截断后查表。"""
        lut = np.zeros((10, 20), dtype=LUT_DTYPE)
        lut["lane"][4, 7] = 3
        lut["dist"][4, 7] = 15.0

        lane, dist = LUTLookup.lookup(lut, (7.8, 4.2))
        assert lane == 3
