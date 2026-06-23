"""测试：坐标映射工具"""

from src.utils.coords import (
    rescale_bbox_stretch,
    bbox_center,
    bbox_height,
    bbox_area,
    euclidean_distance,
)


class TestRescaleBboxStretch:
    """等比缩放 (Stretch) 坐标映射测试。"""

    def test_maps_correctly_1920x1080(self):
        """640x640 中心框映射回 1920x1080。"""
        bbox_640 = (160, 160, 480, 480)  # 640 中心 320x320 区域
        x1, y1, x2, y2 = rescale_bbox_stretch(bbox_640)
        # x: 160*3=480, 480*3=1440
        # y: 160*1.6875=270, 480*1.6875=810
        assert x1 == 480
        assert y1 == 270
        assert x2 == 1440
        assert y2 == 810

    def test_corner_case_clamped(self):
        """边界坐标裁剪到有效范围。"""
        bbox_640 = (-10, -10, 650, 650)  # 出界
        x1, y1, x2, y2 = rescale_bbox_stretch(bbox_640)
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 1919
        assert y2 <= 1079

    def test_full_frame(self):
        """全幅映射。"""
        bbox_640 = (0, 0, 639, 639)
        x1, y1, x2, y2 = rescale_bbox_stretch(bbox_640)
        # x2 = int(round(639*1920/640)) ≈ 1917
        assert x1 == 0
        assert y1 == 0
        assert abs(x2 - 1917) <= 1
        assert abs(y2 - 1078) <= 1


class TestBboxUtils:
    """bbox 辅助函数测试。"""

    def test_center(self):
        cx, cy = bbox_center((100, 200, 300, 400))
        assert cx == 200.0
        assert cy == 300.0

    def test_height(self):
        assert bbox_height((0, 10, 100, 60)) == 50

    def test_area(self):
        # 宽 200, 高 100
        assert bbox_area((100, 200, 300, 300)) == 20000


class TestEuclideanDistance:
    """距离函数测试。"""

    def test_same_point(self):
        assert euclidean_distance((10, 20), (10, 20)) == 0.0

    def test_horizontal(self):
        assert euclidean_distance((0, 0), (30, 0)) == 30.0

    def test_345_triangle(self):
        assert euclidean_distance((0, 0), (3, 4)) == 5.0
