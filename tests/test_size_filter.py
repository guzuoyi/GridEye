"""测试：SizeFilter —— 尺寸过滤逻辑、from_config"""

import pytest
from src.detector.models import Detection
from src.detector.size_filter import SizeFilter, SizeThreshold


# ---- fixtures ---------------------------------------------------------------

@pytest.fixture
def thresholds():
    return {
        "person": SizeThreshold(min_height=35, min_area=1000),
        "car": SizeThreshold(min_height=22, min_area=1500),
    }


@pytest.fixture
def filter_obj(thresholds):
    return SizeFilter(thresholds)


# ---- tests ------------------------------------------------------------------

class TestSizeFilter:
    """尺寸过滤核心逻辑"""

    def test_passing_detection(self, filter_obj):
        """通过阈值的目标被保留。"""
        d = Detection("car", 0.8, (100, 100, 300, 200))  # h=100, area=20000
        d.height = 100
        d.area = 20000
        result = filter_obj.filter([d])
        assert len(result) == 1
        assert result[0].is_size_valid is True

    def test_filtered_out_height(self, filter_obj):
        """高度不足的目标被丢弃。"""
        d = Detection("car", 0.8, (100, 100, 120, 121))  # h=21 < 22
        d.height = 21
        d.area = 21 * 20
        result = filter_obj.filter([d])
        assert len(result) == 0

    def test_filtered_out_area(self, filter_obj):
        """面积不足的目标被丢弃。"""
        d = Detection("car", 0.8, (100, 100, 110, 130))  # h=30 ok, area=300<1500
        d.height = 30
        d.area = 300
        result = filter_obj.filter([d])
        assert len(result) == 0

    def test_threshold_exact_boundary(self, filter_obj):
        """刚好等于阈值 → 通过。"""
        d = Detection("car", 0.8, (100, 100, 200, 122))  # h=22 exactly
        d.height = 22
        d.area = 100 * 22  # 2200 > 1500
        result = filter_obj.filter([d])
        assert len(result) == 1

    def test_unconfigured_class_passes(self, filter_obj):
        """未配置阈值的类别默认通过。"""
        d = Detection("two-wheeler", 0.8, (100, 100, 110, 110))  # tiny
        result = filter_obj.filter([d])
        assert len(result) == 1

    def test_is_valid_method(self, filter_obj):
        """is_valid() 方法用于防闪烁判定。"""
        assert filter_obj.is_valid(Detection("car", 0.8, (100, 100, 300, 200))) is True
        assert filter_obj.is_valid(Detection("car", 0.8, (100, 100, 110, 111))) is False

    def test_mixed_batch(self, filter_obj):
        """混合通过/不通过的目标。"""
        detections = [
            Detection("car", 0.9, (10, 10, 100, 50)),    # h=40, area=3600 → pass
            Detection("car", 0.9, (10, 10, 15, 15)),     # too small → drop
            Detection("person", 0.9, (10, 10, 100, 50)), # h=40, area=3600 → pass
        ]
        for d in detections:
            d.height = d.bbox[3] - d.bbox[1]
            d.area = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
        result = filter_obj.filter(detections)
        assert len(result) == 2


class TestFromConfig:
    """from_config 工厂方法"""

    def test_from_config_no_scale(self):
        raw = {
            "person": {"min_height": 35, "min_area": 1000},
            "car": {"min_height": 22, "min_area": 1500},
        }
        f = SizeFilter.from_config(raw, scale=1.0)
        assert f.thresholds["person"].min_height == 35
        assert f.thresholds["person"].min_area == 1000

    def test_from_config_with_scale(self):
        """防闪烁 0.8× 缩放。"""
        raw = {
            "person": {"min_height": 35, "min_area": 1000},
            "car": {"min_height": 22, "min_area": 1500},
        }
        f = SizeFilter.from_config(raw, scale=0.8)
        assert f.thresholds["person"].min_height == 28    # 35*0.8
        assert f.thresholds["person"].min_area == 800     # 1000*0.8
        assert f.thresholds["car"].min_height == 17       # 22*0.8=17.6→17
        assert f.thresholds["car"].min_area == 1200       # 1500*0.8
