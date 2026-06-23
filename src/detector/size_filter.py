"""尺寸过滤器 —— 基于类别阈值丢弃过小目标"""

from dataclasses import dataclass
from typing import Dict, List

from src.detector.models import Detection


@dataclass
class SizeThreshold:
    min_height: int = 0
    min_area: int = 0


class SizeFilter:
    """按类别最小高度/面积过滤检测结果。

    用法:
        thresholds = {
            "person": SizeThreshold(35, 1000),
            "car": SizeThreshold(22, 1500),
        }
        f = SizeFilter(thresholds)
        passed = f.filter(detections)
    """

    def __init__(self, thresholds: Dict[str, SizeThreshold]):
        """
        参数:
            thresholds: {class_name: SizeThreshold}，键使用与 Detection.class_name 一致的名称
        """
        self.thresholds = thresholds

    def filter(self, detections: List[Detection]) -> List[Detection]:
        """过滤并返回通过尺寸阈值的检测。

        未通过的目标被丢弃，通过的标记 is_size_valid = True。
        """
        passed = []
        for det in detections:
            th = self.thresholds.get(det.class_name, None)
            if th is None:
                # 未配置阈值的类别默认通过
                det.is_size_valid = True
                passed.append(det)
                continue

            if det.height >= th.min_height and det.area >= th.min_area:
                det.is_size_valid = True
                passed.append(det)
            else:
                det.is_size_valid = False
                # 不加入 passed 列表，即丢弃
        return passed

    def is_valid(self, detection: Detection) -> bool:
        """单个检测是否通过（用于防闪烁等二次判定）。"""
        th = self.thresholds.get(detection.class_name, None)
        if th is None:
            return True
        return detection.height >= th.min_height and detection.area >= th.min_area

    @classmethod
    def from_config(cls, thresholds_raw: dict, scale: float = 1.0) -> "SizeFilter":
        """从配置字典创建（兼容 config/default.yaml 格式）。

        参数:
            thresholds_raw: {"person": {"min_height": 35, "min_area": 1000}, ...}
            scale: 缩放系数，防闪烁用 0.8，正常用 1.0
        """
        thresholds = {}
        for cls_name, vals in thresholds_raw.items():
            thresholds[cls_name] = SizeThreshold(
                min_height=int(vals["min_height"] * scale),
                min_area=int(vals["min_area"] * scale),
            )
        return cls(thresholds)
