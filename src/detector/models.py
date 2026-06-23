"""检测结果数据模型

坐标约定: 所有 bbox 和 center 基于 1920x1080 原始分辨率。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    """单帧检测结果。"""
    class_name: str              # car / truck / person / two-wheeler
    confidence: float            # YOLO 置信度 0~1
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) 原始分辨率
    center: tuple[float, float] = (0.0, 0.0)  # (cx, cy) 中心点（自动计算）

    # 尺寸信息（预计算，用于过滤和防闪烁）
    height: int = 0
    area: int = 0

    def __post_init__(self):
        if self.center == (0.0, 0.0):
            cx = (self.bbox[0] + self.bbox[2]) / 2.0
            cy = (self.bbox[1] + self.bbox[3]) / 2.0
            self.center = (cx, cy)
        if self.height == 0:
            self.height = self.bbox[3] - self.bbox[1]
        if self.area == 0:
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            self.area = w * h

    @property
    def is_size_valid(self) -> Optional[bool]:
        """尺寸是否通过过滤器（由 SizeFilter 设置）。
        未经过滤时为 None。
        """
        return getattr(self, "_size_valid", None)

    @is_size_valid.setter
    def is_size_valid(self, val: bool):
        self._size_valid = val
