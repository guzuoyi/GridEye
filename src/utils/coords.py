"""坐标约定与映射工具

坐标约定: (x, y) = (col, row)，原点左上角，x 向右为正，y 向下为正。
所有坐标基于 1920x1080 原始分辨率。
"""

from dataclasses import dataclass


# ---- 坐标映射 ----------------------------------------------------------------

def rescale_bbox_stretch(
    bbox_640: tuple[int, int, int, int],
    orig_w: int = 1920,
    orig_h: int = 1080,
    yolo_size: int = 640,
) -> tuple[int, int, int, int]:
    """将 YOLO 640x640 坐标等比缩放回原始分辨率。

    使用 Stretch 模式（无 letterbox/padding）：
        x_orig = x_640 × (orig_w / yolo_size)
        y_orig = y_640 × (orig_h / yolo_size)

    参数:
        bbox_640: YOLO 输出检测框 (x1, y1, x2, y2) 基于 640x640
        orig_w: 原始图像宽度
        orig_h: 原始图像高度
        yolo_size: YOLO 输入尺寸

    返回:
        (x1, y1, x2, y2) 基于原始分辨率，值已转为 int 并裁剪到有效范围
    """
    x1, y1, x2, y2 = bbox_640
    scale_x = orig_w / yolo_size
    scale_y = orig_h / yolo_size

    x1_orig = int(round(x1 * scale_x))
    y1_orig = int(round(y1 * scale_y))
    x2_orig = int(round(x2 * scale_x))
    y2_orig = int(round(y2 * scale_y))

    # 裁剪到图像有效范围
    x1_orig = max(0, min(x1_orig, orig_w - 1))
    y1_orig = max(0, min(y1_orig, orig_h - 1))
    x2_orig = max(0, min(x2_orig, orig_w - 1))
    y2_orig = max(0, min(y2_orig, orig_h - 1))

    return (x1_orig, y1_orig, x2_orig, y2_orig)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """计算检测框中心点坐标。"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def bbox_height(bbox: tuple[int, int, int, int]) -> int:
    """计算检测框高度（像素）。"""
    _, y1, _, y2 = bbox
    return y2 - y1


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    """计算检测框面积（平方像素）。"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return w * h


def euclidean_distance(
    p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """两点欧氏距离。"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5
