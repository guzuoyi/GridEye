"""车道线自动检测 —— 用于日常查找表维护

流程: IPM 变换 → 灰度 → 高斯模糊 → Canny 边缘 → HoughLinesP → 聚类拟合
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class LaneLine:
    """车道线"""
    lane_id: int            # 车道线编号（从左到右 0,1,2...）
    polynomial: np.ndarray  # 二次多项式系数 [a, b, c] → x = a*y² + b*y + c
    points: np.ndarray      # 拟合点集 (N, 2)
    is_valid: bool = True

    def evaluate(self, y: float) -> float:
        """给定 y 坐标返回拟合的 x 坐标。"""
        a, b, c = self.polynomial
        return a * y * y + b * y + c


class LaneDetector:
    """基于 Hough 变换的车道线检测器。

    用法:
        detector = LaneDetector()
        lanes = detector.detect(birdseye_image)  # 输入鸟瞰图
    """

    def __init__(
        self,
        num_lanes: int = 4,                     # 期望检测的车道线数量
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 50,              # Hough 累加器阈值
        hough_min_line_length: int = 100,
        hough_max_line_gap: int = 50,
        cluster_distance: float = 50.0,         # 线段聚类距离 (像素)
        poly_order: int = 2,                    # 多项式阶数（2=二次）
    ):
        self.num_lanes = num_lanes
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.cluster_distance = cluster_distance
        self.poly_order = poly_order

    def detect(self, birdseye: np.ndarray) -> List[LaneLine]:
        """从鸟瞰图检测车道线。

        参数:
            birdseye: IPM 变换后的鸟瞰图 (H, W, 3) BGR

        返回:
            检测到的车道线列表，按从左到右排序
        """
        gray = cv2.cvtColor(birdseye, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Hough 线段检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        if lines is None or len(lines) == 0:
            logger.warning("未检测到车道线段")
            return []

        # 收集所有线段端点
        segments = lines[:, 0, :]  # (N, 4) → (x1,y1,x2,y2)
        all_points = []
        for seg in segments:
            all_points.append([seg[0], seg[1]])
            all_points.append([seg[2], seg[3]])
        all_points = np.array(all_points, dtype=np.float32)

        # 按 x 坐标聚类 → 区分不同车道线
        clusters = self._cluster_by_x(all_points)

        # 对每个聚类 RANSAC 拟合多项式
        lanes = []
        for i, points in enumerate(clusters):
            if len(points) < 10:
                continue
            poly = self._ransac_fit(points)
            if poly is not None:
                lanes.append(LaneLine(
                    lane_id=i,
                    polynomial=poly,
                    points=points,
                ))

        # 按 x 坐标排序（从左到右）
        lanes.sort(key=lambda l: l.evaluate(birdseye.shape[0] // 2))
        for i, lane in enumerate(lanes):
            lane.lane_id = i

        logger.info(f"检测到 {len(lanes)} 条车道线")
        return lanes

    def _cluster_by_x(
        self, points: np.ndarray
    ) -> List[np.ndarray]:
        """按 x 坐标对点进行简单聚类。"""
        sorted_pts = points[points[:, 0].argsort()]
        if len(sorted_pts) == 0:
            return []

        clusters = []
        current = [sorted_pts[0]]

        for pt in sorted_pts[1:]:
            if abs(pt[0] - current[-1][0]) < self.cluster_distance:
                current.append(pt)
            else:
                if len(current) >= 5:
                    clusters.append(np.array(current))
                current = [pt]

        if len(current) >= 5:
            clusters.append(np.array(current))

        return clusters

    def _ransac_fit(self, points: np.ndarray) -> Optional[np.ndarray]:
        """RANSAC 拟合二次多项式 x = a*y² + b*y + c。

        返回:
            多项式系数 [a, b, c] 或 None
        """
        if len(points) < 10:
            return None

        best_coeffs = None
        best_inliers = 0
        max_iter = min(200, len(points) * 2)
        threshold = 15.0  # 像素

        for _ in range(max_iter):
            if len(points) < 3:
                break
            idx = np.random.choice(len(points), min(3, len(points)), replace=False)
            sample = points[idx]
            try:
                coeffs = np.polyfit(sample[:, 1], sample[:, 0], self.poly_order)
            except np.linalg.LinAlgError:
                continue

            y_all = points[:, 1]
            x_pred = np.polyval(coeffs, y_all)
            errors = np.abs(points[:, 0] - x_pred)
            inliers = np.sum(errors < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_coeffs = coeffs

        if best_coeffs is not None and best_inliers > len(points) * 0.5:
            # 用所有内点重新拟合
            y_all = points[:, 1]
            x_pred = np.polyval(best_coeffs, y_all)
            errors = np.abs(points[:, 0] - x_pred)
            inlier_mask = errors < threshold
            if np.sum(inlier_mask) >= 3:
                best_coeffs = np.polyfit(
                    points[inlier_mask, 1],
                    points[inlier_mask, 0],
                    self.poly_order,
                )
            return best_coeffs

        return None

    def draw_lanes(
        self,
        image: np.ndarray,
        lanes: List[LaneLine],
        color_map: Optional[dict] = None,
    ) -> np.ndarray:
        """在图像上绘制检测到的车道线。"""
        out = image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        for lane in lanes:
            color = colors[lane.lane_id % len(colors)]
            if color_map:
                color = color_map.get(lane.lane_id, color)

            y_vals = np.linspace(0, out.shape[0] - 1, 100)
            x_vals = lane.evaluate(y_vals)
            for i in range(len(y_vals) - 1):
                pt1 = (int(x_vals[i]), int(y_vals[i]))
                pt2 = (int(x_vals[i + 1]), int(y_vals[i + 1]))
                if 0 <= pt1[0] < out.shape[1] and 0 <= pt2[0] < out.shape[1]:
                    cv2.line(out, pt1, pt2, color, 2)

            # 标车道号
            mid = len(y_vals) // 2
            cv2.putText(
                out, str(lane.lane_id),
                (int(x_vals[mid]), int(y_vals[mid])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        return out
