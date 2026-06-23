"""IPM（逆透视变换）矩阵计算 + 查找表生成

基于车道线几何约束（消失点、车道宽度）自动求解外参，
生成 O(1) 查表的像素→(车道号, 纵向距离) 映射。
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
from logging import getLogger

logger = getLogger(__name__)

# 查找表 dtype：每个像素存储 (lane号, 纵向距离_m)
LUT_DTYPE = np.dtype([("lane", "i4"), ("dist", "f4")])


@dataclass
class IPMResult:
    """IPM 标定结果"""
    H: np.ndarray           # 3×3 单应性矩阵 (图像 → 鸟瞰)
    H_inv: np.ndarray       # 逆矩阵 (鸟瞰 → 图像)
    pitch: float            # 俯仰角 (弧度)
    yaw: float              # 偏航角 (弧度)
    camera_height: float    # 相机离地高度 (米)
    fov_scale: float        # 像素→米的缩放因子 (鸟瞰图每像素对应的米数)


class IPMCalibrator:
    """基于车道线几何约束的 IPM 外参标定。

    利用消失点（车道线在图像中的交点）、标准车道宽度 (3.75m)、
    和车道线端点自动求解俯仰角、偏航角、相机高度。

    用法:
        ipm_calib = IPMCalibrator(lane_width_m=3.75)
        result = ipm_calib.calibrate(
            K=camera_K,
            vanishing_point=(vp_x, vp_y),
            lane_endpoints=((x1,y1), (x2,y2)),  # 近端车道线端点
            image_size=(1920, 1080),
            camera_height_guess=6.0,
        )
    """

    def __init__(self, lane_width_m: float = 3.75):
        self.lane_width = lane_width_m  # 标准车道宽度 (米)

    def calibrate(
        self,
        K: np.ndarray,
        vanishing_point: Tuple[float, float],
        lane_endpoints: Tuple[
            Tuple[float, float], Tuple[float, float]
        ],
        image_size: Tuple[int, int],
        camera_height_guess: float = 6.0,
        focal_length_px: Optional[float] = None,
    ) -> IPMResult:
        """计算 IPM 矩阵。

        参数:
            K: 相机内参矩阵 3×3
            vanishing_point: 消失点 (u, v) 在图像中的坐标
            lane_endpoints: 近端路面上一对车道线端点 ((u1,v1), (u2,v2))，
                           两个端点应在同一条水平线上（v1 ≈ v2），
                           且间距对应一条车道宽度 (3.75m)
            image_size: (width, height)
            camera_height_guess: 相机离地高度估计 (米)
            focal_length_px: 焦距 (像素)，若为 None 则从 K 中取

        返回:
            IPMResult
        """
        W, H = image_size
        fx = focal_length_px or K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        vp_x, vp_y = vanishing_point

        # ---- 1. 求解俯仰角 (pitch) 和偏航角 (yaw) ----
        # 消失点偏移图像中心 → 偏航角
        yaw = np.arctan2(vp_x - cx, fx)

        # 俯仰角：消失点的垂直位置
        pitch = np.arctan2(cy - vp_y, fy)

        # ---- 2. 估算相机高度 --- 利用近端车道宽度 ----
        (u1, v1), (u2, v2) = lane_endpoints

        # 近端两点在路面平面上的世界坐标 (假设地面 y=0)
        # 反投影到地面 (z = 0 平面)
        ground_pt1 = self._image_to_ground(
            u1, v1, pitch, yaw, camera_height_guess, K
        )
        ground_pt2 = self._image_to_ground(
            u2, v2, pitch, yaw, camera_height_guess, K
        )

        # 实际测得的车道宽度 vs 预期宽度 → 校正相机高度
        measured_width = np.linalg.norm(ground_pt1 - ground_pt2)
        scale = self.lane_width / max(measured_width, 0.01)
        camera_height = camera_height_guess * scale

        logger.info(
            f"IPM 标定: pitch={np.degrees(pitch):.1f}°, "
            f"yaw={np.degrees(yaw):.1f}°, "
            f"height={camera_height:.2f}m"
        )

        # ---- 3. 构建 IPM 单应性矩阵 ----
        H = self._compute_homography(pitch, yaw, camera_height, K, image_size)
        H_inv = np.linalg.inv(H)

        # 鸟瞰图缩放因子 (米/像素)
        # 取图像底部中心点投影到地面，计算每像素对应的实际距离
        bottom_center = (W // 2, H - 1)
        ground_bc = self._image_to_ground(
            bottom_center[0], bottom_center[1],
            pitch, yaw, camera_height, K,
        )
        # 相邻像素投影后的距离差 → fov_scale
        ground_bc_next = self._image_to_ground(
            bottom_center[0] + 1, bottom_center[1],
            pitch, yaw, camera_height, K,
        )
        fov_scale = abs(ground_bc[0] - ground_bc_next[0])

        return IPMResult(
            H=H,
            H_inv=H_inv,
            pitch=pitch,
            yaw=yaw,
            camera_height=camera_height,
            fov_scale=fov_scale,
        )

    def _image_to_ground(
        self,
        u: float,
        v: float,
        pitch: float,
        yaw: float,
        height: float,
        K: np.ndarray,
    ) -> np.ndarray:
        """将图像坐标反投影到地面 (z=0 平面)，返回世界坐标 (X, Y)。"""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 相机坐标系中的方向向量
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy

        # 旋转：先 pitch 后 yaw
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # 方向向量在相机坐标系中: (x_cam, y_cam, 1)
        # 绕 x 轴旋转 pitch
        # 绕 y 轴旋转 yaw
        # 简化：小角度近似
        dx = x_cam * cos_y + sin_y
        dy = y_cam * cos_p - sin_p
        dz = -y_cam * sin_p + cos_p

        if abs(dz) < 1e-6:
            dz = 1e-6

        t = height / dz if dz < 0 else -height / abs(dz)
        X = dx * t
        Y = dy * t
        return np.array([X, Y], dtype=np.float64)

    def _compute_homography(
        self,
        pitch: float,
        yaw: float,
        height: float,
        K: np.ndarray,
        image_size: Tuple[int, int],
        bev_size: Tuple[int, int] = (400, 800),
        bev_range_m: Tuple[float, float] = (60.0, 20.0),
    ) -> np.ndarray:
        """计算图像→鸟瞰图的单应性矩阵。

        在鸟瞰图中:
        - x 轴: 横向 (车道宽度方向)，范围 ±10m
        - y 轴: 纵向 (前进方向)，范围 0~60m
        """
        W, H = image_size
        bw, bh = bev_size
        range_y, range_x = bev_range_m  # 纵向范围, 横向半宽

        # 鸟瞰图四角对应的世界坐标
        bev_corners_world = np.float32([
            [-range_x, range_y],   # 左上 (左, 远)
            [range_x, range_y],    # 右上 (右, 远)
            [range_x, 0],          # 右下 (右, 近)
            [-range_x, 0],         # 左下 (左, 近)
        ])

        # 鸟瞰图四角在图像中的像素位置
        bev_corners_px = np.float32([
            [0, 0],
            [bw - 1, 0],
            [bw - 1, bh - 1],
            [0, bh - 1],
        ])

        # 世界坐标 → 图像坐标的反向映射
        img_corners = []
        for wx, wy in bev_corners_world:
            u, v = self._ground_to_image(wx, wy, pitch, yaw, height, K)
            img_corners.append([u, v])

        img_corners = np.float32(img_corners)

        # 鸟瞰图像素坐标 → 原始图像像素坐标
        H = cv2.getPerspectiveTransform(bev_corners_px, img_corners)

        return H

    def _ground_to_image(
        self,
        X: float,
        Y: float,
        pitch: float,
        yaw: float,
        height: float,
        K: np.ndarray,
    ) -> Tuple[float, float]:
        """世界地面坐标 (X, Y) 投影到图像坐标 (u, v)。"""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # 地面点在相机坐标系中的位置
        # 先绕 y 轴旋转 -yaw
        xc = X * cos_y - Y * sin_y * sin_p
        yc = Y * cos_p + height
        zc = X * sin_y + Y * cos_y * sin_p

        if abs(zc) < 1e-6:
            zc = 1e-6

        u = fx * xc / zc + cx
        v = fy * yc / zc + cy
        return (u, v)

    @staticmethod
    def save(result: IPMResult, path: str) -> None:
        """保存 IPM 参数到 YAML。"""
        d = {
            "H": result.H.tolist(),
            "H_inv": result.H_inv.tolist(),
            "pitch_rad": float(result.pitch),
            "yaw_rad": float(result.yaw),
            "camera_height_m": float(result.camera_height),
            "fov_scale_m_per_px": float(result.fov_scale),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f)
        logger.info(f"IPM 参数已保存: {path}")

    @staticmethod
    def load(path: str) -> IPMResult:
        """从 YAML 加载 IPM 参数。"""
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return IPMResult(
            H=np.array(d["H"], dtype=np.float64),
            H_inv=np.array(d["H_inv"], dtype=np.float64),
            pitch=d["pitch_rad"],
            yaw=d["yaw_rad"],
            camera_height=d["camera_height_m"],
            fov_scale=d.get("fov_scale_m_per_px", 0.05),
        )


# =============================================================================
# LUTGenerator —— 查找表生成
# =============================================================================

class LUTGenerator:
    """基于 IPM 矩阵和车道线方程生成像素→(车道号, 纵向距离) 查找表。

    查找表存储为 NumPy 结构化数组，shape=(H, W)，每个元素包含：
        lane: 整数车道号 (0=硬路肩, 1,2,3...=行车道)
        dist: 纵向距离 (米)

    用法:
        gen = LUTGenerator(ipm_result, lane_width_m=3.75)
        lanes = [LaneLine(...), ...]
        lut = gen.generate(frame, lanes)
        gen.save_lut(lut, "config/lut.npy")
    """

    def __init__(
        self,
        ipm_result: IPMResult,
        lane_width_m: float = 3.75,
        image_size: Tuple[int, int] = (1920, 1080),
        max_distance_m: float = 200.0,
    ):
        self.ipm = ipm_result
        self.lane_width_m = lane_width_m
        self.image_size = image_size
        self.max_distance_m = max_distance_m

    def generate(
        self, frame: np.ndarray, lanes: list
    ) -> np.ndarray:
        """生成完整查找表。

        参数:
            frame: 当前帧 (用于获取尺寸)
            lanes: LaneLine 列表（至少包含车道边界信息）

        返回:
            lut: shape=(H, W) 的 LUT_DTYPE 数组
        """
        h, w = frame.shape[:2]
        lut = np.zeros((h, w), dtype=LUT_DTYPE)
        lut["lane"] = -1
        lut["dist"] = -1.0

        H_inv = self.ipm.H_inv

        # 为每个像素计算地面位置
        # 创建像素坐标网格
        ys, xs = np.mgrid[0:h, 0:w]
        pixels = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs.ravel())], axis=1)  # (N, 3)

        # IPM 反投影到地面坐标（齐次坐标变换）
        ground = (H_inv @ pixels.T).T  # (N, 3)
        ground[:, 0] /= ground[:, 2]
        ground[:, 1] /= ground[:, 2]

        gx = ground[:, 0].reshape(h, w)  # 地面 x 坐标
        gy = ground[:, 1].reshape(h, w)  # 地面 y 坐标 = 纵向距离

        # 纵向距离
        lut["dist"] = gy.astype("f4")

        # 车道号分配：基于车道线方程划分
        lane_lines = sorted(lanes, key=lambda l: l.evaluate(h // 2) if hasattr(l, 'evaluate') else 0)

        # 为每个像素确定车道号
        for y_idx in range(h):
            # 在中间高度评估车道线位置
            eval_y = float(y_idx)

            # 收集各车道线在此高度的 x 坐标
            boundaries = []
            for i, lane in enumerate(lane_lines):
                if hasattr(lane, 'evaluate'):
                    bx = lane.evaluate(eval_y)
                    boundaries.append(bx)
                else:
                    boundaries.append(float(lane.polynomial[2]))

            # 对每个像素行，按 x 位置分配车道号
            for x_idx in range(w):
                px = float(x_idx)
                lane_id = self._assign_lane(px, boundaries)
                lut["lane"][y_idx, x_idx] = lane_id

        # 距离过滤：超过最大距离的标记为无效
        mask = lut["dist"] > self.max_distance_m
        lut["lane"][mask] = -1
        lut["dist"][mask] = -1.0

        logger.info(
            f"LUT 生成完成: {w}×{h}, "
            f"有效像素 {np.sum(lut['lane'] >= 0)}"
        )
        return lut

    def _assign_lane(self, px: float, boundaries: list) -> int:
        """根据像素 x 坐标和车道边界分配车道号。

        车道号约定：
            0 = 硬路肩（最左边界之左）
            1 = 第一行车道（边界[0] 和 边界[1] 之间）
            2 = 第二行车道 ...
            -1 = 无边界信息可用
        """
        if not boundaries:
            return -1

        for i in range(len(boundaries)):
            if px < boundaries[i]:
                return i  # 0=硬路肩, 1=第一车道...
        return len(boundaries)  # 最右侧之外 = 硬路肩或出口

    @staticmethod
    def save_lut(lut: np.ndarray, path: str) -> None:
        """保存查找表到 .npy 文件。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, lut)
        logger.info(f"LUT 已保存: {path}")

    @staticmethod
    def load_lut(path: str) -> np.ndarray:
        """从 .npy 文件加载查找表。"""
        lut = np.load(path)
        logger.info(f"LUT 已加载: {path}, shape={lut.shape}")
        return lut


class LUTLookup:
    """O(1) 查找表查询。

    用法:
        lane, dist = LUTLookup.lookup_point(lut, x, y)
        lane, dist = LUTLookup.lookup(lut, (cx, cy))  # 使用中心点
    """

    @staticmethod
    def lookup_point(lut: np.ndarray, x: int, y: int) -> tuple:
        """查单个像素点的车道号和距离。

        参数:
            lut: LUT_DTYPE 数组
            x: 列索引 (0~W-1)
            y: 行索引 (0~H-1)

        返回:
            (lane_number: int, distance_m: float)
        """
        if lut is None:
            return (-1, -1.0)
        h, w = lut.shape
        x = int(x)
        y = int(y)
        if x < 0 or x >= w or y < 0 or y >= h:
            return (-1, -1.0)
        lane = int(lut["lane"][y, x])
        dist = float(lut["dist"][y, x])
        return (lane, dist)

    @staticmethod
    def lookup(lut: np.ndarray, center: tuple) -> tuple:
        """查目标中心点的车道号和距离。

        参数:
            lut: LUT_DTYPE 数组
            center: (cx, cy) 中心点坐标 (float)

        返回:
            (lane_number: int, distance_m: float)
        """
        cx, cy = center
        return LUTLookup.lookup_point(lut, int(cx), int(cy))


# =============================================================================
# RoadMask —— 基于 IPM 的路面区域检测
# =============================================================================

class RoadMask:
    """5 边形路面掩码 —— 首帧基于车道线自动生成 + 可拖拽微调。

    用法:
        mask = RoadMask(W=1920, H=1080)
        mask.generate_initial()
        mask.confirm_with_gui(frame)  # 交互确认顶点
        is_road = mask.contains(cx, cy)
    """

    def __init__(self, image_size: tuple = (1920, 1080)):
        self.W, self.H = image_size
        self._mask: np.ndarray | None = None
        # 5 个顶点: [top_left, top_right, bottom_right, bottom_left]
        self.corners: list[list[int]] = [
            [int(self.W * 0.25), int(self.H * 0.45)],  # 0 TL
            [int(self.W * 0.75), int(self.H * 0.45)],  # 1 TR
            [int(self.W * 0.85), int(self.H * 0.85)],  # 2 R
            [int(self.W * 0.50), int(self.H * 0.95)],  # 3 BC (bottom center)
            [int(self.W * 0.15), int(self.H * 0.85)],  # 4 L
        ]
        self._dragging_idx = -1
        self._drag_offset = (0, 0)
        self._scale = (1.0, 1.0)  # display->original coords

    def generate_initial(self) -> np.ndarray:
        """生成初始 5 边形掩码（基于默认值）。"""
        return self._update_mask()

    def generate_from_lanes(self, frame: np.ndarray) -> np.ndarray:
        """基于车道线检测自动生成初始 5 边形。

        步骤:
            1. Canny 边缘检测
            2. HoughLinesP 检测线段
            3. 按斜率/位置聚合成左右车道线
            4. 用车道线条的端点构建 5 边形
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=50, minLineLength=80, maxLineGap=50,
        )

        left_pts, right_pts = [], []
        W, H = self.W, self.H

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                if y2 == y1:
                    continue
                slope = (x2 - x1) / (y2 - y1)
                center_x = (x1 + x2) / 2
                # 过滤水平线 (slope < 0.1)
                if abs(slope) < 0.1:
                    continue
                # 按位置和斜率分左右
                if center_x < W * 0.5 and slope > 0:
                    left_pts.append((x1, y1))
                    left_pts.append((x2, y2))
                elif center_x >= W * 0.5 and slope < 0:
                    right_pts.append((x1, y1))
                    right_pts.append((x2, y2))

        # 如果车道线检测失败，回退默认值
        if len(left_pts) < 2 or len(right_pts) < 2:
            self.corners = [
                [int(W * 0.25), int(H * 0.45)],
                [int(W * 0.75), int(H * 0.45)],
                [int(W * 0.85), int(H * 0.85)],
                [int(W * 0.50), int(H * 0.95)],
                [int(W * 0.15), int(H * 0.85)],
            ]
            return self._update_mask()

        # 提取最上端和最下端的点
        left_pts.sort(key=lambda p: p[1])  # 按 y 排序
        right_pts.sort(key=lambda p: p[1])

        top_y = max(min(left_pts[0][1], right_pts[0][1]), 0)
        bot_y = min(max(left_pts[-1][1], right_pts[-1][1]), H - 1)

        # 拟合左右车道线
        left_arr = np.array(left_pts)
        right_arr = np.array(right_pts)
        if len(left_arr) >= 2:
            l_coeffs = np.polyfit(left_arr[:, 1], left_arr[:, 0], 1)
        else:
            l_coeffs = [0, int(W * 0.25)]
        if len(right_arr) >= 2:
            r_coeffs = np.polyfit(right_arr[:, 1], right_arr[:, 0], 1)
        else:
            r_coeffs = [0, int(W * 0.75)]

        # 计算四个角点
        tl_x = max(0, int(l_coeffs[0] * top_y + l_coeffs[1]))
        tr_x = min(W - 1, int(r_coeffs[0] * top_y + r_coeffs[1]))
        bl_x = max(0, int(l_coeffs[0] * bot_y + l_coeffs[1]))
        br_x = min(W - 1, int(r_coeffs[0] * bot_y + r_coeffs[1]))
        bc_x = (bl_x + br_x) // 2  # 底部中点

        # 向外扩展 20% 以覆盖完整车道宽度
        margin = int((tr_x - tl_x) * 0.15)
        tl_x = max(0, tl_x - margin)
        tr_x = min(W - 1, tr_x + margin)
        bl_x = max(0, bl_x - margin)
        br_x = min(W - 1, br_x + margin)

        # 加一些底部垂直扩展
        bot_y = min(H - 1, bot_y + int(H * 0.05))

        self.corners = [
            [tl_x, top_y],      # 0 TL
            [tr_x, top_y],      # 1 TR
            [br_x, bot_y],      # 2 R
            [bc_x, min(H-1, bot_y + int(H*0.03))],  # 3 BC
            [bl_x, bot_y],      # 4 L
        ]

        return self._update_mask()

    def _update_mask(self) -> np.ndarray:
        self._mask = np.zeros((self.H, self.W), dtype=np.uint8)
        pts = np.array(self.corners, dtype=np.int32)
        cv2.fillPoly(self._mask, [pts], 1)
        return self._mask

    def contains(self, x: float, y: float) -> bool:
        if self._mask is None:
            return True
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= self.W or iy < 0 or iy >= self.H:
            return False
        return bool(self._mask[iy, ix])

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    def draw(self, image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """绘制半透明四边形 + 红色顶点。"""
        if self._mask is None:
            return image
        overlay = image.copy()
        overlay[self._mask == 1] = (0, 255, 0)
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        pts = np.array(self.corners, dtype=np.int32)
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)
        for i, (cx, cy) in enumerate(self.corners):
            cv2.circle(result, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(result, str(i), (cx + 12, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return result

    def _mouse_callback(self, event, x, y, flags, param):
        # Convert display coords to original image coords
        sx, sy = self._scale
        ox, oy = int(x * sx), int(y * sy)
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (cx, cy) in enumerate(self.corners):
                dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if dist < 30:
                    self._dragging_idx = i
                    self._drag_offset = (cx - ox, cy - oy)
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging_idx >= 0:
            self.corners[self._dragging_idx] = [
                max(0, min(self.W - 1, ox + self._drag_offset[0])),
                max(0, min(self.H - 1, oy + self._drag_offset[1])),
            ]
            self._update_mask()
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging_idx = -1

    def confirm_with_gui(self, frame: np.ndarray, window_name: str = "Road Mask") -> bool:
        """交互式标定：拖拽红色圆圈调整 5 边形，Enter 确认。"""
        if self._mask is None:
            self.generate_initial()

        h, w = frame.shape[:2]
        disp_w, disp_h = 1280, 720
        self._scale = (w / disp_w, h / disp_h)
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        while True:
            overlay = self.draw(frame)
            display = cv2.resize(overlay, (disp_w, disp_h))

            info = [
                "Drag red circles (0-3) to fit road area",
                "R: reset  Enter: confirm  Q/Esc: cancel",
            ]
            for i, txt in enumerate(info):
                cv2.putText(display, txt, (10, 25 + i * 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:
                cv2.destroyWindow(window_name)
                return True
            elif key == ord('q') or key == 27:
                self._mask = None
                cv2.destroyWindow(window_name)
                return False
            elif key == ord('r'):
                self.__init__(image_size=(self.W, self.H))
                self._scale = (w / disp_w, h / disp_h)
                self.generate_initial()
                self._update_mask()
