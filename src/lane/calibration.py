"""相机内参标定 —— 基于棋盘格自动检测角点并计算内参矩阵 K 和畸变系数"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class CalibrationResult:
    """标定结果"""
    K: np.ndarray          # 内参矩阵 3×3
    dist: np.ndarray       # 畸变系数 (5,)
    rms_error: float       # 重投影误差 (px)
    image_size: Tuple[int, int]  # (width, height)


class CameraCalibrator:
    """基于棋盘格的相机内参标定。

    用法:
        calib = CameraCalibrator(pattern=(9, 6), square_size_mm=25)
        result = calib.calibrate(image_paths)  # 或 calib.calibrate_from_frames(frames)
        calib.save(result, "config/camera_params.yaml")
    """

    def __init__(
        self,
        pattern: Tuple[int, int] = (9, 6),  # 内角点 (cols, rows)
        square_size_mm: float = 25.0,
    ):
        self.pattern = pattern
        self.square_size = square_size_mm
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

    def calibrate(
        self,
        image_paths: list[str],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> CalibrationResult:
        """从图像文件列表标定相机内参。

        参数:
            image_paths: 棋盘格图像文件路径列表（≥ 5 张建议 ≥ 15 张）
            image_size: 图像尺寸 (w, h)，若为 None 则从第一张图像获取

        返回:
            CalibrationResult
        """
        frames = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning(f"无法读取图像: {p}")
                continue
            frames.append(img)
        return self.calibrate_from_frames(frames, image_size)

    def calibrate_from_frames(
        self,
        frames: list[np.ndarray],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> CalibrationResult:
        """从图像帧列表标定相机内参。"""
        if len(frames) < 3:
            raise ValueError(f"至少需要 3 张图像，当前: {len(frames)}")

        if image_size is None:
            h, w = frames[0].shape[:2]
            image_size = (w, h)

        # 准备世界坐标系中的棋盘格角点坐标
        objp = np.zeros((self.pattern[0] * self.pattern[1], 3), np.float32)
        objp[:, :2] = (
            np.mgrid[0 : self.pattern[0], 0 : self.pattern[1]].T.reshape(-1, 2)
            * self.square_size
        )

        objpoints = []  # 3D 世界坐标
        imgpoints = []  # 2D 图像坐标

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern, None)
            if not ret:
                logger.warning("未检测到棋盘格角点，跳过此帧")
                continue

            # 亚像素精化
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self._criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners_sub)

        if len(objpoints) < 3:
            raise RuntimeError(
                f"成功检测角点的图像不足 (需要 ≥ 3，实际 {len(objpoints)})"
            )

        logger.info(f"标定: {len(objpoints)}/{len(frames)} 张图像成功检测角点")

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )

        return CalibrationResult(
            K=K,
            dist=dist,
            rms_error=rms,
            image_size=image_size,
        )

    def undistort(
        self, image: np.ndarray, result: CalibrationResult
    ) -> np.ndarray:
        """对图像去畸变。"""
        return cv2.undistort(image, result.K, result.dist)

    @staticmethod
    def save(result: CalibrationResult, path: str) -> None:
        """保存标定结果到 YAML 文件。"""
        d = {
            "K": result.K.tolist(),
            "dist": result.dist.tolist(),
            "rms_error": float(result.rms_error),
            "image_size": list(result.image_size),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=None)
        logger.info(f"标定结果已保存: {path}")

    @staticmethod
    def load(path: str) -> CalibrationResult:
        """从 YAML 文件加载标定结果。"""
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return CalibrationResult(
            K=np.array(d["K"], dtype=np.float64),
            dist=np.array(d["dist"], dtype=np.float64),
            rms_error=d["rms_error"],
            image_size=tuple(d["image_size"]),
        )
