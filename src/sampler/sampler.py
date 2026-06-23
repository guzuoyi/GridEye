"""视频抽帧器

按配置的目标帧率从视频流中均匀抽取帧。
支持视频文件、RTSP 流、摄像头。
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class SampledFrame:
    """抽帧结果"""
    frame_idx: int            # 帧序号（从 0 起连续）
    timestamp: float          # 实际时间戳 (秒)
    image: np.ndarray         # 原始图像 (H, W, 3)


class VideoSampler:
    """视频抽帧器。

    用法:
        sampler = VideoSampler(source="data/test.mp4", target_fps=2)
        for frame in sampler:
            process(frame.image)
    """

    def __init__(
        self,
        source: str = "data/test_video.mp4",
        target_fps: int = 2,
        original_width: int = 1920,
        original_height: int = 1080,
    ):
        """
        参数:
            source: 视频文件路径 / RTSP URL / 摄像头 index (整数)
            target_fps: 目标抽帧速率
            original_width / original_height: 期望的图像分辨率（抽帧时 resize）
        """
        self.source = source
        self.target_fps = target_fps
        self.original_width = original_width
        self.original_height = original_height
        self._cap = None
        self._source_fps = None
        self._frame_interval = None
        self._frame_count = 0  # 已抽取帧计数器

    # --- context manager ----------------------------------------------------
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self

    def __next__(self) -> SampledFrame:
        frame = self.next()
        if frame is None:
            self.close()
            raise StopIteration
        return frame

    # --- open / close -------------------------------------------------------
    def open(self) -> None:
        """打开视频源。"""
        # 支持摄像头 index (int) 或 字符串路径
        if isinstance(self.source, int) or self.source.isdigit():
            source_id = int(self.source) if isinstance(self.source, str) else self.source
            self._cap = cv2.VideoCapture(source_id)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._source_fps <= 0:
            logger.warning("无法获取源帧率，假设为 25 FPS")
            self._source_fps = 25.0

        # 每 N 帧抽取一帧
        self._frame_interval = max(1, int(self._source_fps / self.target_fps))
        self._frame_count = 0

        logger.info(
            f"打开视频源: fps_src={self._source_fps:.1f}, "
            f"target_fps={self.target_fps}, interval={self._frame_interval}帧"
        )

    def close(self) -> None:
        """关闭视频源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # --- next frame ---------------------------------------------------------
    def next(self) -> SampledFrame | None:
        """获取下一帧，返回 None 表示视频结束。"""
        if self._cap is None:
            raise RuntimeError("视频源未打开，请先调用 open()")

        # 跳过中间帧，直到到达下一个抽帧点
        read_count = 0
        while read_count < self._frame_interval:
            ret, _ = self._cap.read()
            if not ret:
                return None  # 视频结束
            read_count += 1

        ret, frame = self._cap.read()
        if not ret:
            return None

        # resize 到目标分辨率
        if frame.shape[1] != self.original_width or frame.shape[0] != self.original_height:
            frame = cv2.resize(frame, (self.original_width, self.original_height))

        timestamp = time.time()
        sampled = SampledFrame(
            frame_idx=self._frame_count,
            timestamp=timestamp,
            image=frame,
        )
        self._frame_count += 1
        return sampled

    @property
    def source_fps(self) -> float:
        """源视频帧率。"""
        return self._source_fps or 0.0

    @property
    def estimated_frame_count(self) -> int:
        """估算总抽帧数（需在 open() 后调用）。"""
        if self._cap is None:
            return 0
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total // max(1, self._frame_interval or 1)
