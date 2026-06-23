"""测试：VideoSampler —— 抽帧数与帧率校验"""

import pytest
import tempfile
import os
from pathlib import Path

cv2 = pytest.importorskip("cv2", reason="opencv-python 未安装，跳过抽帧测试")
import numpy as np

from src.sampler.sampler import VideoSampler


# ---- helpers ----------------------------------------------------------------

def make_test_video(path: str, duration_sec: float = 5.0, fps: int = 25, size=(640, 480)):
    """创建一段纯色测试视频。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    n_frames = int(duration_sec * fps)
    for i in range(n_frames):
        # 每帧颜色微变，便于区分
        color = ((i * 10) % 256, 128, 128)
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ---- tests ------------------------------------------------------------------

class TestVideoSampler:
    """VideoSampler 测试"""

    def test_frame_count_matches_target_fps(self):
        """抽帧数应 ≈ target_fps × 时长。"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            # 创建 5 秒 25fps 测试视频
            make_test_video(video_path, duration_sec=5.0, fps=25, size=(1920, 1080))
            target_fps = 2

            sampler = VideoSampler(
                source=video_path,
                target_fps=target_fps,
                original_width=1920,
                original_height=1080,
            )
            sampler.open()

            frames = []
            while True:
                f = sampler.next()
                if f is None:
                    break
                frames.append(f)

            sampler.close()

            # 5s × 2fps = 10 帧，允许 ±1 误差
            assert abs(len(frames) - 10) <= 1, f"期望 ~10 帧，实际 {len(frames)}"

        finally:
            os.unlink(video_path)

    def test_frame_indices_continuous(self):
        """帧序号应连续。"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            make_test_video(video_path, duration_sec=3.0, fps=25, size=(640, 480))
            sampler = VideoSampler(source=video_path, target_fps=2)
            sampler.open()

            indices = []
            while True:
                f = sampler.next()
                if f is None:
                    break
                indices.append(f.frame_idx)

            sampler.close()

            assert indices == list(range(len(indices))), "帧序号应 0,1,2,... 连续"

        finally:
            os.unlink(video_path)

    def test_image_shape_correct(self):
        """抽帧图像尺寸应等于配置的 original_width/original_height。"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            make_test_video(video_path, duration_sec=1.0, fps=25, size=(640, 480))
            sampler = VideoSampler(source=video_path, target_fps=2,
                                   original_width=1920, original_height=1080)
            sampler.open()

            f = sampler.next()
            sampler.close()

            assert f is not None
            assert f.image.shape[0] == 1080  # height
            assert f.image.shape[1] == 1920  # width
            assert f.image.shape[2] == 3     # BGR

        finally:
            os.unlink(video_path)

    def test_raises_on_unopened(self):
        """未打开时调用 next() 应报错。"""
        sampler = VideoSampler(source="data/fake.mp4", target_fps=2)
        with pytest.raises(RuntimeError):
            sampler.next()

    def test_raises_on_invalid_source(self):
        """无效视频源应报错。"""
        sampler = VideoSampler(source="nonexistent_file_999.mp4", target_fps=2)
        with pytest.raises(RuntimeError):
            sampler.open()

    def test_as_iterator(self):
        """VideoSampler 应可作为迭代器使用。"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        sampler = None
        try:
            make_test_video(video_path, duration_sec=2.0, fps=25, size=(640, 480))
            sampler = VideoSampler(source=video_path, target_fps=2)
            sampler.open()

            count = 0
            for frame in sampler:
                assert frame.image is not None
                count += 1
            assert count > 0

        finally:
            if sampler is not None:
                sampler.close()
            os.unlink(video_path)
