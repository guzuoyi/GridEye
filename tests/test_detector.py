"""测试：YOLODetector —— 类别映射、坐标映射、推理接口"""

import pytest
import numpy as np

np = pytest.importorskip("numpy")  # sanity check
from src.detector.yolo import YOLODetector


class TestYOLODetectorInit:
    """初始化逻辑"""

    def test_init_defaults(self):
        det = YOLODetector(model_path="models/yolov11n.pt")
        assert det.input_size == 640
        assert det.orig_width == 1920
        assert det.orig_height == 1080
        assert det.conf_threshold == 0.25
        assert det.is_loaded() is False  # 惰性加载

    def test_custom_params(self):
        det = YOLODetector(
            model_path="custom.pt",
            class_names=["car", "person"],
            input_size=320,
            orig_width=1280,
            orig_height=720,
            conf_threshold=0.5,
        )
        assert det.input_size == 320
        assert det.orig_width == 1280
        assert det.conf_threshold == 0.5

    def test_class_names_default(self):
        det = YOLODetector()
        assert "car" in det.class_names
        assert "person" in det.class_names
        assert "truck" in det.class_names
        assert "two-wheeler" not in det.class_names
        assert "bicycle" in det.class_names  # raw YOLO class name


class TestResizeFrame:
    """_resize_frame 方法"""

    def test_resize_to_640(self):
        """1920x1080 BGR 图像 → 640x640。"""
        import cv2
        cv2 = pytest.importorskip("cv2")

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        det = YOLODetector()
        resized = det._resize_frame(frame)

        assert resized.shape == (640, 640, 3)
        assert resized.dtype == np.uint8


class TestDetectWithoutModel:
    """无模型时的行为"""

    def test_detect_raises_without_model(self):
        """未加载模型时调用 detect() 会尝试加载（然后失败）。"""
        det = YOLODetector(model_path="nonexistent_model.pt")
        with pytest.raises(Exception):  # 可能是 FileNotFoundError 或 RuntimeError
            det.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
