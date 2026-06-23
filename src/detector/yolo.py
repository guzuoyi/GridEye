"""YOLOv11 目标检测器

基于 ultralytics YOLO，将原始帧 resize 到 640x640（Stretch 模式），
推理后将检测框映射回 1920x1080 原始分辨率。
"""

from typing import List, Optional
import numpy as np

from src.detector.models import Detection
from src.utils.coords import rescale_bbox_stretch, bbox_center


class YOLODetector:
    """YOLOv11 检测器封装。

    用法:
        detector = YOLODetector(
            model_path="models/yolov11n.pt",
            class_names=["car", "truck", "person", "bicycle", "motorcycle"],
            input_size=640,
            orig_width=1920,
            orig_height=1080,
            conf_threshold=0.25,
        )
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        model_path: str = "models/yolov11n.pt",
        class_names: Optional[List[str]] = None,
        input_size: int = 640,
        orig_width: int = 1920,
        orig_height: int = 1080,
        conf_threshold: float = 0.25,
        night_mode: bool = False,
    ):
        self.model_path = model_path
        self.class_names = class_names or ["car", "truck", "person", "bicycle", "motorcycle"]
        self.input_size = input_size
        self.orig_width = orig_width
        self.orig_height = orig_height
        self.conf_threshold = conf_threshold
        self.night_mode = night_mode
        self._model = None
        self._clahe = None

        # 类别名映射：YOLO 输出类别编号 → 统一类别名
        # 用户只需在 config 中列出关注的类别，两轮车会合并
        self._class_map: Optional[dict] = None

    def load(self) -> None:
        """加载检测模型（惰性加载，支持 YOLO / RTDETR）。"""
        model_path_lower = self.model_path.lower()
        if 'rtdetr' in model_path_lower or 'rt-detr' in model_path_lower:
            from ultralytics import RTDETR
            self._model = RTDETR(self.model_path)
        else:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path, task="detect")
        # 构建类别编号 → 名称映射
        self._build_class_map()

    def _build_class_map(self) -> None:
        """构建 YOLO 类别编号到统一类别名的映射。"""
        # YOLO COCO 类别名 → 我们的统一类别名
        coco_to_unified = {
            "car": "car",
            "truck": "truck",
            "bus": "truck",          # bus 合并到 truck
            "person": "person",
            "bicycle": "two-wheeler",
            "motorcycle": "two-wheeler",
        }
        # 从 YOLO 模型获取类别名列表
        if self._model is not None and hasattr(self._model, "names"):
            yolo_names = self._model.names  # {0: "person", 1: "bicycle", ...}
            self._class_map = {}
            for idx, yolo_cls in yolo_names.items():
                unified = coco_to_unified.get(yolo_cls, None)
                if unified is not None and unified in self.class_names:
                    self._class_map[idx] = unified

    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """对一帧图像执行目标检测。

        参数:
            frame: (H, W, 3) BGR 图像，预期 1920×1080

        返回:
            Detection 列表（已映射到原始分辨率）
        """
        if self._model is None:
            self.load()

        # 1. 夜间预处理
        if self.night_mode:
            frame = self._preprocess_night(frame)

        # 2. resize 到 640x640（Stretch）
        frame_640 = self._resize_frame(frame)

        # 3. YOLO 推理
        results = self._model(frame_640, verbose=False)
        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # 3. 提取并映射检测框
        detections = []
        for box in boxes:
            conf = float(box.conf.item())
            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls.item())
            unified_cls = self._class_map.get(cls_id, None) if self._class_map else None
            if unified_cls is None:
                continue  # 非关注类别

            # YOLO 输出 (x1,y1,x2,y2) 在 640 坐标系
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox_640 = (int(x1), int(y1), int(x2), int(y2))

            # Stretch 映射回 1920x1080
            bbox_orig = rescale_bbox_stretch(
                bbox_640,
                orig_w=self.orig_width,
                orig_h=self.orig_height,
                yolo_size=self.input_size,
            )
            center = bbox_center(bbox_orig)

            det = Detection(
                class_name=unified_cls,
                confidence=conf,
                bbox=bbox_orig,
                center=center,
            )
            detections.append(det)

        return detections

    def _preprocess_night(self, frame: np.ndarray) -> np.ndarray:
        """夜间增强：CLAHE + 亮度拉伸，让暗处目标可见。"""
        import cv2

        # 转灰度做 CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Gamma 提亮暗区
        gamma = 1.3
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                         for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

        return enhanced

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """将帧 resize 到 YOLO 输入尺寸（Stretch 模式）。"""
        import cv2
        return cv2.resize(frame, (self.input_size, self.input_size))
