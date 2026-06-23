"""错误样本归档器

目录结构:
    archive/
    ├── event_confirmation/
    │   ├── TP/    # 正确识别
    │   ├── FP/    # 误报
    │   └── FN/    # 漏报
    └── risk_clearance/
        ├── TP/
        ├── FP/
        └── FN/

每个样本: 裁剪图(PNG) + 元数据(JSON)
"""

import json
import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from logging import getLogger

logger = getLogger(__name__)

_counter = [0]  # global counter for unique filenames


class Scene(Enum):
    EVENT_CONFIRM = "event_confirmation"
    RISK_CLEARANCE = "risk_clearance"


class Label(Enum):
    TP = "TP"  # True Positive
    FP = "FP"  # False Positive
    FN = "FN"  # False Negative


@dataclass
class ArchiveSample:
    """单条归档样本"""
    timestamp: str                     # ISO8601
    scene: str                         # "event_confirmation" | "risk_clearance"
    label: str                         # "TP" | "FP" | "FN"
    record_id: str = ""
    input: dict = field(default_factory=dict)       # {class, lane, distance, dwell_frames, ...}
    qwen_output: dict = field(default_factory=dict)  # Qwen 原始响应
    ground_truth: str = ""             # 人工标注
    image_path: str = ""               # 相对路径
    parsed_result: dict = field(default_factory=dict)  # 解析后的结构化结果

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "scene": self.scene,
            "label": self.label,
            "record_id": self.record_id,
            "input": self.input,
            "qwen_output": self.qwen_output,
            "ground_truth": self.ground_truth,
            "image_path": self.image_path,
            "parsed_result": self.parsed_result,
        }


class Archiver:
    """错误样本自动归档器。

    用法:
        archiver = Archiver(root="archive")
        archiver.archive(sample, image, label=Label.TP)
    """

    def __init__(self, root: str = "archive"):
        self.root = Path(root)
        self._ensure_dirs()

    def _ensure_dirs(self):
        for scene in Scene:
            for label in Label:
                (self.root / scene.value / label.value).mkdir(
                    parents=True, exist_ok=True
                )

    def archive(
        self,
        sample: ArchiveSample,
        image: Optional[np.ndarray] = None,
        label: Optional[Label] = None,
    ) -> str:
        """归档一个样本。

        参数:
            sample: 结构化样本数据
            image: 目标裁剪图 (BGR), 可选
            label: 标签 (TP/FP/FN), 可选

        返回:
            归档目录路径
        """
        if label:
            sample.label = label.value

        scene_dir = self.root / sample.scene / sample.label

        # 生成文件名
        ts = sample.timestamp.replace(":", "-").replace(".", "-")
        rid = sample.record_id[:8] if sample.record_id else "unknown"
        cls_name = sample.input.get("class", "unknown")
        filename_base = f"{ts}_{rid}_{cls_name}"

        # 保存图片
        if image is not None:
            img_path = scene_dir / f"{filename_base}.png"
            cv2.imwrite(str(img_path), image)
            sample.image_path = str(img_path.relative_to(self.root))
            logger.debug(f"Image saved: {img_path}")

        # 保存 JSON
        json_path = scene_dir / f"{filename_base}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(
            f"Archived [{sample.label}] {sample.scene}: {filename_base}"
        )
        return str(scene_dir)

    def list_samples(
        self, scene: Optional[Scene] = None, label: Optional[Label] = None
    ) -> list[dict]:
        """列出已归档的样本元数据。"""
        samples = []
        scenes = [scene] if scene else list(Scene)
        labels = [label] if label else list(Label)

        for sc in scenes:
            for lb in labels:
                d = self.root / sc.value / lb.value
                if not d.exists():
                    continue
                for jf in d.glob("*.json"):
                    with open(jf, encoding="utf-8") as f:
                        samples.append(json.load(f))
        return samples

    def sample_count(
        self, scene: Optional[Scene] = None, label: Optional[Label] = None
    ) -> int:
        """统计归档数量。"""
        count = 0
        scenes = [scene] if scene else list(Scene)
        labels = [label] if label else list(Label)
        for sc in scenes:
            for lb in labels:
                d = self.root / sc.value / lb.value
                if d.exists():
                    count += len(list(d.glob("*.json")))
        return count

    @staticmethod
    def make_sample(
        scene: Scene,
        record_id: str = "",
        input_data: dict = None,
        qwen_output: str = "",
        parsed_result: dict = None,
        ground_truth: str = "",
        label: str = "",
        timestamp: str = None,
    ) -> ArchiveSample:
        """快捷构造 ArchiveSample。"""
        import datetime
        if timestamp is None:
            _counter[0] += 1
            timestamp = datetime.datetime.now().strftime(
                f"%Y-%m-%dT%H-%M-%S-{_counter[0]:04d}"
            )
        return ArchiveSample(
            timestamp=timestamp,
            scene=scene.value,
            label=label,
            record_id=record_id,
            input=input_data or {},
            qwen_output={"raw": qwen_output},
            ground_truth=ground_truth,
            parsed_result=parsed_result or {},
        )
