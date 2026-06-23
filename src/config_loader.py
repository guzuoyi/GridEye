"""配置加载器 —— YAML 加载 + 环境变量覆盖 + 参数校验"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class YoloConfig:
    model_path: str = "models/yolov11n.pt"
    input_size: int = 640
    confidence_threshold: float = 0.25
    classes: list = field(default_factory=lambda: ["car", "truck", "person", "two-wheeler"])


@dataclass
class SizeFilterConfig:
    thresholds: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "person": {"min_height": 35, "min_area": 1000},
        "car": {"min_height": 22, "min_area": 1500},
        "truck": {"min_height": 28, "min_area": 2000},
        "two-wheeler": {"min_height": 22, "min_area": 1000},
    })


@dataclass
class FlickerConfig:
    window_frames: int = 10
    stable_threshold: int = 7
    size_scale: float = 0.8


@dataclass
class PositionCacheConfig:
    match_distance_px: float = 40.0
    disappear_timeout_sec: float = 3.0


@dataclass
class StateMachineConfig:
    dwell_frames_to_warning: int = 5
    confirmed_refresh_interval: float = 30.0
    manual_required_timeout: float = 300.0
    stationary_threshold_px: float = 150.0


@dataclass
class QwenConfig:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model_name: str = "qwen3.5-9b"
    crop_size: int = 448
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_sec: float = 15.0
    max_retries: int = 3
    max_concurrency: int = 2


@dataclass
class VideoConfig:
    source: str = "data/test_video.mp4"
    fps: int = 2
    original_width: int = 1920
    original_height: int = 1080


@dataclass
class AppConfig:
    """顶层配置容器"""
    video: VideoConfig = field(default_factory=VideoConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    size_filter: SizeFilterConfig = field(default_factory=SizeFilterConfig)
    flicker: FlickerConfig = field(default_factory=FlickerConfig)
    position_cache: PositionCacheConfig = field(default_factory=PositionCacheConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    latency_max_sec: float = 5.0


class ConfigLoader:
    """从 YAML 加载配置，支持环境变量覆盖。"""

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = Path(config_path)

    def load(self) -> AppConfig:
        """加载并返回 AppConfig。"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = self._parse(raw)
        self._validate(cfg)
        return cfg

    def _parse(self, raw: dict) -> AppConfig:
        """将 YAML 字典解析为 AppConfig。"""
        # --- video ---
        v = raw.get("video", {})
        video = VideoConfig(
            source=v.get("source", "data/test_video.mp4"),
            fps=v.get("fps", 2),
            original_width=v.get("original_width", 1920),
            original_height=v.get("original_height", 1080),
        )

        # --- yolo ---
        y = raw.get("yolo", {})
        yolo = YoloConfig(
            model_path=y.get("model_path", "models/yolov11n.pt"),
            input_size=y.get("input_size", 640),
            confidence_threshold=y.get("confidence_threshold", 0.25),
            classes=y.get("classes", ["car", "truck", "person", "two-wheeler"]),
        )

        # --- size_filter ---
        sf = raw.get("size_filter", {})
        size_filter = SizeFilterConfig(thresholds={})
        for cls_name in sf:
            size_filter.thresholds[cls_name] = {
                "min_height": sf[cls_name].get("min_height", 0),
                "min_area": sf[cls_name].get("min_area", 0),
            }

        # --- flicker ---
        fl = raw.get("flicker", {})
        flicker = FlickerConfig(
            window_frames=fl.get("window_frames", 10),
            stable_threshold=fl.get("stable_threshold", 7),
            size_scale=fl.get("size_scale", 0.8),
        )

        # --- position_cache ---
        pc = raw.get("position_cache", {})
        position_cache = PositionCacheConfig(
            match_distance_px=pc.get("match_distance_px", 40),
            disappear_timeout_sec=pc.get("disappear_timeout_sec", 3.0),
        )

        # --- state_machine ---
        sm = raw.get("state_machine", {})
        state_machine = StateMachineConfig(
            dwell_frames_to_warning=sm.get("dwell_frames_to_warning", 5),
            confirmed_refresh_interval=sm.get("confirmed_refresh_interval", 30.0),
            manual_required_timeout=sm.get("manual_required_timeout", 300),
        )

        # --- qwen ---
        qw = raw.get("qwen", {})
        qwen = QwenConfig(
            base_url=os.getenv("QWEN_BASE_URL", qw.get("base_url", "http://localhost:1234/v1")),
            api_key=os.getenv("QWEN_API_KEY", qw.get("api_key", "lm-studio")),
            model_name=os.getenv("QWEN_MODEL_NAME", qw.get("model_name", "qwen3.5-9b")),
            crop_size=qw.get("crop_size", 448),
            temperature=qw.get("temperature", 0.1),
            max_tokens=qw.get("max_tokens", 512),
            timeout_sec=qw.get("timeout_sec", 15),
            max_retries=qw.get("max_retries", 3),
            max_concurrency=qw.get("max_concurrency", 2),
        )

        # --- latency ---
        lat = raw.get("latency", {})
        latency_max = lat.get("end_to_end_max_sec", 5.0)

        return AppConfig(
            video=video,
            yolo=yolo,
            size_filter=size_filter,
            flicker=flicker,
            position_cache=position_cache,
            state_machine=state_machine,
            qwen=qwen,
            latency_max_sec=latency_max,
        )

    def _validate(self, cfg: AppConfig) -> None:
        """校验配置参数的合法性。"""
        if cfg.video.fps < 1 or cfg.video.fps > 30:
            raise ValueError(f"video.fps 应在 1~30 范围，当前值: {cfg.video.fps}")

        if cfg.video.original_width <= 0 or cfg.video.original_height <= 0:
            raise ValueError("video 分辨率必须为正数")

        if cfg.yolo.input_size <= 0:
            raise ValueError("yolo.input_size 必须为正数")

        if cfg.position_cache.match_distance_px <= 0:
            raise ValueError("position_cache.match_distance_px 必须为正数")

        if cfg.position_cache.disappear_timeout_sec <= 0:
            raise ValueError("position_cache.disappear_timeout_sec 必须为正数")

        if cfg.state_machine.dwell_frames_to_warning < 1:
            raise ValueError("state_machine.dwell_frames_to_warning 必须 >= 1")

        if cfg.qwen.crop_size <= 0:
            raise ValueError("qwen.crop_size 必须为正数")

        if cfg.latency_max_sec <= 0:
            raise ValueError("latency.end_to_end_max_sec 必须为正数")
