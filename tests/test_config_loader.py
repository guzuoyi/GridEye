"""测试：ConfigLoader —— 配置加载、缺失字段报错、类型校验"""

import pytest
import tempfile
import os
from pathlib import Path

from src.config_loader import ConfigLoader, AppConfig, QwenConfig


# ---- fixtures ---------------------------------------------------------------

@pytest.fixture
def valid_yaml_path():
    """创建临时有效配置文件。"""
    content = """
video:
  source: "test.mp4"
  fps: 2
  original_width: 1920
  original_height: 1080

yolo:
  model_path: "models/yolov11n.pt"
  input_size: 640
  confidence_threshold: 0.25

size_filter:
  person:
    min_height: 35
    min_area: 1000
  car:
    min_height: 22
    min_area: 1500
  truck:
    min_height: 28
    min_area: 2000
  two-wheeler:
    min_height: 22
    min_area: 1000

flicker:
  window_frames: 10
  stable_threshold: 7
  size_scale: 0.8

position_cache:
  match_distance_px: 40
  disappear_timeout_sec: 3.0

state_machine:
  dwell_frames_to_warning: 5
  confirmed_refresh_interval: 30.0
  manual_required_timeout: 300

qwen:
  base_url: "http://localhost:1234/v1"
  api_key: "lm-studio"
  model_name: "qwen3.5-9b"
  crop_size: 448
  temperature: 0.1
  max_tokens: 512
  timeout_sec: 15
  max_retries: 3
  max_concurrency: 2

latency:
  end_to_end_max_sec: 5.0
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


# ---- tests ------------------------------------------------------------------

def test_load_valid_config(valid_yaml_path):
    """正常加载合法配置。"""
    loader = ConfigLoader(valid_yaml_path)
    cfg = loader.load()
    assert isinstance(cfg, AppConfig)
    assert cfg.video.fps == 2
    assert cfg.video.original_width == 1920
    assert cfg.yolo.input_size == 640
    assert cfg.size_filter.thresholds["person"]["min_height"] == 35
    assert cfg.flicker.window_frames == 10
    assert cfg.flicker.size_scale == 0.8
    assert cfg.position_cache.match_distance_px == 40
    assert cfg.state_machine.dwell_frames_to_warning == 5
    assert cfg.state_machine.confirmed_refresh_interval == 30.0
    assert cfg.state_machine.manual_required_timeout == 300
    assert cfg.qwen.base_url == "http://localhost:1234/v1"
    assert cfg.qwen.model_name == "qwen3.5-9b"


def test_missing_file():
    """配置文件不存在时报错。"""
    loader = ConfigLoader("nonexistent.yaml")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_invalid_fps():
    """fps 超出合理范围应报错。"""
    content = "video:\n  source: t.mp4\n  fps: 0\n  original_width: 1920\n  original_height: 1080\n"
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        loader = ConfigLoader(path)
        with pytest.raises(ValueError, match="fps"):
            loader.load()
    finally:
        os.unlink(path)


def test_invalid_dwell_frames():
    """dwell_frames 必须 >= 1。"""
    content = """video:
  source: t.mp4
  fps: 2
  original_width: 1920
  original_height: 1080
state_machine:
  dwell_frames_to_warning: 0
"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        loader = ConfigLoader(path)
        with pytest.raises(ValueError, match="dwell_frames"):
            loader.load()
    finally:
        os.unlink(path)


def test_default_values():
    """缺失字段时使用默认值。"""
    content = """video:
  source: t.mp4
  fps: 2
  original_width: 1920
  original_height: 1080
"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        loader = ConfigLoader(path)
        cfg = loader.load()
        # 默认值
        assert cfg.yolo.input_size == 640
        assert cfg.size_filter.thresholds == {}  # 无 size_filter 节 → 空
        assert cfg.qwen.base_url == "http://localhost:1234/v1"
        assert cfg.position_cache.match_distance_px == 40
    finally:
        os.unlink(path)


def test_env_override_qwen_url():
    """环境变量 QWEN_BASE_URL 覆盖 YAML。"""
    content = """video:
  source: t.mp4
  fps: 2
  original_width: 1920
  original_height: 1080
qwen:
  base_url: "http://localhost:1234/v1"
"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        os.environ["QWEN_BASE_URL"] = "http://override:9999/v1"
        loader = ConfigLoader(path)
        cfg = loader.load()
        assert cfg.qwen.base_url == "http://override:9999/v1"
    finally:
        os.unlink(path)
        del os.environ["QWEN_BASE_URL"]
