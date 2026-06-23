"""测试：MainPipeline 集成"""

import pytest
import numpy as np
from src.config_loader import AppConfig
from src.pipeline.pipeline import MainPipeline


@pytest.fixture
def app_config():
    return AppConfig()


class TestPipelineInit:

    def test_init_with_config(self, app_config):
        pipe = MainPipeline(app_config)
        assert pipe.total_frames == 0
        assert pipe._components_ready is False

    def test_setup_creates_components(self, app_config):
        app_config.video.source = "nonexistent.mp4"
        pipe = MainPipeline(app_config)
        pipe.setup()
        assert pipe.sampler is not None
        assert pipe.yolo is not None
        assert pipe.cache is not None
        assert pipe.state_machine is not None
        assert pipe.qwen_client is not None
        pipe.stop()


class TestQwenReduction:

    def test_perfect_reduction(self, app_config):
        pipe = MainPipeline(app_config)
        pipe.total_frames = 1000
        pipe.total_qwen_calls = 5
        assert pipe.qwen_call_reduction == pytest.approx(0.995)

    def test_no_frames(self, app_config):
        pipe = MainPipeline(app_config)
        assert pipe.qwen_call_reduction == 1.0
