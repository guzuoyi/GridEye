"""测试：Archiver + MetricsCalculator"""

import json
import pytest
import tempfile
import os
import numpy as np
from pathlib import Path

from src.archive.archiver import (
    Archiver, ArchiveSample, Scene, Label,
)
from src.archive.metrics import (
    ClassificationMetrics,
    MetricsCalculator,
    generate_report,
)


# ===== Archiver ==============================================================

@pytest.fixture
def archiver(tmp_path):
    return Archiver(root=str(tmp_path / "archive"))


class TestArchiver:
    """归档器测试"""

    def test_dirs_created(self, archiver):
        """初始化时创建所有目录。"""
        for scene in ["event_confirmation", "risk_clearance"]:
            for label in ["TP", "FP", "FN"]:
                p = archiver.root / scene / label
                assert p.exists()

    def test_archive_sample_with_image(self, archiver):
        """归档带图片的样本。"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[16:48, 16:48] = (0, 255, 0)

        sample = Archiver.make_sample(
            Scene.EVENT_CONFIRM,
            record_id="rec12345",
            input_data={"class": "car", "lane": 1, "distance": 45.0},
        )
        archiver.archive(sample, image=img, label=Label.TP)

        # 验证 JSON 存在
        json_files = list(
            (archiver.root / "event_confirmation" / "TP").glob("*.json")
        )
        assert len(json_files) == 1

        # 验证图片存在
        png_files = list(
            (archiver.root / "event_confirmation" / "TP").glob("*.png")
        )
        assert len(png_files) == 1

    def test_archive_without_image(self, archiver):
        """纯文本归档（风险消除场景无图片）。"""
        sample = Archiver.make_sample(
            Scene.RISK_CLEARANCE,
            record_id="rec_risk",
            input_data={"class": "truck", "event_type": "违停"},
        )
        archiver.archive(sample, image=None, label=Label.TP)
        json_files = list(
            (archiver.root / "risk_clearance" / "TP").glob("*.json")
        )
        assert len(json_files) == 1

    def test_sample_count(self, archiver):
        """统计各标签数量。"""
        for _ in range(3):
            s = Archiver.make_sample(Scene.EVENT_CONFIRM, record_id="r")
            archiver.archive(s, label=Label.TP)
        for _ in range(2):
            s = Archiver.make_sample(Scene.EVENT_CONFIRM, record_id="r")
            archiver.archive(s, label=Label.FP)

        assert archiver.sample_count(
            scene=Scene.EVENT_CONFIRM, label=Label.TP
        ) == 3
        assert archiver.sample_count(
            scene=Scene.EVENT_CONFIRM, label=Label.FP
        ) == 2

    def test_list_samples(self, archiver):
        """列出已归档样本。"""
        s = Archiver.make_sample(Scene.EVENT_CONFIRM, record_id="abc")
        archiver.archive(s, label=Label.TP)

        samples = archiver.list_samples(Scene.EVENT_CONFIRM, Label.TP)
        assert len(samples) == 1
        assert samples[0]["record_id"] == "abc"

    def test_to_dict_serializable(self, archiver):
        """to_dict 可以 JSON 序列化。"""
        sample = Archiver.make_sample(
            Scene.EVENT_CONFIRM,
            input_data={"class": "car"},
            qwen_output='{"is_anomaly": true}',
            parsed_result={"is_anomaly": True},
        )
        d = sample.to_dict()
        json.dumps(d)  # 不抛异常


# ===== Metrics ===============================================================

class TestClassificationMetrics:
    """指标计算"""

    def test_perfect(self):
        m = ClassificationMetrics(tp=10, fp=0, fn=0, tn=5)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.accuracy == 1.0

    def test_all_wrong(self):
        m = ClassificationMetrics(tp=0, fp=10, fn=5, tn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0

    def test_mixed(self):
        m = ClassificationMetrics(tp=8, fp=2, fn=2, tn=8)
        assert m.precision == pytest.approx(0.8)
        assert m.recall == pytest.approx(0.8)
        assert m.f1 == pytest.approx(0.8)
        assert m.accuracy == pytest.approx(0.8)

    def test_zero_denominator(self):
        m = ClassificationMetrics(tp=0, fp=0, fn=0, tn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_to_dict(self):
        m = ClassificationMetrics(tp=7, fp=3, fn=1)
        d = m.to_dict()
        assert d["tp"] == 7
        assert "precision" in d


class TestMetricsCalculator:
    """从归档计算指标"""

    def test_calculate_from_archiver(self, archiver):
        """归档 TP/FP/FN 后计算指标。"""
        # 归档 8 TP
        for _ in range(8):
            s = Archiver.make_sample(Scene.EVENT_CONFIRM, record_id="r")
            archiver.archive(s, label=Label.TP)
        # 归档 2 FP
        for _ in range(2):
            s = Archiver.make_sample(Scene.EVENT_CONFIRM, record_id="r")
            archiver.archive(s, label=Label.FP)

        calc = MetricsCalculator()
        m = calc.calculate(archiver)
        assert m.tp == 8
        assert m.fp == 2
        assert m.precision == pytest.approx(0.8)

    def test_calculate_separate(self, archiver):
        """分别计算两个场景。"""
        for _ in range(5):
            s = Archiver.make_sample(Scene.EVENT_CONFIRM)
            archiver.archive(s, label=Label.TP)
        for _ in range(3):
            s = Archiver.make_sample(Scene.RISK_CLEARANCE)
            archiver.archive(s, label=Label.FP)

        event_m, risk_m = MetricsCalculator().calculate_separate(archiver)
        assert event_m.tp == 5
        assert risk_m.fp == 3


class TestReport:
    """报告生成"""

    def test_generates_markdown(self):
        e = ClassificationMetrics(tp=8, fp=2, fn=1, tn=5)
        r = ClassificationMetrics(tp=3, fp=0, fn=0)
        report = generate_report(e, r, qwen_calls=15, total_frames=500)
        assert "# 交通事件" in report
        assert "## 事件确认" in report
        assert "H1" in report
        assert "97.00%" in report  # 1 - 15/500 = 0.97
