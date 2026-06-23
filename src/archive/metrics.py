"""评估指标计算器 —— Precision, Recall, F1, Accuracy + 混淆矩阵"""

from dataclasses import dataclass, field
from typing import List, Optional
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """二分类指标"""
    tp: int = 0     # True Positive
    fp: int = 0     # False Positive
    fn: int = 0     # False Negative
    tn: int = 0     # True Negative

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class ExperimentMetrics:
    """实验级完整指标"""
    event_confirm: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    risk_clearance: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    qwen_call_count: int = 0
    total_frames: int = 0

    @property
    def call_reduction_ratio(self) -> float:
        """Qwen 调用减少比例 (H1 验证)"""
        baseline = self.total_frames  # 逐帧调用
        if baseline == 0:
            return 1.0
        return 1.0 - (self.qwen_call_count / baseline)


class MetricsCalculator:
    """从归档 JSON 文件计算评估指标。

    用法:
        calc = MetricsCalculator("archive")
        event_m = calc.calculate(Scene.EVENT_CONFIRM)
        risk_m  = calc.calculate(Scene.RISK_CLEARANCE)
    """

    def __init__(self, archive_root: str = "archive"):
        self.root = archive_root

    def calculate(self, archiver) -> ClassificationMetrics:
        """从 Archiver 中读取 TP/FP/FN 计数。

        参数:
            archiver: Archiver 实例

        返回:
            ClassificationMetrics
        """
        from src.archive.archiver import Scene, Label
        m = ClassificationMetrics()
        for sc in [Scene.EVENT_CONFIRM, Scene.RISK_CLEARANCE]:
            for lb in [Label.TP, Label.FP, Label.FN]:
                cnt = archiver.sample_count(scene=sc, label=lb)
                if lb == Label.TP:
                    m.tp += cnt
                elif lb == Label.FP:
                    m.fp += cnt
                elif lb == Label.FN:
                    m.fn += cnt
        return m

    def calculate_separate(
        self, archiver
    ) -> tuple[ClassificationMetrics, ClassificationMetrics]:
        """分别计算事件确认和风险消除的指标。"""
        from src.archive.archiver import Scene, Label

        def _count(scene):
            m = ClassificationMetrics()
            for lb in [Label.TP, Label.FP, Label.FN]:
                cnt = archiver.sample_count(scene=scene, label=lb)
                if lb == Label.TP: m.tp = cnt
                elif lb == Label.FP: m.fp = cnt
                elif lb == Label.FN: m.fn = cnt
            return m

        return (
            _count(Scene.EVENT_CONFIRM),
            _count(Scene.RISK_CLEARANCE),
        )


def generate_report(
    event_metrics: ClassificationMetrics,
    risk_metrics: ClassificationMetrics,
    qwen_calls: int = 0,
    total_frames: int = 0,
) -> str:
    """生成 Markdown 格式的实验报告。

    返回:
        Markdown 字符串
    """
    e = event_metrics.to_dict()
    r = risk_metrics.to_dict()

    reduction = 0.0
    if total_frames > 0:
        reduction = 1.0 - qwen_calls / max(total_frames, 1)

    report = f"""# 交通事件二次识别 — 实验报告

## 事件确认

| 指标 | 值 |
|------|-----|
| TP | {e['tp']} |
| FP | {e['fp']} |
| FN | {e['fn']} |
| **Precision** | **{e['precision']:.2%}** |
| **Recall**    | **{e['recall']:.2%}** |
| **F1**        | **{e['f1']:.2%}** |
| **Accuracy**  | **{e['accuracy']:.2%}** |

### 混淆矩阵

| | Predicted Positive | Predicted Negative |
|---|:---:|:---:|
| **Actual Positive** | TP={e['tp']} | FN={e['fn']} |
| **Actual Negative** | FP={e['fp']} | TN={e['tn']} |

## 风险消除

| 指标 | 值 |
|------|-----|
| Precision | {r['precision']:.2%} |
| Recall    | {r['recall']:.2%} |
| F1        | {r['f1']:.2%} |
| Accuracy  | {r['accuracy']:.2%} |

## 系统效率 (H1)

| 指标 | 值 |
|------|-----|
| Qwen 调用次数 | {qwen_calls} |
| 总帧数 | {total_frames} |
| **调用减少** | **{reduction:.2%}** |
| 通过标准 | ≥ 90% |
| 状态 | {"✅ 通过" if reduction >= 0.9 else "❌ 未通过"} |

---
Generated at {__import__('datetime').datetime.now().isoformat()}
"""
    return report
