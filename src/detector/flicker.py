"""防闪烁抑制器 —— 消除远处小目标检测不稳定的影响"""

from collections import deque
from typing import Dict, Optional

from src.detector.models import Detection


class FlickerSuppressor:
    """为每个（伪 ID）目标维护尺寸有效性滑动窗口。

    最近 N 帧中 ≥ K 帧有效 → 认定为稳定目标，驻留帧数正常累加。
    否则 → 驻留帧数重置为 0。

    用法:
        fs = FlickerSuppressor(window_frames=10, stable_threshold=7)
        for det in detections:
            pseudo_id = f"{det.class_name}_{int(det.center[0])}_{int(det.center[1])}"
            if fs.is_stable(pseudo_id, det.is_size_valid):
                # 稳定，累加驻留帧数
    """

    def __init__(
        self,
        window_frames: int = 10,
        stable_threshold: int = 7,
    ):
        if stable_threshold > window_frames:
            raise ValueError(
                f"stable_threshold ({stable_threshold}) 不能大于 window_frames ({window_frames})"
            )
        self.window_frames = window_frames
        self.stable_threshold = stable_threshold
        self._history: Dict[str, deque] = {}  # pseudo_id → deque[bool]

    def update(
        self, pseudo_id: str, size_valid: bool
    ) -> bool:
        """记录一帧的尺寸有效性，返回是否稳定。

        参数:
            pseudo_id: 目标伪标识（由 class + 位置生成）
            size_valid: 本帧该目标是否通过尺寸过滤

        返回:
            True 表示目标稳定（最近 N 帧 ≥ K 帧有效）
        """
        if pseudo_id not in self._history:
            self._history[pseudo_id] = deque(maxlen=self.window_frames)

        q = self._history[pseudo_id]
        q.append(size_valid)

        return self._check_stable(q)

    def is_stable(self, pseudo_id: str) -> bool:
        """查询目标当前是否稳定（不追加新帧）。"""
        q = self._history.get(pseudo_id)
        if q is None:
            return False
        return self._check_stable(q)

    def _check_stable(self, q: deque) -> bool:
        """滑动窗口内有效帧 ≥ threshold 则稳定。"""
        return sum(q) >= self.stable_threshold

    def reset(self, pseudo_id: str) -> None:
        """重置目标的历史记录（目标消失时调用）。"""
        self._history.pop(pseudo_id, None)

    def cleanup_stale(self, active_ids: set[str]) -> None:
        """清理不再活跃的目标历史。"""
        stale = [k for k in self._history if k not in active_ids]
        for k in stale:
            del self._history[k]

    def __len__(self) -> int:
        return len(self._history)
