"""测试：FlickerSuppressor —— 滑动窗口稳定性判定"""

import pytest
from src.detector.flicker import FlickerSuppressor


class TestFlickerSuppressor:
    """防闪烁核心逻辑"""

    def test_stable_when_enough_valid_frames(self):
        """最近 10 帧中 ≥ 7 帧有效 → 稳定。"""
        fs = FlickerSuppressor(window_frames=10, stable_threshold=7)
        pid = "car_100_200"

        # 前 6 帧有效（第 6 帧时有效 total=6, 还不稳定）
        for i in range(6):
            stable = fs.update(pid, True)
        assert stable is False  # 6/10 < 7

        # 第 7 帧有效 → 稳定
        stable = fs.update(pid, True)
        assert stable is True   # 7/10 >= 7

    def test_resets_on_invalid_frames(self):
        """有效帧不足 → 不稳定。"""
        fs = FlickerSuppressor(window_frames=10, stable_threshold=7)
        pid = "truck_300_400"

        # 前 5 帧有效，后 5 帧无效
        for _ in range(5):
            fs.update(pid, True)
        stable = fs.is_stable(pid)
        assert stable is False  # 5 帧还不够

        for _ in range(5):
            stable = fs.update(pid, False)
        assert stable is False  # 5 valid / 10 < 7

    def test_window_sliding(self):
        """滑动窗口：旧帧逐出，新帧进入。"""
        fs = FlickerSuppressor(window_frames=10, stable_threshold=7)
        pid = "person_500_600"

        # 第 1-7 帧有效 → 稳定
        for _ in range(7):
            fs.update(pid, True)
        assert fs.is_stable(pid) is True

        # 第 8-10 帧无效 → 稳定性下降
        for _ in range(3):
            fs.update(pid, False)
        # 目前窗口: [T,T,T,T,T,T,T,F,F,F] → 7 valid → 仍稳定
        assert fs.is_stable(pid) is True

        # 第 11 帧无效，第 1 帧 T 逐出 → [T,T,T,T,T,T,F,F,F,F] → 6 valid → 不稳定
        fs.update(pid, False)
        assert fs.is_stable(pid) is False

    def test_multiple_targets(self):
        """多目标独立维护。"""
        fs = FlickerSuppressor(window_frames=10, stable_threshold=7)
        a, b = "car_A", "truck_B"

        for _ in range(10):
            fs.update(a, True)   # a: 全有效
            fs.update(b, False)  # b: 全无效

        assert fs.is_stable(a) is True
        assert fs.is_stable(b) is False

    def test_reset(self):
        """reset 清除目标历史。"""
        fs = FlickerSuppressor(window_frames=10, stable_threshold=7)
        pid = "car_X"

        for _ in range(8):
            fs.update(pid, True)
        assert fs.is_stable(pid) is True

        fs.reset(pid)
        assert fs.is_stable(pid) is False

    def test_cleanup_stale(self):
        """cleanup_stale 清除不再活跃的目标。"""
        fs = FlickerSuppressor(window_frames=10, stable_threshold=7)
        for _ in range(7):
            fs.update("a", True)   # a: 7 帧有效 → 稳定
        fs.update("b", True)       # b: 1 帧
        assert len(fs) == 2

        fs.cleanup_stale({"a"})  # 只保留 a
        assert len(fs) == 1
        assert fs.is_stable("a") is True

    def test_invalid_threshold(self):
        """stable_threshold > window_frames 应报错。"""
        with pytest.raises(ValueError):
            FlickerSuppressor(window_frames=10, stable_threshold=11)

    def test_is_stable_on_unknown(self):
        """未知 ID 返回 False。"""
        fs = FlickerSuppressor()
        assert fs.is_stable("unknown") is False
