"""异步 Qwen Worker — 支持图片发送"""

import queue, threading, time, cv2, json, re
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class QwenTask:
    track_id: int
    event_type: str
    image: Optional[np.ndarray] = None
    image_path: str = ""
    locked_cx: float = 0.0
    locked_cy: float = 0.0


class QwenWorkerPool:
    def __init__(self, qwen_client, max_workers=1):
        self.client = qwen_client
        self.queue = queue.Queue(maxsize=20)
        self.callback: Optional[Callable] = None
        self._running = False
        self._max_workers = max_workers
        self.total_calls = 0

    def set_callback(self, cb): self.callback = cb

    def enqueue(self, task: QwenTask):
        try:
            self.queue.put_nowait(task)
        except queue.Full:
            pass

    def start(self):
        self._running = True
        for _ in range(self._max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()

    def stop(self): self._running = False

    def _worker_loop(self):
        while self._running:
            try:
                task = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                is_anomaly, event_type = self._call_qwen(task)
                self.total_calls += 1
                if self.callback:
                    self.callback(task.track_id, is_anomaly, event_type,
                                  task.image_path, task.event_type)
            except Exception as e:
                print(f"[QwenWorker] error: {e}")

    def _call_qwen(self, task):
        if task.event_type == "EVENT_CONFIRM":
            system = (
                "你是高速公路交通监控专家。你将看到一张拼接图：左边是T0(20秒前)、右边是T20(当前)，"
                "两图为同一位置截图。"
                "请对比两帧：若目标在同一位置静止未移动(≥20秒)，则属于违停/抛锚/事故异常；"
                "若已驶离或仍在行驶，则正常。"
                "用JSON回复："
                '{"is_anomaly": true/false, "event_type": "违停|抛锚|事故|正常", '
                '"confidence": 0.0-1.0, "reasoning": "20字内简述"}'
            )
            text = "T0和T20同一位置。若车辆静止没动→异常，已开走→正常。"
        else:
            system = (
                "你是高速公路交通监控专家。此框标注了之前报警的违停区域。"
                "请判断该区域内是否有明显车辆存在。"
                'JSON回复：{"车辆存在": true/false, "confidence": 0.0-1.0, '
                '"event_type": "不正常|正常"}'
            )
            text = ""

        # 保存裁剪图
        if task.image is not None:
            task.image_path = f"results/crop_{time.strftime('%Y%m%d_%H%M%S')}_{task.track_id}.jpg"
            cv2.imwrite(task.image_path, task.image)

        # 调用 Qwen（带图片）
        result = self.client.chat(
            system_prompt=system, user_text=text, image=task.image)

        content = result.get("content", "")
        try:
            obj = json.loads(re.search(r'\{.*\}', content, re.DOTALL).group())
            if task.event_type == "EVENT_CONFIRM":
                return obj.get("is_anomaly", False), obj.get("event_type", "")
            else:
                et = obj.get("event_type", "")
                car_exists = obj.get("车辆存在", True)
                if car_exists:
                    return True, et   # 有车，风险未消除
                else:
                    return False, et  # 无车，风险消除
        except:
            return False, ""
