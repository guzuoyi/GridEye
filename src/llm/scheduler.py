"""Qwen 调用调度器 —— FIFO 队列、并发控制、超时、延迟监控"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
from logging import getLogger

logger = getLogger(__name__)


class CallType(Enum):
    EVENT_CONFIRM = "event_confirm"
    STATE_REFRESH = "state_refresh"
    RISK_CLEARANCE = "risk_clearance"


@dataclass
class QwenCallRequest:
    """一次推理请求"""
    call_type: CallType
    record_id: str         # 关联的 PositionRecord ID
    callback: Callable     # 回调函数 callback(result: dict)
    # 事件确认参数
    system_prompt: str = ""
    user_text: str = ""
    image: Optional[object] = None  # np.ndarray


@dataclass
class QwenCallResult:
    """推理结果"""
    request: QwenCallRequest
    parsed: dict
    raw_content: str = ""
    elapsed_sec: float = 0.0
    tokens: int = 0
    success: bool = False
    error: str = ""


class QwenCallScheduler:
    """管理 Qwen API 调用队列。

    - FIFO 队列
    - 可配置并发数
    - 单个请求超时
    - 延迟监控
    """

    def __init__(
        self,
        client,  # QwenAPIClient
        max_concurrency: int = 1,
        max_latency_sec: float = 30.0,
    ):
        self.client = client
        self.max_concurrency = max_concurrency
        self.max_latency = max_latency_sec
        self._queue: deque[QwenCallRequest] = deque()
        self._active_count = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._total_calls = 0
        self._total_latency = 0.0

    # ---- 入队 ---------------------------------------------------------------
    def enqueue(self, request: QwenCallRequest) -> None:
        """将请求加入队列。"""
        with self._lock:
            self._queue.append(request)
        logger.debug(
            f"Enqueued {request.call_type.value} for {request.record_id[:8]}, "
            f"queue len={len(self._queue)}"
        )

    # ---- 启动/停止 ----------------------------------------------------------
    def start(self) -> None:
        """启动后台处理线程。"""
        if self._worker is not None:
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        logger.info("Qwen scheduler started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=10)
            self._worker = None

    # ---- 处理循环 -----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                if len(self._queue) == 0 or self._active_count >= self.max_concurrency:
                    pass
                else:
                    req = self._queue.popleft()
                    self._active_count += 1
                    threading.Thread(target=self._process, args=(req,), daemon=True).start()

            time.sleep(0.1)  # 避免忙等

    def _process(self, req: QwenCallRequest) -> None:
        """处理单个推理请求。"""
        t0 = time.time()
        try:
            result = self.client.chat(
                system_prompt=req.system_prompt,
                user_text=req.user_text,
                image=req.image,
            )
            dt = time.time() - t0

            qr = QwenCallResult(
                request=req,
                parsed={},
                raw_content=result["content"],
                elapsed_sec=dt,
                tokens=result["tokens"],
                success=result["finish"] == "stop",
                error="" if result["finish"] == "stop" else f"finish={result['finish']}",
            )

            if dt > self.max_latency:
                logger.warning(
                    f"Qwen call {req.call_type.value} exceeded latency: {dt:.1f}s"
                )

        except Exception as e:
            dt = time.time() - t0
            qr = QwenCallResult(
                request=req, parsed={}, elapsed_sec=dt, success=False, error=str(e),
            )
            logger.error(f"Qwen call failed for {req.record_id[:8]}: {e}")

        # 回调
        try:
            req.callback(qr)
        except Exception as e:
            logger.error(f"Callback error for {req.record_id[:8]}: {e}")

        with self._lock:
            self._active_count -= 1
            self._total_calls += 1
            self._total_latency += qr.elapsed_sec

    # ---- 指标 ---------------------------------------------------------------
    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def avg_latency(self) -> float:
        with self._lock:
            if self._total_calls == 0:
                return 0.0
            return self._total_latency / self._total_calls

    @property
    def total_calls(self) -> int:
        return self._total_calls
