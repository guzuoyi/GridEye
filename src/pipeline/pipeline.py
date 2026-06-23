"""主流水线 —— 集成所有 Phase 模块，端到端运行"""

import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from logging import getLogger

import csv
from pathlib import Path
from src.cache.cache import (
    State, StateMachine, PositionCache, PositionRecord,
)
from src.llm.client import QwenAPIClient
from src.llm.prompts import (
    SYSTEM_EVENT_CONFIRM, prompt_event_confirm,
    SYSTEM_STATE_REFRESH, prompt_state_refresh,
    SYSTEM_RISK_CLEARANCE, prompt_risk_clearance,
)
from src.llm.parser import parse_event_confirm, parse_risk_clearance
from src.llm.scheduler import QwenCallScheduler, QwenCallRequest, CallType
from src.archive.archiver import Archiver, ArchiveSample, Scene, Label
from src.utils.coords import bbox_center

logger = getLogger(__name__)

# 状态对应的颜色 (BGR)
STATE_COLORS = {
    State.IDLE: (128, 128, 128),              # gray
    State.WARNING: (0, 165, 255),              # orange
    State.CONFIRMED: (0, 0, 255),              # red
    State.CLEARED: (255, 255, 0),              # cyan
    State.MANUAL_REQUIRED: (0, 0, 200),        # dark red
}


@dataclass
class FrameResult:
    """单帧处理结果"""
    frame_idx: int
    timestamp: float
    detections: list = field(default_factory=list)
    records: list = field(default_factory=list)
    elapsed_sec: Dict[str, float] = field(default_factory=dict)  # {module: seconds}


class MainPipeline:
    """主流水线。

    用法:
        pipeline = MainPipeline(config)
        pipeline.setup()
        for result in pipeline.run():
            print(f"Frame {result.frame_idx}: {len(result.records)} targets")
    """

    def __init__(self, config):
        """
        参数:
            config: AppConfig (from ConfigLoader)
        """
        self.cfg = config
        self._running = False
        self._components_ready = False

        # 各组件 Placeholder
        self.sampler = None
        self.yolo = None
        self.size_filter = None
        self.flicker = None
        self.cache = None
        self.state_machine = None
        self.qwen_client = None
        self.qwen_scheduler = None
        self.archiver = None
        self.lut_maintainer = None

        # 统计
        self.total_frames = 0
        self.total_qwen_calls = 0
        # CSV + 截图输出
        self._results_dir = Path("results")
        self._results_dir.mkdir(exist_ok=True)
        self._csv_path = self._results_dir / "qwen_results.csv"
        self._init_csv()

    # ---- Setup --------------------------------------------------------------

    def setup(self) -> None:
        """初始化所有组件。"""
        logger.info("Initializing pipeline...")

        # --- Video ---
        from src.sampler.sampler import VideoSampler
        self.sampler = VideoSampler(
            source=self.cfg.video.source,
            target_fps=self.cfg.video.fps,
            original_width=self.cfg.video.original_width,
            original_height=self.cfg.video.original_height,
        )

        # --- YOLO ---
        from src.detector.yolo import YOLODetector
        self.yolo = YOLODetector(
            model_path=self.cfg.yolo.model_path,
            class_names=self.cfg.yolo.classes,
            input_size=self.cfg.yolo.input_size,
            orig_width=self.cfg.video.original_width,
            orig_height=self.cfg.video.original_height,
            conf_threshold=self.cfg.yolo.confidence_threshold,
            night_mode=getattr(self.cfg.yolo, "night_mode", False),
        )

        # --- SizeFilter ---
        from src.detector.size_filter import SizeFilter
        self.size_filter = SizeFilter.from_config(
            self.cfg.size_filter.thresholds, scale=1.0
        )

        # --- FlickerSuppressor ---
        from src.detector.flicker import FlickerSuppressor
        self.flicker = FlickerSuppressor(
            window_frames=self.cfg.flicker.window_frames,
            stable_threshold=self.cfg.flicker.stable_threshold,
        )

        # --- Position Cache ---
        self.cache = PositionCache(
            match_distance_px=self.cfg.position_cache.match_distance_px,
            disappear_timeout_sec=self.cfg.position_cache.disappear_timeout_sec,
        )

        # --- State Machine ---
        self.state_machine = StateMachine(
            dwell_frames_to_warning=self.cfg.state_machine.dwell_frames_to_warning,
            disappear_timeout_sec=self.cfg.position_cache.disappear_timeout_sec,
            refresh_interval_sec=self.cfg.state_machine.confirmed_refresh_interval,
            stationary_threshold_px=self.cfg.state_machine.stationary_threshold_px,
        )

        # --- Qwen ---
        self.qwen_client = QwenAPIClient(
            base_url=self.cfg.qwen.base_url,
            api_key=self.cfg.qwen.api_key,
            model_name=self.cfg.qwen.model_name,
            crop_size=self.cfg.qwen.crop_size,
            temperature=self.cfg.qwen.temperature,
            max_tokens=self.cfg.qwen.max_tokens,
            timeout_sec=self.cfg.qwen.timeout_sec,
            max_retries=self.cfg.qwen.max_retries,
        )
        self.qwen_scheduler = QwenCallScheduler(
            self.qwen_client,
            max_concurrency=self.cfg.qwen.max_concurrency,
            max_latency_sec=self.cfg.latency_max_sec,
        )
        self.qwen_scheduler.start()

        # --- Archiver ---
        self.archiver = Archiver(root="archive")

        self._components_ready = True
        logger.info("Pipeline initialized")

    # ---- Run ----------------------------------------------------------------

    def run(self, max_frames: int = 0, visualize: bool = False):
        """运行主循环。

        参数:
            max_frames: 最大帧数 (0 = 无限)
            visualize: 是否显示实时可视化窗口

        Yields:
            FrameResult 每帧处理结果
        """
        if not self._components_ready:
            raise RuntimeError("Call setup() before run()")

        self.sampler.open()
        self._running = True
        frame_idx = 0

        logger.info(f"Pipeline started (max_frames={max_frames or 'unlimited'})")

        try:
            while self._running:
                if max_frames > 0 and frame_idx >= max_frames:
                    break

                # 1. 抽帧
                t0 = time.time()
                sampled = self.sampler.next()
                if sampled is None:
                    break
                t_sample = time.time()

                frame = sampled.image

                # 2. YOLO 检测
                detections = self.yolo.detect(frame)
                t_yolo = time.time()

                # 3. 尺寸过滤
                detections = self.size_filter.filter(detections)
                t_filter = time.time()

                # 4. 防闪烁
                for det in detections:
                    pid = self._make_pseudo_id(det)
                    size_ok = self.size_filter.is_valid(det)
                    stable = self.flicker.update(pid, size_ok)
                    if not stable:
                        det.consecutive_frames = 0  # 标记不稳定
                t_flicker = time.time()

                # 5. 位置缓存匹配
                records = self.cache.match_or_create(detections, now=sampled.timestamp)
                t_match = time.time()

                # 6. 查表定位 + 状态机 + Qwen 触发
                for rec in records:
                    self._process_record(rec, frame, sampled.timestamp)
                t_state = time.time()

                # 7. 过期清理
                stale = self.cache.cleanup_expired(now=sampled.timestamp)
                for rec in stale:
                    if rec.state == State.CLEARED:
                        self._enqueue_qwen(rec, frame)
                t_cleanup = time.time()

                # 8. 可视化
                vis_frame = None
                if visualize:
                    vis_frame = self._draw_overlay(frame, records, detections)

                total_elapsed = time.time() - t0
                result = FrameResult(
                    frame_idx=frame_idx,
                    timestamp=sampled.timestamp,
                    detections=detections,
                    records=records,
                    elapsed_sec={
                        "sample": t_sample - t0,
                        "yolo": t_yolo - t_sample,
                        "filter": t_filter - t_yolo,
                        "flicker": t_flicker - t_filter,
                        "match": t_match - t_flicker,
                        "state": t_state - t_match,
                        "cleanup": t_cleanup - t_state,
                        "total": total_elapsed,
                    },
                )

                if frame_idx % 30 == 0:
                    logger.info(
                        f"Frame {frame_idx}: {len(detections)} detections, "
                        f"{len(records)} targets, {total_elapsed:.2f}s"
                    )

                self.total_frames += 1
                frame_idx += 1

                if visualize and vis_frame is not None:
                    cv2.imshow("Traffic Detection", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                yield result

        finally:
            self.sampler.close()
            if visualize:
                cv2.destroyAllWindows()
            logger.info(
                f"Pipeline stopped. Frames={self.total_frames}, "
                f"Qwen calls={self.qwen_scheduler.total_calls}"
            )

    # ---- Internal -----------------------------------------------------------

    def _init_csv(self) -> None:
        """初始化 CSV 文件，写入表头。"""
        if not self._csv_path.exists():
            import csv
            with open(self._csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    'timestamp', 'frame_idx', 'record_id', 'class', 'lane',
                    'distance', 'consecutive_frames', 'state',
                    'is_anomaly', 'event_type', 'confidence', 'reasoning',
                    'risk_cleared', 'need_cleanup',
                    'elapsed_sec', 'tokens', 'crop_image'
                ])

    def _make_pseudo_id(self, detection) -> str:
        """为检测生成伪 ID（用于防闪烁）。"""
        cx, cy = detection.center
        return f"{detection.class_name}_{int(cx//20)}_{int(cy//20)}"

    def _process_record(
        self, rec: PositionRecord, frame: np.ndarray, ts: float
    ) -> None:
        """处理单条位置记录：查表 + 状态机 + Qwen 触发。"""
        # 查表定位（如果有 LUT）
        if self.lut_maintainer is not None:
            lane, dist = self.lut_maintainer.lookup(
                int(rec.center_x), int(rec.center_y)
            )
            rec.lane_number = lane
            rec.longitudinal_distance = dist

        # 过滤：consecutive_frames < 2 的单帧目标不触发状态机
        if rec.consecutive_frames < 2:
            return

        # 过滤：画面顶部 40% 区域（y < 0.4*H）非道路，跳过
        h = frame.shape[0]
        if rec.center_y < h * 0.4:
            return

        # 过滤：lane=-1（道路外目标）且距离 < 5m（过于靠近边缘）
        if rec.lane_number < 0 and self.lut_maintainer is not None:
            return

        # 状态机评估
        new_state, needs_qwen, reason = self.state_machine.evaluate(
            rec, current_time=ts
        )

        if new_state != rec.state or needs_qwen:
            self.state_machine.apply_transition(rec, new_state, current_time=ts)

        if needs_qwen and rec.state == State.WARNING:
            self._enqueue_qwen(rec, frame)

    def _save_crop(self, rec: PositionRecord, image: np.ndarray) -> str:
        """保存裁剪图到 results/ 目录，返回相对路径。"""
        ts = time.strftime("%Y%m%d_%H%M%S")
        rid = rec.id[:8]
        fname = f"crop_{ts}_{rid}.jpg"
        path = self._results_dir / fname
        cv2.imwrite(str(path), image)
        return str(path)

    def _append_csv(self, rec: PositionRecord, result, parsed: dict,
                    crop_path: str, call_type: str) -> None:
        """追加一条 Qwen 结果到 CSV。"""
        with open(self._csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                self.total_frames,
                rec.id[:8],
                rec.class_name,
                rec.lane_number,
                round(rec.longitudinal_distance, 1),
                rec.consecutive_frames,
                rec.state.value,
                parsed.get('is_anomaly', ''),
                parsed.get('event_type', ''),
                parsed.get('confidence', ''),
                parsed.get('reasoning', ''),
                parsed.get('risk_cleared', ''),
                parsed.get('need_cleanup', ''),
                round(result.elapsed_sec, 1) if hasattr(result, 'elapsed_sec') else '',
                result.tokens if hasattr(result, 'tokens') else '',
                crop_path,
            ])

    def _enqueue_qwen(self, rec: PositionRecord, frame: np.ndarray) -> None:
        """将 Qwen 调用入队（同时保存裁剪图）。"""
        crop = None
        crop_path = ""
        if rec.state == State.WARNING:
            crop = self._crop_target(frame, rec)
            if crop is not None:
                crop_path = self._save_crop(rec, crop)
            req = QwenCallRequest(
                call_type=CallType.EVENT_CONFIRM,
                record_id=rec.id,
                system_prompt=SYSTEM_EVENT_CONFIRM,
                user_text=prompt_event_confirm(
                    rec.class_name,
                    rec.lane_number,
                    rec.longitudinal_distance,
                    rec.consecutive_frames,
                    rec.elapsed_in_state(State.WARNING),
                ),
                image=crop,
                callback=lambda r, cp=crop_path: self._on_qwen_result(rec.id, r, cp),
            )
            self.qwen_scheduler.enqueue(req)

        elif rec.state == State.CLEARED:
            req = QwenCallRequest(
                call_type=CallType.RISK_CLEARANCE,
                record_id=rec.id,
                system_prompt=SYSTEM_RISK_CLEARANCE,
                user_text=prompt_risk_clearance(
                    rec.class_name,
                    rec.lane_number,
                    rec.longitudinal_distance,
                    rec.qwen_result.event_type if rec.qwen_result else "未知",
                    "?",
                    "?",
                ),
                image=None,
                callback=lambda r: self._on_qwen_result(rec.id, r),
            )
            self.qwen_scheduler.enqueue(req)

    def _on_qwen_result(self, record_id: str, result, crop_path: str = "") -> None:
        """Qwen 回调：解析结果并更新状态。"""
        rec = self.cache.get(record_id)
        if rec is None:
            return

        parsed = {}
        if result.success:
            raw = result.raw_content
            if rec.state == State.WARNING:
                parsed = parse_event_confirm(raw)
                from src.cache.cache import QwenResult
                qr = QwenResult(
                    is_anomaly=parsed.get("is_anomaly", False),
                    event_type=parsed.get("event_type", ""),
                    confidence=parsed.get("confidence", 0.0),
                    reasoning=parsed.get("reasoning", ""),
                )
                new_state, _, _ = self.state_machine.evaluate(
                    rec, qwen_result=qr
                )
                self.state_machine.apply_transition(rec, new_state, qwen_result=qr)

                # 归档
                sample = ArchiveSample(
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    scene=Scene.EVENT_CONFIRM.value,
                    label="",  # 待人工复核
                    record_id=rec.id,
                    input={
                        "class": rec.class_name,
                        "lane": rec.lane_number,
                        "distance": rec.longitudinal_distance,
                        "dwell_frames": rec.consecutive_frames,
                    },
                    qwen_output=raw,
                    parsed_result=parsed,
                )
                self.archiver.archive(sample, label=Label.TP)

            elif rec.state == State.CLEARED:
                parsed = parse_risk_clearance(raw)
                from src.cache.cache import QwenResult
                qr = QwenResult(
                    risk_cleared=parsed.get("risk_cleared", False),
                    need_cleanup=parsed.get("need_cleanup", False),
                    confidence=parsed.get("confidence", 0.0),
                    reasoning=parsed.get("reasoning", ""),
                )
                new_state, _, _ = self.state_machine.evaluate(
                    rec, qwen_result=qr
                )
                self.state_machine.apply_transition(rec, new_state, qwen_result=qr)

            if result.success:
                self._append_csv(rec, result, parsed, crop_path, rec.state.value)

            self.total_qwen_calls += 1

    def _crop_target(
        self, frame: np.ndarray, rec: PositionRecord
    ) -> Optional[np.ndarray]:
        """裁剪目标区域。"""
        margin = 20
        cx, cy = int(rec.center_x), int(rec.center_y)
        size = 80  # 默认裁剪大小
        x1 = max(0, cx - size)
        y1 = max(0, cy - size)
        x2 = min(frame.shape[1], cx + size)
        y2 = min(frame.shape[0], cy + size)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    # ---- Visualization ------------------------------------------------------

    def _draw_overlay(
        self, frame: np.ndarray, records: list, detections: list = None
    ) -> np.ndarray:
        """在帧上叠加检测框和状态标签。"""
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. 画 YOLO 检测框（绿色）
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(out, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 2. 画位置缓存记录（彩色圆 + 状态）
        # 连续帧 < 2 的不画（单帧误检不显示）
        for rec in records:
            if rec.consecutive_frames < 2:
                continue
            color = STATE_COLORS.get(rec.state, (255, 255, 255))
            cx, cy = int(rec.center_x), int(rec.center_y)
            r = 25
            cv2.circle(out, (cx, cy), 4, color, -1)  # 实心
            cv2.circle(out, (cx, cy), r, color, 1)   # 空心

            state_short = rec.state.value[:4]
            label = (
                f"{rec.class_name} [{state_short}] "
                f"d={rec.consecutive_frames} L{rec.lane_number}"
            )
            cv2.putText(out, label, (cx - r, cy - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 3. 顶部信息栏
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)

        nd = len(detections) if detections else 0
        info = f"Frame: {self.total_frames} | Targets: {len(records)} | Dets: {nd} | Qwen: {self.total_qwen_calls}"
        cv2.putText(out, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 缩放到适合屏幕
        display = cv2.resize(out, (1280, 720))
        return display

    def stop(self) -> None:
        """停止流水线。"""
        self._running = False
        if self.qwen_scheduler:
            self.qwen_scheduler.stop()

    # ---- Metrics (H1) -------------------------------------------------------

    @property
    def qwen_call_reduction(self) -> float:
        """Qwen 调用减少比例。"""
        if self.total_frames == 0:
            return 1.0
        return 1.0 - self.total_qwen_calls / max(self.total_frames, 1)
