#!/usr/bin/env python3
"""v7 — 网格化时空聚合 + 双帧 Qwen + 空间冷却"""

import sys, cv2, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.config_loader import ConfigLoader
from src.logging_config import setup_logging
from src.grid_engine import SpatialTemporalGrid, AlertCandidate
from src.qwen_worker import QwenWorkerPool, QwenTask

KEEP = {0: "person", 2: "car", 5: "bus", 7: "truck"}  # COCO class IDs

def main():
    setup_logging("ERROR")
    cfg = ConfigLoader("config/default.yaml").load()
    cfg.video.source = "data/test2.mp4"
    import torch; gpu = torch.cuda.is_available()

    from src.llm.client import QwenAPIClient
    qwen = QwenAPIClient(base_url=cfg.qwen.base_url, api_key=cfg.qwen.api_key,
                        model_name=cfg.qwen.model_name, timeout_sec=60)
    pool = QwenWorkerPool(qwen, 1); pool.start()

    from ultralytics import YOLO
    model = YOLO("models/yolo26s.pt")
    if gpu: model.to("cuda:0")
    print(f"YOLOv26s | GPU={gpu}")

    cap = cv2.VideoCapture(cfg.video.source)
    fps = cap.get(cv2.CAP_PROP_FPS); iv = max(1, int(fps / 5))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    grid = SpatialTemporalGrid(w, h)
    print(f"Grid: {grid.cols}x{grid.rows} cells ({grid.CELL_SIZE}px) | "
          f"Hold={grid.HOLD_THRESHOLD}frames")

    # T0 快照缓存
    t0_cache = {}  # {cand_id: (frame, cx, cy)}
    cand_id_counter = 0
    fn = 0
    MAX_FRAMES = 3000

    last_qwen_msg = ""  # 窗口实时显示
    count_history = []  # 记录 count 曲线
    qwen_record = []    # (frame, type, msg)

    def on_qwen(tid, an, et, ip, task_type=""):
        nonlocal last_qwen_msg
        qwen_record.append((fn, task_type, f"{task_type[:6]} track={tid} an={an}"))

        for c in grid.candidates:
            if id(c) == tid:
                c.qwen_pending = False
                if task_type == "EVENT_CONFIRM":
                    if an:
                        # 分支1: 异常 → 生成报警框(alerted=True)
                        c.alerted = True
                    else:
                        # 分支2: 非异常 → 锁50帧
                        c.locked_frames = 50
                elif task_type == "RISK_CLEARANCE":
                    if an:
                        # 分支1-2: 风险未消除(车还在) → 锁50帧
                        c.locked_frames = 50
                    else:
                        # 分支1-1: 风险消除(车走了) → 取消报警框
                        c.alerted = False
                        c.require_drop_below_20 = False
                        grid._alerted_zones = [(az_cx, az_cy, u) for az_cx, az_cy, u in grid._alerted_zones
                                              if abs(az_cx-c.cx) >= grid.CLUSTER_DIST or abs(az_cy-c.cy) >= grid.CLUSTER_DIST]
                last_qwen_msg = f"[{task_type[:6]}] track={tid} an={an} type={et}"
                print(f"  {last_qwen_msg}")
                break
        try:
            with open("results/qwen_results.csv", "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{tid},,{an},{et},{ip}\n")
        except: pass
    pool.set_callback(on_qwen)

    while True:
        for _ in range(iv): cap.grab()
        ok, frame = cap.retrieve()
        if not ok: break
        fn += 1
        if fn > MAX_FRAMES: break

        # 亮度检测 → 动态置信度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        if brightness < 40:
            conf = 0.08      # 极暗
        elif brightness < 80:
            conf = 0.1       # 昏暗
        else:
            conf = 0.3       # 明亮

        results = model.track(frame, persist=True, verbose=False,
                             classes=list(KEEP.keys()), conf=conf,
                             tracker="bytetrack.yaml")
        boxes = results[0].boxes

        # 构建 track 数据
        tracks = []
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                tracks.append({
                    'tid': int(box.id.item()),
                    'cls': KEEP.get(int(box.cls.item()), "unk"),
                    'cx': (x1+x2)/2, 'cy': (y1+y2)/2,
                    'w': x2-x1, 'h': y2-y1,
                })

        new_alerts = grid.update(tracks)

        for cand in new_alerts:
            cid = id(cand)
            if cid not in t0_cache:
                t0_cache[cid] = (frame.copy(), round(cand.cx), round(cand.cy))

            if not cand.qwen_pending:
                is_clear = cand.qwen_pending is False and cand.alerted  # 已触发过的=消警
                # 未触发过但 require_drop 没满足 → 跳过
                if not is_clear and not cand.alerted and cand.require_drop_below_20:
                    continue

                cand.qwen_pending = True
                if not cand.alerted:
                    cand.alerted = True
                    cand.require_drop_below_20 = True
                    grid._alerted_zones.append((cand.cx, cand.cy, time.time() + 3600))
                    qwen_record.append((fn, "TRIGGER", f"EVENT_CONFIRM sent"))
                elif is_clear:
                    qwen_record.append((fn, "TRIGGER", f"CLEARED sent"))
                mrg = int(max(cand.w, cand.h) / 2) + 30
                px, py = round(cand.cx), round(cand.cy)
                if is_clear:
                    # 消警：裁剪锁定区域图（同报警时一致）
                    crop = frame[max(0,py-mrg):min(h,py+mrg),max(0,px-mrg):min(w,px+mrg)]
                    pool.enqueue(QwenTask(
                        track_id=cid, event_type="RISK_CLEARANCE",
                        image=crop, locked_cx=cand.cx, locked_cy=cand.cy))
                else:
                    # 双帧确认
                    t0_fr, _, _ = t0_cache.get(cid, (None, 0, 0))
                    t0_img = None
                    if t0_fr is not None:
                        t0_img = t0_fr[max(0,py-mrg):min(h,py+mrg),max(0,px-mrg):min(w,px+mrg)]
                    t20_img = frame[max(0,py-mrg):min(h,py+mrg),max(0,px-mrg):min(w,px+mrg)]
                    dual = np.hstack([t0_img,t20_img]) if (t0_img is not None and t0_img.shape==t20_img.shape) else t20_img
                    pool.enqueue(QwenTask(
                        track_id=cid, event_type="EVENT_CONFIRM",
                        image=dual, locked_cx=cand.cx, locked_cy=cand.cy))

        # 绘制
        out = frame.copy()
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        # 候选框
        for cand in grid.candidates:
            c = (0, 0, 255) if cand.confirmed else (0, 165, 255)
            x1 = max(0, round(cand.cx - cand.w/2))
            y1 = max(0, round(cand.cy - cand.h/2))
            x2 = min(w, round(cand.cx + cand.w/2))
            y2 = min(h, round(cand.cy + cand.h/2))
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            label = "CONFIRMED" if cand.confirmed else "ALERT"
            cv2.putText(out, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        # 热点 mask 叠加
        hot = grid.get_hot_mask()
        if hot.sum() > 0:
            hot_viz = cv2.resize(hot, (w, h), interpolation=cv2.INTER_NEAREST)
            hot_viz = cv2.cvtColor(hot_viz, cv2.COLOR_GRAY2BGR)
            mask = hot_viz[..., 0] > 0
            hot_viz[mask] = (0, 0, 255)
            out = cv2.addWeighted(out, 0.85, hot_viz, 0.15, 0)

        # 网格热力图窗口（独立显示）
        heatmap = np.zeros((grid.rows * 20, grid.cols * 20), dtype=np.float32)
        mc = max(c.count for row in grid.grid for c in row)
        count_history.append(mc)
        for y in range(grid.rows):
            for x in range(grid.cols):
                v = min(grid.grid[y][x].count / 120, 1.0)
                heatmap[y*20:(y+1)*20, x*20:(x+1)*20] = v
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.putText(heatmap, f"Grid Count (max {mc})",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("GridHeatmap", heatmap)

        cv2.rectangle(out, (0, 0), (w, 54), (0, 0, 0), -1)
        cv2.putText(out, f"F:{fn} | Alerts:{len(grid.candidates)} | Qwen:{pool.total_calls} | B:{brightness:.0f} c:{conf:.2f}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if last_qwen_msg:
            from PIL import Image, ImageDraw, ImageFont
            pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("simhei.ttf", 16)
            except:
                font = ImageFont.load_default()
            draw.text((10, 36), last_qwen_msg, fill=(0, 255, 255), font=font)
            out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("TrafficGrid", cv2.resize(out, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(
            "TrafficGrid", cv2.WND_PROP_VISIBLE) < 1:
            break

    # 视频结束 → 等待 Qwen 完成（保持窗口打开）
    print(f"Video ended. Waiting for pending Qwen... Qwen calls: {pool.total_calls}")
    for wait_i in range(120):
        time.sleep(0.5)
        out = frame.copy() if frame is not None else np.zeros((h,w,3), dtype=np.uint8)
        cv2.putText(out, f"Video ended. Waiting for Qwen... (call {pool.total_calls})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if last_qwen_msg:
            cv2.putText(out, last_qwen_msg, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("TrafficGrid", cv2.resize(out, (1280, 720)))
        key = cv2.waitKey(500) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("TrafficGrid", cv2.WND_PROP_VISIBLE) < 1:
            break
    print(f"Done waiting. Qwen total: {pool.total_calls}")

    cap.release(); pool.stop()
    # 保存 count 曲线
    import json
    with open("results/count_history.json", "w") as f:
        json.dump({"video": "test1.mp4", "frames": list(range(len(count_history))),
                    "counts": [int(c) for c in count_history],
                    "qwen_record": qwen_record}, f)
    print(f"\\nDone: {fn}f | Qwen={pool.total_calls} | Alerts={len(grid.candidates)}")
    print("Saved: results/count_history.json")

if __name__ == "__main__":
    main()
