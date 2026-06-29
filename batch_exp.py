#!/usr/bin/env python3
"""批量实验: 3视频 × 3方法"""

import sys, cv2, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.config_loader import ConfigLoader
from src.logging_config import setup_logging
from src.grid_engine import SpatialTemporalGrid
from src.fsm.tracker_manager import TrackerManager
from src.fsm.tracker_fsm import State as FSMState

KEEP = {0: "person", 2: "car", 5: "bus", 7: "truck"}
setup_logging("ERROR")

VIDEOS = ["data/test1.mp4", "data/test2.mp4"]
MAX_FRAMES = 500

results = {}

from ultralytics import YOLO

for video in VIDEOS:
    vname = Path(video).stem
    if not Path(video).exists():
        print(f"Skip {video}")
        continue
    print(f"\n{'='*50}\n  {vname}\n{'='*50}")

    model = YOLO("models/yolo26s.pt")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS); iv = max(1, int(fps/5))
    w, h = int(cap.get(3)), int(cap.get(4))

    video_data = {}

    # ── v5: Tracker FSM ──
    print("  v5 Tracker...", end=" ")
    mgr = TrackerManager()
    calls = 0
    for _ in range(MAX_FRAMES):
        for __ in range(iv): cap.grab()
        ok, frame = cap.retrieve()
        if not ok: break
        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if brightness < 40: conf = 0.08
        elif brightness < 80: conf = 0.1
        else: conf = 0.3
        r = model.track(frame, persist=True, verbose=False, classes=list(KEEP.keys()), conf=conf, tracker="bytetrack.yaml")
        boxes = r[0].boxes
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                st, nq, _ = mgr.update(int(box.id.item()), KEEP.get(int(box.cls.item()),"unk"), (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)
                if nq: calls += 1
        calls += len(mgr.mark_missing())
    video_data["v5_tracker"] = int(calls)
    print(f"{calls} calls")

    # Reset
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    mgr = TrackerManager()
    calls = 0

    # ── v6: Tracker + Cooldown ──
    print("  v6 +Cooldown...", end=" ")
    mgr._cooldowns = []
    for _ in range(MAX_FRAMES):
        for __ in range(iv): cap.grab()
        ok, frame = cap.retrieve()
        if not ok: break
        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if brightness < 40: conf = 0.08
        elif brightness < 80: conf = 0.1
        else: conf = 0.3
        r = model.track(frame, persist=True, verbose=False, classes=list(KEEP.keys()), conf=conf, tracker="bytetrack.yaml")
        boxes = r[0].boxes
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                st, nq, rs = mgr.update(int(box.id.item()), KEEP.get(int(box.cls.item()),"unk"), (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)
                if nq: calls += 1
        calls += len(mgr.mark_missing())
    video_data["v6_cooldown"] = int(calls)
    print(f"{calls} calls")

    # Reset
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── v7: Grid ──
    print("  v7 Grid...", end=" ")
    grid = SpatialTemporalGrid(w, h)
    calls = 0; hot = []
    for f in range(MAX_FRAMES):
        for __ in range(iv): cap.grab()
        ok, frame = cap.retrieve()
        if not ok: break
        brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if brightness < 40: conf = 0.08
        elif brightness < 80: conf = 0.1
        else: conf = 0.3
        r = model.track(frame, persist=True, verbose=False, classes=list(KEEP.keys()), conf=conf, tracker="bytetrack.yaml")
        boxes = r[0].boxes
        tracks = []
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                tracks.append({'tid': int(box.id.item()), 'cls': KEEP.get(int(box.cls.item()),"unk"),
                              'cx': (x1+x2)/2, 'cy': (y1+y2)/2, 'w': x2-x1, 'h': y2-y1})
        new_alerts = grid.update(tracks)
        # 模拟回调：标记已报警，加入 zones 拦截重复
        for c in new_alerts:
            if not c.alerted:
                calls += 1
                c.alerted = True
                grid._alerted_zones.append((c.cx, c.cy, time.time() + 3600))
        mc = max((c.count for row in grid.grid for c in row), default=0)
        hot.append(mc)

    video_data["v7_grid"] = int(calls)
    video_data["v7_max_count"] = int(max(hot)) if hot else 0
    print(f"{calls} calls, max_count={video_data['v7_max_count']}")

    results[vname] = video_data
    cap.release()

Path("results").mkdir(exist_ok=True)
with open("results/batch_exp.json","w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print(f"{'Video':<15} {'v5(Tracker)':>12} {'v6(+Cool)':>12} {'v7(Grid)':>12}")
print("-"*60)
for vname, d in results.items():
    print(f"{vname:<15} {d['v5_tracker']:>12} {d['v6_cooldown']:>12} {d['v7_grid']:>12}")
print("="*60)
