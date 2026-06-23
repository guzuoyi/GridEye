#!/usr/bin/env python3
"""高速公路交通事件二次识别系统 —— 运行入口

用法:
    python run.py                          # 使用 config/default.yaml
    python run.py --config my_config.yaml  # 自定义配置
    python run.py --source data/test.mp4   # 指定视频文件
    python run.py --source 0               # 使用摄像头
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import ConfigLoader
from src.logging_config import setup_logging
from src.pipeline.pipeline import MainPipeline


def main():
    parser = argparse.ArgumentParser(description="高速公路交通事件二次识别系统")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--source", default=None, help="视频文件/RTSP/摄像头ID")
    parser.add_argument("--max-frames", type=int, default=0, help="最大处理帧数(0=无限)")
    parser.add_argument("--no-viz", action="store_true", help="禁用可视化窗口")
    args = parser.parse_args()

    # 加载配置
    cfg = ConfigLoader(args.config).load()
    if args.source:
        cfg.video.source = args.source
    setup_logging("INFO")

    print("=" * 60)
    print("  Highway Traffic Event Detection System")
    print("=" * 60)
    print(f"  Video:  {cfg.video.source}")
    print(f"  YOLO:   {cfg.yolo.model_path}")
    print(f"  Qwen:   {cfg.qwen.model_name} @ {cfg.qwen.base_url}")
    print(f"  FPS:    {cfg.video.fps}")
    print("=" * 60)

    pipeline = MainPipeline(cfg)
    pipeline.setup()

    try:
        pipeline.run(max_frames=args.max_frames, visualize=not args.no_viz)
    except KeyboardInterrupt:
        print("\n[System] Interrupted by user")
    finally:
        pipeline.stop()
        print("\n" + "=" * 60)
        print("  Stats")
        print("=" * 60)
        print(f"  Frames:        {pipeline.total_frames}")
        print(f"  Qwen calls:    {pipeline.total_qwen_calls}")
        print(f"  Reduction:     {pipeline.qwen_call_reduction:.1%}")
        print("=" * 60)


if __name__ == "__main__":
    main()
