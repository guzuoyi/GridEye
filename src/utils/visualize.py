"""抽帧可视化工具 —— 将抽帧结果写入临时目录供人工确认"""

import cv2
from pathlib import Path
from src.sampler.sampler import SampledFrame


def save_frames_to_dir(
    frames: list[SampledFrame],
    output_dir: str = "debug_frames",
    prefix: str = "frame",
    draw_info: bool = True,
) -> list[Path]:
    """将抽帧结果写入目录。

    参数:
        frames: 抽帧列表
        output_dir: 输出目录
        prefix: 文件名前缀
        draw_info: 是否在图像上叠加帧号+时间戳

    返回:
        写入的文件路径列表
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for f in frames:
        img = f.image.copy()
        if draw_info:
            text = f"idx={f.frame_idx}  ts={f.timestamp:.2f}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        filename = f"{prefix}_{f.frame_idx:04d}.jpg"
        filepath = out / filename
        cv2.imwrite(str(filepath), img)
        paths.append(filepath)

    return paths


def save_frame(frame: SampledFrame, output_path: str) -> None:
    """保存单帧图像。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame.image)
