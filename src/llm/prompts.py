"""Prompt 模板 —— 三种调用场景"""

# =============================================================================
# 场景一：事件确认（有图片）
# =============================================================================

SYSTEM_EVENT_CONFIRM = """You are a highway traffic incident analysis expert. Analyze the given image and context to determine if an anomaly exists.

Output ONLY valid JSON, no extra text. Format:
{"is_anomaly": true/false, "event_type": "违停|抛锚|事故|遗撒|正常", "confidence": 0.0-1.0, "reasoning": "brief reason"}"""


def prompt_event_confirm(
    class_name: str,
    lane_number: int,
    distance_m: float,
    dwell_frames: int,
    dwell_seconds: float,
) -> str:
    """构建事件确认 prompt。

    参数:
        class_name: 目标类别 (car/truck/person/two-wheeler)
        lane_number: 车道号 (≥1 为行车道)
        distance_m: 纵向距离 (米)
        dwell_frames: 连续驻留帧数
        dwell_seconds: 连续驻留秒数
    """
    return (
        f"Context: A {class_name} is in lane {lane_number} at {distance_m:.0f}m distance, "
        f"continuously present for {dwell_frames} frames ({dwell_seconds:.0f}s). "
        f"Analyze if this is an anomaly."
    )


# =============================================================================
# 场景二：状态刷新（有图片）
# =============================================================================

SYSTEM_STATE_REFRESH = """You are a highway traffic incident analyst. Re-evaluate a previously confirmed incident with the latest image.

Output ONLY JSON:
{"status_changed": true/false, "updated_event_type": "违停|抛锚|事故|遗撒|正常", "confidence": 0.0-1.0, "reasoning": "brief"}"""


def prompt_state_refresh(
    class_name: str,
    lane_number: int,
    event_type: str,
    elapsed_seconds: float,
) -> str:
    """构建状态刷新 prompt。"""
    return (
        f"Previously confirmed {class_name} {event_type} in lane {lane_number}, "
        f"persisting for {elapsed_seconds:.0f}s. "
        f"Has the situation changed? Re-evaluate with current image."
    )


# =============================================================================
# 场景三：风险消除确认（纯文本，无图片）
# =============================================================================

SYSTEM_RISK_CLEARANCE = """You are a highway traffic incident analyst. Determine if a previously confirmed incident has been resolved now that the target has disappeared.

Output ONLY JSON:
{"risk_cleared": true/false, "need_cleanup": true/false, "confidence": 0.0-1.0, "reasoning": "brief"}"""


def prompt_risk_clearance(
    class_name: str,
    lane_number: int,
    distance_m: float,
    event_type: str,
    confirmed_time: str,
    disappeared_time: str,
) -> str:
    """构建风险消除 prompt。"""
    return (
        f"Previous incident: {class_name} {event_type} in lane {lane_number} at {distance_m:.0f}m. "
        f"Confirmed at {confirmed_time}, disappeared at {disappeared_time}. "
        f"The target is no longer visible. Has the risk been cleared? "
        f"Does the scene need cleanup?"
    )
