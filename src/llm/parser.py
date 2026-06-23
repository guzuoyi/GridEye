"""Qwen 响应解析器 —— JSON 提取 + 正则回退"""

import json
import re
from typing import Optional
from logging import getLogger

logger = getLogger(__name__)


def extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出中提取 JSON 对象。

    策略:
        1. 直接 json.loads(text)
        2. 正则匹配第一个 {...} 块
        3. 失败返回 None
    """
    text = text.strip()
    if not text:
        return None

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 正则在 text 中找最外层 {...}
    # 支持嵌套的简单处理：找第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 最后尝试提取所有合法的 key:value
    logger.warning(f"Failed to parse JSON from: {text[:100]}")
    return None


def parse_event_confirm(raw_text: str) -> dict:
    """解析事件确认响应。

    返回:
        {"is_anomaly": bool, "event_type": str, "confidence": float, "reasoning": str}
        解析失败返回 is_anomaly=False
    """
    d = extract_json(raw_text) or {}
    return {
        "is_anomaly": bool(d.get("is_anomaly", False)),
        "event_type": str(d.get("event_type", "正常")),
        "confidence": float(d.get("confidence", 0.0)),
        "reasoning": str(d.get("reasoning", "")),
    }


def parse_state_refresh(raw_text: str) -> dict:
    """解析状态刷新响应。"""
    d = extract_json(raw_text) or {}
    return {
        "status_changed": bool(d.get("status_changed", False)),
        "updated_event_type": str(d.get("updated_event_type", "")),
        "confidence": float(d.get("confidence", 0.0)),
        "reasoning": str(d.get("reasoning", "")),
    }


def parse_risk_clearance(raw_text: str) -> dict:
    """解析风险消除响应。"""
    d = extract_json(raw_text) or {}
    return {
        "risk_cleared": bool(d.get("risk_cleared", False)),
        "need_cleanup": bool(d.get("need_cleanup", False)),
        "confidence": float(d.get("confidence", 0.0)),
        "reasoning": str(d.get("reasoning", "")),
    }
