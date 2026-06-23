"""统一日志配置"""

import logging


def setup_logging(level: str = "INFO", fmt: str = None) -> None:
    """配置全局 logging 格式。

    参数:
        level: 日志级别 (DEBUG / INFO / WARNING / ERROR)
        fmt: 自定义格式字符串，默认含时间戳+模块名+级别
    """
    if fmt is None:
        fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger，避免重复添加 handler。"""
    return logging.getLogger(name)
