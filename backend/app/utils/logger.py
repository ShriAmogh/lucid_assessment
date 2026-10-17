"""
Structured logging module for LucidPlus OCR.
Provides timestamped, level-formatted logs with contextual metadata.
"""

import sys
import logging
from typing import Optional


def get_logger(name: str = "lucid_ocr", level: Optional[int] = None) -> logging.Logger:
    """Returns a configured logger with standard formatting."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level or logging.INFO)
        logger.propagate = False

    return logger
