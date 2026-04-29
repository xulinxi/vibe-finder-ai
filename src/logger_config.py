"""
Structured logging setup for Vibe Finder AI.

Logs flow to two destinations:
    1. Console (INFO+)  — readable summary for the operator
    2. logs/vibe_finder.log (DEBUG+) — full audit trail for debugging

Every pipeline stage emits at least one INFO log so the agent's reasoning
is observable end-to-end.
"""

import logging
import sys
from pathlib import Path

from src.config import LOGS_DIR

_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)-14s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(level: str = "INFO", log_file: str = "vibe_finder.log") -> logging.Logger:
    """Configure the root vibe_finder logger. Idempotent — safe to call multiple times."""
    global _initialized

    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / log_file

    logger = logging.getLogger("vibe_finder")

    if _initialized:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    _initialized = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger that inherits the vibe_finder configuration."""
    return logging.getLogger(f"vibe_finder.{name}")
