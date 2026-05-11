"""
utils/logger.py – Structured logging configuration.

Uses Rich for coloured console output and writes plain-text to file.
"""

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

_LOG_FILE = Path("./output/audio_analysis.log")
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        # Pretty console output via Rich
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            show_path=False,
        ),
        # Plain-text log file
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for the given module."""
    return logging.getLogger(name)
