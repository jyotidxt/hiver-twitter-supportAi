"""Logging utilities for the preprocessing pipeline.

Provides a configured logger with colored terminal output and file logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


# ANSI color codes for terminal output
class Colors:
    """ANSI escape codes for colored terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log level names."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colored level name."""
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        record.levelname = f"{color}{record.levelname:<8}{Colors.RESET}"
        return super().format(record)


def get_logger(
    name: str = "hiver-pipeline",
    level: str = "INFO",
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Create and configure a logger with colored terminal output.

    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file for persistent logging.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_fmt = ColoredFormatter(
        fmt="%(levelname)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (no colors)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    """Log a visually distinct section header.

    Args:
        logger: The logger instance.
        title: Section title text.
    """
    separator = "═" * 60
    logger.info("")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}{separator}{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}{separator}{Colors.RESET}")


def log_metric(logger: logging.Logger, label: str, value: object) -> None:
    """Log a key-value metric in a formatted way.

    Args:
        logger: The logger instance.
        label: The metric label.
        value: The metric value.
    """
    logger.info(f"  {Colors.BOLD}{label:<35}{Colors.RESET} {value}")
