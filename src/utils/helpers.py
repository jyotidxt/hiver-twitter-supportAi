"""Shared helper utilities for the Hiver Support AI pipeline.

Provides reusable functions used across multiple modules to avoid
code duplication.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create a directory and all parent directories if they don't exist.

    Args:
        path: The directory path to create.

    Returns:
        The same path for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(
    path: Path,
    dtype_overrides: dict[str, type] | None = None,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV file with error handling and sensible defaults.

    Args:
        path: Path to the CSV file.
        dtype_overrides: Optional dictionary of column name to dtype.
        parse_dates: Optional list of columns to parse as dates.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    kwargs: dict[str, Any] = {"low_memory": False, "encoding": "utf-8"}
    if dtype_overrides:
        kwargs["dtype"] = dtype_overrides
    if parse_dates:
        kwargs["parse_dates"] = parse_dates

    return pd.read_csv(path, **kwargs)


def safe_write_json(
    data: dict[str, Any] | list[Any],
    path: Path,
    indent: int = 2,
) -> Path:
    """Write data to a JSON file with directory creation.

    Args:
        data: The data to serialize.
        path: Output file path.
        indent: JSON indentation level.

    Returns:
        The path of the written file.
    """
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    return path


def timestamp_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format.

    Returns:
        ISO formatted UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length with ellipsis.

    Args:
        text: The text to truncate.
        max_length: Maximum character length.

    Returns:
        Truncated text with '...' appended if needed.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_number(n: int | float) -> str:
    """Format a number with comma separators.

    Args:
        n: The number to format.

    Returns:
        Formatted string with commas.
    """
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def calculate_percentage(part: int | float, total: int | float) -> float:
    """Safely calculate percentage avoiding division by zero.

    Args:
        part: The numerator.
        total: The denominator.

    Returns:
        Percentage value rounded to 1 decimal, or 0.0 if total is zero.
    """
    if total == 0:
        return 0.0
    return round(part / total * 100, 1)
