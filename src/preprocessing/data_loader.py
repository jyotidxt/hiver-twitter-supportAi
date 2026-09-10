"""Data loader module for the preprocessing pipeline.

Responsible for loading raw CSV data, validating schema, and reporting
dataset statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig

from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)

# Expected columns in the raw dataset
EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def load_raw_data(config: PipelineConfig) -> pd.DataFrame:
    """Load and validate the raw Twitter support dataset.

    This function reads the CSV file, validates the expected columns exist,
    reports dataset statistics, and detects missing values.

    Args:
        config: Pipeline configuration with data paths.

    Returns:
        A pandas DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    raw_path = config.raw_data_path

    log_section(logger, "LOADING RAW DATA")
    logger.info(f"  Source: {raw_path}")

    # Check file exists
    if not raw_path.exists():
        logger.error(f"Raw data file not found: {raw_path}")
        logger.error(
            "Please download the dataset from Kaggle:\n"
            "  kaggle datasets download -d thoughtvector/customer-support-on-twitter\n"
            f"  Extract 'twcs.csv' to {raw_path.parent}"
        )
        sys.exit(1)

    # Load with progress indication
    logger.info("  Reading CSV file...")
    try:
        df = pd.read_csv(
            raw_path,
            dtype={
                "tweet_id": str,
                "author_id": str,
                "inbound": object,
                "text": str,
                "response_tweet_id": str,
                "in_response_to_tweet_id": str,
            },
            parse_dates=["created_at"],
            low_memory=False,
        )
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    # Validate columns
    _validate_columns(df)

    # Clean up 'inbound' column to boolean
    df["inbound"] = df["inbound"].map(
        {"True": True, "False": False, True: True, False: False}
    )

    # Report statistics
    _report_statistics(df)

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Validate that all expected columns exist in the DataFrame.

    Args:
        df: The loaded DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.error(f"Found columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing}")
    logger.info(f"  {Colors.GREEN}✓ All expected columns present{Colors.RESET}")


def _report_statistics(df: pd.DataFrame) -> None:
    """Report dataset shape and missing value statistics.

    Args:
        df: The loaded DataFrame.
    """
    log_section(logger, "RAW DATASET STATISTICS")
    log_metric(logger, "Total rows", f"{len(df):,}")
    log_metric(logger, "Total columns", f"{len(df.columns)}")
    log_metric(logger, "Memory usage", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Missing values
    logger.info("")
    logger.info(f"  {Colors.BOLD}Missing Values:{Colors.RESET}")
    for col in EXPECTED_COLUMNS:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        if missing_count > 0:
            logger.info(
                f"    {col:<30} {missing_count:>8,} ({missing_pct:.1f}%)"
            )
        else:
            logger.info(
                f"    {col:<30} {Colors.GREEN}{'none':>8}{Colors.RESET}"
            )

    # Inbound/outbound split
    inbound_count = df["inbound"].sum() if df["inbound"].dtype == bool else 0
    outbound_count = len(df) - inbound_count
    logger.info("")
    log_metric(logger, "Customer tweets (inbound)", f"{inbound_count:,}")
    log_metric(logger, "Brand tweets (outbound)", f"{outbound_count:,}")
    log_metric(logger, "Unique authors", f"{df['author_id'].nunique():,}")
