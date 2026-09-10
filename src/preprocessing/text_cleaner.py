"""Text cleaning module for the preprocessing pipeline.

Cleans and normalizes tweet text while preserving customer tone,
emojis, and meaningful punctuation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig

from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)

# Regex patterns
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+", re.IGNORECASE
)
RT_PATTERN = re.compile(r"^RT\s+@\w+:?\s*", re.IGNORECASE)
LEADING_MENTION_PATTERN = re.compile(r"^(@\w+\s*)+")
MULTI_SPACE_PATTERN = re.compile(r"\s{2,}")
NEWLINE_PATTERN = re.compile(r"[\n\r]+")


def clean_text(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Apply text cleaning transformations to the tweet text column.

    Cleaning steps (all configurable via config):
    1. Remove URLs
    2. Remove RT prefixes
    3. Strip leading @mentions (common in reply tweets)
    4. Normalize whitespace
    5. Optionally lowercase
    6. Preserve emojis and meaningful punctuation

    Args:
        df: DataFrame with a 'text' column.
        config: Pipeline configuration with preprocessing flags.

    Returns:
        DataFrame with cleaned 'text' column and original text preserved
        in 'text_original'.
    """
    log_section(logger, "TEXT CLEANING")

    show_progress = config.show_progress_bars

    # Preserve original text
    df = df.copy()
    df["text_original"] = df["text"]

    total = len(df)
    logger.info(f"  Processing {total:,} messages...")

    # Apply cleaning
    texts = df["text"].fillna("").astype(str).tolist()

    cleaned_texts: list[str] = []
    changes_count = 0

    iterator = tqdm(
        texts,
        desc="  Cleaning text",
        disable=not show_progress,
        unit="msg",
    )

    for text in iterator:
        original = text
        cleaned = _clean_single(text, config)
        if cleaned != original:
            changes_count += 1
        cleaned_texts.append(cleaned)

    df["text"] = cleaned_texts

    # Report
    log_section(logger, "TEXT CLEANING RESULTS")
    log_metric(logger, "Total messages processed", f"{total:,}")
    log_metric(logger, "Messages modified", f"{changes_count:,}")
    log_metric(
        logger,
        "Modification rate",
        f"{changes_count / total * 100:.1f}%" if total > 0 else "0%",
    )

    # Show sample changes
    _show_samples(df, n=3)

    return df


def _clean_single(text: str, config: PipelineConfig) -> str:
    """Clean a single text string based on configuration.

    Args:
        text: The raw tweet text.
        config: Pipeline configuration.

    Returns:
        The cleaned text string.
    """
    if not text or text.strip() == "":
        return ""

    # Remove URLs
    if config.remove_urls:
        text = URL_PATTERN.sub("", text)

    # Remove RT prefix
    if config.remove_rt_prefix:
        text = RT_PATTERN.sub("", text)

    # Strip leading @mentions (but keep @mentions within the text body)
    if config.strip_mentions_from_start:
        text = LEADING_MENTION_PATTERN.sub("", text)

    # Normalize newlines to spaces
    text = NEWLINE_PATTERN.sub(" ", text)

    # Normalize whitespace
    if config.normalize_whitespace:
        text = MULTI_SPACE_PATTERN.sub(" ", text)

    # Lowercase (optional)
    if config.lowercase:
        text = text.lower()

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def _show_samples(df: pd.DataFrame, n: int = 3) -> None:
    """Show sample before/after cleaning examples.

    Args:
        df: DataFrame with 'text' and 'text_original' columns.
        n: Number of samples to show.
    """
    changed = df[df["text"] != df["text_original"]]
    if len(changed) == 0:
        return

    samples = changed.sample(n=min(n, len(changed)), random_state=42)

    logger.info("")
    logger.info(f"  {Colors.BOLD}Sample Transformations:{Colors.RESET}")
    for _, row in samples.iterrows():
        original = str(row["text_original"])[:80]
        cleaned = str(row["text"])[:80]
        logger.info(f"    {Colors.RED}BEFORE:{Colors.RESET} {original}...")
        logger.info(f"    {Colors.GREEN}AFTER: {Colors.RESET} {cleaned}...")
        logger.info("")
