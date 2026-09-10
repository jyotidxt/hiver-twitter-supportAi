"""Quality filter module for the preprocessing pipeline.

Removes unusable records including empty tweets, duplicates, system
messages, and extremely short conversations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig

from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)

# Patterns indicating system or deleted messages
SYSTEM_PATTERNS = [
    "this tweet is unavailable",
    "this account doesn't exist",
    "this tweet has been deleted",
    "tweet unavailable",
    "media is unavailable",
    "account suspended",
]


def apply_quality_filters(
    df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply quality filters to remove unusable records.

    Filters applied:
    1. Remove empty/null text
    2. Remove system/deleted messages
    3. Remove duplicate messages
    4. Remove conversations shorter than minimum length

    Every removal is counted and reported.

    Args:
        df: DataFrame with conversation threads.
        config: Pipeline configuration with quality settings.

    Returns:
        A tuple of:
        - Filtered DataFrame
        - Dictionary with removal counts per filter
    """
    log_section(logger, "QUALITY FILTERS")

    removal_counts: dict[str, int] = {}
    initial_count = len(df)
    initial_conversations = (
        df["conversation_id"].nunique() if "conversation_id" in df.columns else 0
    )

    logger.info(f"  Starting with {initial_count:,} messages")
    logger.info(
        f"  Starting with {initial_conversations:,} conversations"
    )

    # Filter 1: Remove empty/null text
    if config.remove_empty:
        empty_mask = df["text"].isna() | (df["text"].str.strip() == "")
        empty_count = empty_mask.sum()
        df = df[~empty_mask].copy()
        removal_counts["empty_text"] = int(empty_count)
        logger.info(
            f"  {Colors.YELLOW}✗ Empty text removed:{Colors.RESET} {empty_count:,}"
        )

    # Filter 2: Remove system/deleted messages
    if config.remove_system_messages:
        system_mask = df["text"].str.lower().apply(
            lambda x: any(p in str(x) for p in SYSTEM_PATTERNS)
        )
        system_count = system_mask.sum()
        df = df[~system_mask].copy()
        removal_counts["system_messages"] = int(system_count)
        logger.info(
            f"  {Colors.YELLOW}✗ System/deleted removed:{Colors.RESET} {system_count:,}"
        )

    # Filter 3: Remove messages shorter than minimum length
    short_mask = df["text"].str.len() < config.min_text_length
    short_count = short_mask.sum()
    df = df[~short_mask].copy()
    removal_counts["too_short"] = int(short_count)
    logger.info(
        f"  {Colors.YELLOW}✗ Too short removed:{Colors.RESET} {short_count:,}"
    )

    # Filter 4: Remove duplicate messages (same author, same text)
    if config.remove_duplicates:
        pre_dedup = len(df)
        df = df.drop_duplicates(
            subset=["author_id", "text"], keep="first"
        ).copy()
        dup_count = pre_dedup - len(df)
        removal_counts["duplicates"] = int(dup_count)
        logger.info(
            f"  {Colors.YELLOW}✗ Duplicates removed:{Colors.RESET} {dup_count:,}"
        )

    # Filter 5: Remove conversations below minimum thread length
    if "conversation_id" in df.columns:
        thread_sizes = df.groupby("conversation_id").size()
        short_threads = thread_sizes[
            thread_sizes < config.minimum_thread_length
        ].index
        short_thread_msgs = df["conversation_id"].isin(short_threads)
        short_thread_count = short_thread_msgs.sum()
        df = df[~short_thread_msgs].copy()
        removal_counts["short_conversations"] = int(short_thread_count)
        removal_counts["short_conversations_threads"] = int(len(short_threads))
        logger.info(
            f"  {Colors.YELLOW}✗ Short conversations:{Colors.RESET} "
            f"{len(short_threads):,} threads ({short_thread_count:,} messages)"
        )

    # Summary
    total_removed = initial_count - len(df)
    final_conversations = (
        df["conversation_id"].nunique() if "conversation_id" in df.columns else 0
    )

    log_section(logger, "QUALITY FILTER RESULTS")
    log_metric(logger, "Messages before filtering", f"{initial_count:,}")
    log_metric(logger, "Messages after filtering", f"{len(df):,}")
    log_metric(logger, "Total messages removed", f"{total_removed:,}")
    log_metric(
        logger,
        "Removal rate",
        f"{total_removed / initial_count * 100:.1f}%" if initial_count > 0 else "0%",
    )
    log_metric(logger, "Conversations before", f"{initial_conversations:,}")
    log_metric(logger, "Conversations after", f"{final_conversations:,}")

    return df.reset_index(drop=True), removal_counts
