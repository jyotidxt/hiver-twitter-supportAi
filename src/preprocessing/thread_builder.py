"""Thread reconstruction module for the preprocessing pipeline.

Rebuilds complete customer ↔ brand conversation threads from flat tweet data
using tweet reply relationships.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig

from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)


def build_threads(
    df: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    """Reconstruct conversation threads from flat tweet data.

    Each thread is a sequence of messages between a customer and the brand,
    ordered chronologically. Threads are identified by tracing
    in_response_to_tweet_id chains back to the root customer message.

    Args:
        df: The brand-filtered DataFrame.
        config: Pipeline configuration with thread settings.

    Returns:
        A DataFrame with an added 'conversation_id' column and a 'role'
        column ('customer' or 'brand'), sorted by conversation_id
        and chronological order.
    """
    log_section(logger, "RECONSTRUCTING CONVERSATION THREADS")

    brand = config.selected_brand.lower()
    show_progress = config.show_progress_bars

    # Build lookup structures
    tweet_lookup: dict[str, int] = {}
    for idx, row in df.iterrows():
        tweet_lookup[str(row["tweet_id"])] = idx

    # Build response mapping: tweet_id -> list of response tweet_ids
    response_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        tid = str(row["tweet_id"])
        resp_to = row["in_response_to_tweet_id"]
        if pd.notna(resp_to):
            parent_id = str(resp_to).strip()
            if parent_id not in response_map:
                response_map[parent_id] = []
            response_map[parent_id].append(tid)

    # Find root tweets (tweets with no in_response_to_tweet_id, or whose
    # parent is not in our filtered dataset)
    root_tweets = []
    for _, row in df.iterrows():
        resp_to = row["in_response_to_tweet_id"]
        if pd.isna(resp_to) or str(resp_to).strip() not in tweet_lookup:
            root_tweets.append(str(row["tweet_id"]))

    logger.info(f"  Root tweets identified: {len(root_tweets):,}")

    # Build conversation threads by traversing from each root
    conversations: list[dict] = []
    conversation_id = 0
    max_depth = config.max_thread_depth

    iterator = tqdm(
        root_tweets,
        desc="  Building threads",
        disable=not show_progress,
        unit="thread",
    )

    for root_id in iterator:
        thread_tweets = _collect_thread(
            root_id, response_map, tweet_lookup, df, max_depth
        )

        if len(thread_tweets) > 0:
            for tweet_data in thread_tweets:
                tweet_data["conversation_id"] = f"conv_{conversation_id:06d}"
            conversations.extend(thread_tweets)
            conversation_id += 1

    # Create result DataFrame
    if not conversations:
        logger.warning("No conversations were reconstructed.")
        return pd.DataFrame()

    result_df = pd.DataFrame(conversations)

    # Add role column
    result_df["role"] = result_df["author_id"].apply(
        lambda x: "brand" if str(x).lower() == brand else "customer"
    )

    # Sort by conversation_id and timestamp
    if "created_at" in result_df.columns and result_df["created_at"].notna().any():
        result_df = result_df.sort_values(
            ["conversation_id", "created_at"], ascending=[True, True]
        ).reset_index(drop=True)
    else:
        result_df = result_df.sort_values(
            ["conversation_id", "tweet_id"], ascending=[True, True]
        ).reset_index(drop=True)

    # Report statistics
    _report_thread_stats(result_df)

    return result_df


def _collect_thread(
    root_id: str,
    response_map: dict[str, list[str]],
    tweet_lookup: dict[str, int],
    df: pd.DataFrame,
    max_depth: int,
) -> list[dict]:
    """Collect all tweets in a conversation thread via BFS.

    Args:
        root_id: The tweet_id of the root (starting) tweet.
        response_map: Mapping from parent tweet_id to child tweet_ids.
        tweet_lookup: Mapping from tweet_id to DataFrame index.
        df: The source DataFrame.
        max_depth: Maximum traversal depth to prevent infinite loops.

    Returns:
        A list of dictionaries, each representing a tweet in the thread.
    """
    thread: list[dict] = []
    queue: list[tuple[str, int]] = [(root_id, 0)]  # (tweet_id, depth)
    visited: set[str] = set()

    while queue:
        current_id, depth = queue.pop(0)

        if current_id in visited or depth > max_depth:
            continue
        visited.add(current_id)

        if current_id in tweet_lookup:
            idx = tweet_lookup[current_id]
            row = df.loc[idx]
            thread.append({
                "tweet_id": str(row["tweet_id"]),
                "author_id": str(row["author_id"]),
                "inbound": row["inbound"],
                "created_at": row["created_at"],
                "text": row["text"] if pd.notna(row["text"]) else "",
                "in_response_to_tweet_id": (
                    str(row["in_response_to_tweet_id"])
                    if pd.notna(row["in_response_to_tweet_id"])
                    else None
                ),
                "response_tweet_id": (
                    str(row["response_tweet_id"])
                    if pd.notna(row["response_tweet_id"])
                    else None
                ),
            })

        # Add children to queue
        if current_id in response_map:
            for child_id in response_map[current_id]:
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

    return thread


def _report_thread_stats(df: pd.DataFrame) -> None:
    """Report statistics about reconstructed conversation threads.

    Args:
        df: The DataFrame with conversation threads.
    """
    thread_sizes = df.groupby("conversation_id").size()
    role_counts = df.groupby("conversation_id")["role"].value_counts().unstack(fill_value=0)

    log_section(logger, "THREAD RECONSTRUCTION RESULTS")
    log_metric(logger, "Total conversations", f"{thread_sizes.shape[0]:,}")
    log_metric(logger, "Total messages", f"{len(df):,}")
    log_metric(logger, "Avg messages per thread", f"{thread_sizes.mean():.1f}")
    log_metric(logger, "Max thread length", f"{thread_sizes.max()}")
    log_metric(logger, "Min thread length", f"{thread_sizes.min()}")
    log_metric(logger, "Median thread length", f"{thread_sizes.median():.0f}")

    if "brand" in role_counts.columns:
        avg_brand = role_counts["brand"].mean()
        log_metric(logger, "Avg brand replies per thread", f"{avg_brand:.1f}")
    if "customer" in role_counts.columns:
        avg_customer = role_counts["customer"].mean()
        log_metric(logger, "Avg customer msgs per thread", f"{avg_customer:.1f}")

    # Distribution
    logger.info("")
    logger.info(f"  {Colors.BOLD}Thread Length Distribution:{Colors.RESET}")
    bins = [1, 2, 3, 4, 5, 10, 20, float("inf")]
    labels = ["1", "2", "3", "4", "5-9", "10-19", "20+"]
    dist = pd.cut(thread_sizes, bins=bins, labels=labels, right=False).value_counts().sort_index()
    for label, count in dist.items():
        bar = "█" * min(int(count / thread_sizes.shape[0] * 50), 50)
        logger.info(f"    {label:>6} messages: {count:>6,}  {bar}")
