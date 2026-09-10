"""Brand filter module for the preprocessing pipeline.

Filters the raw dataset to retain only conversations involving the
configured target brand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig

from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)


def filter_brand(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Filter the dataset to keep only conversations for the selected brand.

    This function identifies all tweets authored by the target brand and all
    customer tweets that are part of conversations with that brand.

    Args:
        df: The raw DataFrame containing all tweets.
        config: Pipeline configuration with brand selection.

    Returns:
        A filtered DataFrame containing only brand-relevant tweets.
    """
    brand = config.selected_brand
    initial_count = len(df)

    log_section(logger, f"FILTERING FOR BRAND: {brand}")

    # Step 1: Find all brand tweets (outbound from the target brand)
    brand_mask = df["author_id"].str.lower() == brand.lower()
    brand_tweets = df[brand_mask]
    logger.info(f"  Brand tweets found: {len(brand_tweets):,}")

    if len(brand_tweets) == 0:
        logger.error(f"No tweets found for brand '{brand}'.")
        logger.error("Available brands (top 20 by volume):")
        outbound = df[df["inbound"] == False]
        top_brands = outbound["author_id"].value_counts().head(20)
        for b, count in top_brands.items():
            logger.error(f"    {b}: {count:,}")
        raise ValueError(f"Brand '{brand}' not found in dataset.")

    # Step 2: Find tweet_ids that the brand is responding to
    brand_responding_to = set(
        brand_tweets["in_response_to_tweet_id"].dropna().unique()
    )

    # Step 3: Find tweet_ids in response_tweet_id of brand tweets
    brand_response_ids = set()
    for ids in brand_tweets["response_tweet_id"].dropna():
        for tid in str(ids).split(","):
            tid = tid.strip()
            if tid:
                brand_response_ids.add(tid)

    # Step 4: Find customer tweets that are in conversations with the brand
    # A customer tweet is relevant if:
    #   - The brand replied to it (customer tweet_id is in brand's in_response_to_tweet_id)
    #   - The customer is replying to a brand tweet (customer's in_response_to_tweet_id is a brand tweet_id)
    brand_tweet_ids = set(brand_tweets["tweet_id"].unique())

    customer_mask = (
        df["tweet_id"].isin(brand_responding_to)
        | df["in_response_to_tweet_id"].isin(brand_tweet_ids)
    )

    # Combine brand tweets and related customer tweets
    filtered_df = df[brand_mask | customer_mask].copy()

    # Step 5: Iteratively expand to capture full conversation chains
    # Some conversations have multiple back-and-forth exchanges
    prev_size = 0
    iteration = 0
    while len(filtered_df) != prev_size and iteration < 10:
        prev_size = len(filtered_df)
        iteration += 1

        all_tweet_ids = set(filtered_df["tweet_id"].unique())
        all_response_to = set(
            filtered_df["in_response_to_tweet_id"].dropna().unique()
        )

        # Expand response IDs
        expanded_response_ids = set()
        for ids in filtered_df["response_tweet_id"].dropna():
            for tid in str(ids).split(","):
                tid = tid.strip()
                if tid:
                    expanded_response_ids.add(tid)

        # Find additional connected tweets
        expand_mask = (
            df["tweet_id"].isin(all_response_to)
            | df["tweet_id"].isin(expanded_response_ids)
            | df["in_response_to_tweet_id"].isin(all_tweet_ids)
        )
        filtered_df = df[brand_mask | customer_mask | expand_mask].copy()

    removed_count = initial_count - len(filtered_df)

    # Report results
    log_section(logger, "BRAND FILTER RESULTS")
    log_metric(logger, "Original dataset size", f"{initial_count:,}")
    log_metric(logger, "Filtered dataset size", f"{len(filtered_df):,}")
    log_metric(logger, "Records removed", f"{removed_count:,}")
    log_metric(
        logger,
        "Retention rate",
        f"{len(filtered_df) / initial_count * 100:.1f}%",
    )
    log_metric(logger, "Expansion iterations", f"{iteration}")

    # Metadata
    n_brand = filtered_df[brand_mask[filtered_df.index]].shape[0] if brand_mask.any() else 0
    n_customer = len(filtered_df) - n_brand
    log_metric(logger, "Brand messages in filtered set", f"{n_brand:,}")
    log_metric(logger, "Customer messages in filtered set", f"{n_customer:,}")

    return filtered_df.reset_index(drop=True)
