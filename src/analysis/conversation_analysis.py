"""Conversation structure analysis module.

Analyzes conversation patterns including turn-taking, response depth,
and participation balance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger, log_metric, log_section

logger = get_logger(__name__)


def analyze_conversations(
    df: pd.DataFrame, config: PipelineConfig
) -> dict[str, Any]:
    """Analyze conversation structure and patterns.

    Computes:
    - Messages per conversation distribution
    - Customer vs brand participation rates
    - Single-turn vs multi-turn classification
    - Response depth distribution
    - Brand responsiveness metrics

    Args:
        df: The processed conversations DataFrame.
        config: Pipeline configuration.

    Returns:
        Dictionary of conversation analysis results.
    """
    log_section(logger, "CONVERSATION STRUCTURE ANALYSIS")

    results: dict[str, Any] = {}

    # ── Messages per conversation ──
    thread_sizes = df.groupby("conversation_id").size()
    results["messages_per_conversation"] = {
        "distribution": thread_sizes.value_counts().sort_index().to_dict(),
        "percentiles": {
            "p10": round(float(thread_sizes.quantile(0.10)), 1),
            "p25": round(float(thread_sizes.quantile(0.25)), 1),
            "p50": round(float(thread_sizes.quantile(0.50)), 1),
            "p75": round(float(thread_sizes.quantile(0.75)), 1),
            "p90": round(float(thread_sizes.quantile(0.90)), 1),
            "p95": round(float(thread_sizes.quantile(0.95)), 1),
        },
    }

    # ── Customer vs Brand participation ──
    role_counts = (
        df.groupby(["conversation_id", "role"])
        .size()
        .unstack(fill_value=0)
    )

    participation: dict[str, Any] = {}
    for role in ["customer", "brand"]:
        if role in role_counts.columns:
            col = role_counts[role]
            participation[role] = {
                "mean": round(float(col.mean()), 2),
                "median": round(float(col.median()), 2),
                "max": int(col.max()),
                "conversations_with_zero": int((col == 0).sum()),
            }
    results["participation"] = participation

    log_metric(
        logger,
        "Avg customer msgs/thread",
        f"{participation.get('customer', {}).get('mean', 0):.2f}",
    )
    log_metric(
        logger,
        "Avg brand msgs/thread",
        f"{participation.get('brand', {}).get('mean', 0):.2f}",
    )

    # ── Single-turn vs Multi-turn ──
    single_turn = int((thread_sizes == 2).sum())  # Exactly 1 customer + 1 brand
    multi_turn = int((thread_sizes > 2).sum())
    total_convs = int(thread_sizes.shape[0])

    results["turn_structure"] = {
        "single_turn_count": single_turn,
        "multi_turn_count": multi_turn,
        "single_turn_pct": round(single_turn / total_convs * 100, 1) if total_convs > 0 else 0,
        "multi_turn_pct": round(multi_turn / total_convs * 100, 1) if total_convs > 0 else 0,
        "total": total_convs,
    }

    log_metric(logger, "Single-turn conversations", f"{single_turn:,} ({results['turn_structure']['single_turn_pct']}%)")
    log_metric(logger, "Multi-turn conversations", f"{multi_turn:,} ({results['turn_structure']['multi_turn_pct']}%)")

    # ── Response depth ──
    # How many back-and-forth exchanges per conversation
    depth_data: list[dict[str, Any]] = []
    for conv_id, group in df.groupby("conversation_id"):
        roles = group.sort_values("created_at")["role"].tolist() if "created_at" in group.columns else group["role"].tolist()
        # Count role switches as a proxy for depth
        switches = sum(
            1 for i in range(1, len(roles)) if roles[i] != roles[i - 1]
        )
        depth_data.append({
            "conversation_id": conv_id,
            "total_messages": len(group),
            "role_switches": switches,
            "depth": (switches + 1) // 2,  # Approximate exchange rounds
        })

    depth_df = pd.DataFrame(depth_data)
    results["response_depth"] = {
        "mean_switches": round(float(depth_df["role_switches"].mean()), 2),
        "median_switches": round(float(depth_df["role_switches"].median()), 2),
        "max_switches": int(depth_df["role_switches"].max()),
        "mean_depth": round(float(depth_df["depth"].mean()), 2),
        "depth_distribution": depth_df["depth"].value_counts().sort_index().head(15).to_dict(),
    }

    log_metric(logger, "Avg response depth", f"{depth_df['depth'].mean():.2f} exchanges")
    log_metric(logger, "Max response depth", f"{depth_df['depth'].max()} exchanges")

    # ── Brand responsiveness ──
    # Conversations where brand has 0 replies
    if "brand" in role_counts.columns:
        no_brand_reply = int((role_counts["brand"] == 0).sum())
        results["brand_responsiveness"] = {
            "conversations_without_brand_reply": no_brand_reply,
            "brand_reply_rate": round(
                (1 - no_brand_reply / total_convs) * 100, 1
            ) if total_convs > 0 else 0,
        }
        log_metric(
            logger,
            "Brand reply rate",
            f"{results['brand_responsiveness']['brand_reply_rate']}%",
        )

    # Multi-turn length buckets
    buckets = {
        "2 messages": int((thread_sizes == 2).sum()),
        "3-4 messages": int(((thread_sizes >= 3) & (thread_sizes <= 4)).sum()),
        "5-6 messages": int(((thread_sizes >= 5) & (thread_sizes <= 6)).sum()),
        "7-10 messages": int(((thread_sizes >= 7) & (thread_sizes <= 10)).sum()),
        "11+ messages": int((thread_sizes >= 11).sum()),
    }
    results["length_buckets"] = buckets

    logger.info("")
    logger.info(f"  Thread Length Buckets:")
    for label, count in buckets.items():
        pct = count / total_convs * 100 if total_convs > 0 else 0
        bar = "\u2588" * min(int(pct), 50)
        logger.info(f"    {label:<16} {count:>6,}  ({pct:5.1f}%)  {bar}")

    return results
