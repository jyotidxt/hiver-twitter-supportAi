"""Dataset overview module for EDA.

Computes and exports comprehensive dataset-level statistics
as both JSON and Markdown formats.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import PipelineConfig
from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)


def compute_overview(df: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    """Compute comprehensive dataset statistics.

    Args:
        df: The processed conversations DataFrame.
        config: Pipeline configuration.

    Returns:
        Dictionary containing all computed statistics.
    """
    log_section(logger, "DATASET OVERVIEW")

    stats: dict[str, Any] = {}

    # Basic counts
    total_messages = len(df)
    total_conversations = df["conversation_id"].nunique()
    customer_messages = len(df[df["role"] == "customer"])
    brand_replies = len(df[df["role"] == "brand"])
    unique_customers = df[df["role"] == "customer"]["author_id"].nunique()

    stats["basic_counts"] = {
        "total_conversations": int(total_conversations),
        "total_messages": int(total_messages),
        "customer_messages": int(customer_messages),
        "brand_replies": int(brand_replies),
        "unique_customers": int(unique_customers),
        "brand": config.selected_brand,
    }

    log_metric(logger, "Total conversations", f"{total_conversations:,}")
    log_metric(logger, "Total messages", f"{total_messages:,}")
    log_metric(logger, "Customer messages", f"{customer_messages:,}")
    log_metric(logger, "Brand replies", f"{brand_replies:,}")
    log_metric(logger, "Unique customers", f"{unique_customers:,}")

    # Thread length statistics
    thread_sizes = df.groupby("conversation_id").size()
    stats["thread_length"] = {
        "mean": round(float(thread_sizes.mean()), 2),
        "median": round(float(thread_sizes.median()), 2),
        "std": round(float(thread_sizes.std()), 2),
        "min": int(thread_sizes.min()),
        "max": int(thread_sizes.max()),
        "q25": round(float(thread_sizes.quantile(0.25)), 2),
        "q75": round(float(thread_sizes.quantile(0.75)), 2),
    }

    log_metric(logger, "Average thread length", f"{thread_sizes.mean():.2f}")
    log_metric(logger, "Median thread length", f"{thread_sizes.median():.0f}")
    log_metric(logger, "Longest conversation", f"{thread_sizes.max()} messages")
    log_metric(logger, "Shortest conversation", f"{thread_sizes.min()} messages")

    # Find the actual longest conversation ID
    longest_conv_id = thread_sizes.idxmax()
    shortest_conv_id = thread_sizes.idxmin()
    stats["notable_conversations"] = {
        "longest_id": str(longest_conv_id),
        "longest_length": int(thread_sizes.max()),
        "shortest_id": str(shortest_conv_id),
        "shortest_length": int(thread_sizes.min()),
    }

    # Role balance
    role_by_conv = df.groupby(["conversation_id", "role"]).size().unstack(fill_value=0)
    if "customer" in role_by_conv.columns and "brand" in role_by_conv.columns:
        stats["role_balance"] = {
            "avg_customer_msgs_per_thread": round(float(role_by_conv["customer"].mean()), 2),
            "avg_brand_replies_per_thread": round(float(role_by_conv["brand"].mean()), 2),
            "customer_to_brand_ratio": round(
                float(customer_messages / brand_replies) if brand_replies > 0 else 0, 2
            ),
        }
        log_metric(
            logger,
            "Customer:Brand ratio",
            f"{customer_messages / brand_replies:.2f}:1" if brand_replies > 0 else "N/A",
        )

    # Text length statistics
    text_lengths = df["text"].fillna("").str.len()
    stats["text_length"] = {
        "mean": round(float(text_lengths.mean()), 2),
        "median": round(float(text_lengths.median()), 2),
        "max": int(text_lengths.max()),
        "min": int(text_lengths.min()),
        "by_role": {},
    }
    for role in ["customer", "brand"]:
        role_lengths = df[df["role"] == role]["text"].fillna("").str.len()
        stats["text_length"]["by_role"][role] = {
            "mean": round(float(role_lengths.mean()), 2),
            "median": round(float(role_lengths.median()), 2),
        }

    # Temporal stats
    if "created_at" in df.columns and df["created_at"].notna().any():
        df_dated = df.dropna(subset=["created_at"])
        if len(df_dated) > 0:
            try:
                dates = pd.to_datetime(df_dated["created_at"])
                stats["temporal"] = {
                    "earliest": str(dates.min()),
                    "latest": str(dates.max()),
                    "span_days": int((dates.max() - dates.min()).days),
                }
            except Exception:
                stats["temporal"] = {"note": "Could not parse dates"}

    stats["_metadata"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "brand": config.selected_brand,
        "source": str(config.processed_conversations_path),
    }

    return stats


def export_overview_json(
    stats: dict[str, Any], output_dir: Path
) -> Path:
    """Export overview statistics as JSON.

    Args:
        stats: The computed statistics dictionary.
        output_dir: Directory to save the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_overview.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"  Exported overview JSON: {output_path}")
    return output_path


def export_overview_markdown(
    stats: dict[str, Any], output_dir: Path, brand: str
) -> Path:
    """Export overview statistics as a Markdown table.

    Args:
        stats: The computed statistics dictionary.
        output_dir: Directory to save the Markdown file.
        brand: The brand name for the report header.

    Returns:
        Path to the saved Markdown file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_overview.md"

    bc = stats.get("basic_counts", {})
    tl = stats.get("thread_length", {})
    rb = stats.get("role_balance", {})
    txtl = stats.get("text_length", {})

    lines = [
        f"# Dataset Overview \u2014 {brand}",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|------|",
        f"| Total Conversations | {bc.get('total_conversations', 0):,} |",
        f"| Total Messages | {bc.get('total_messages', 0):,} |",
        f"| Customer Messages | {bc.get('customer_messages', 0):,} |",
        f"| Brand Replies | {bc.get('brand_replies', 0):,} |",
        f"| Unique Customers | {bc.get('unique_customers', 0):,} |",
        "",
        "## Thread Length",
        "",
        "| Statistic | Value |",
        "|-----------|------|",
        f"| Mean | {tl.get('mean', 0)} |",
        f"| Median | {tl.get('median', 0)} |",
        f"| Std Dev | {tl.get('std', 0)} |",
        f"| Min | {tl.get('min', 0)} |",
        f"| Max | {tl.get('max', 0)} |",
        f"| 25th Percentile | {tl.get('q25', 0)} |",
        f"| 75th Percentile | {tl.get('q75', 0)} |",
        "",
        "## Role Balance",
        "",
        "| Metric | Value |",
        "|--------|------|",
        f"| Avg Customer Messages/Thread | {rb.get('avg_customer_msgs_per_thread', 0)} |",
        f"| Avg Brand Replies/Thread | {rb.get('avg_brand_replies_per_thread', 0)} |",
        f"| Customer:Brand Ratio | {rb.get('customer_to_brand_ratio', 0)}:1 |",
        "",
        "## Text Length",
        "",
        "| Metric | Value |",
        "|--------|------|",
        f"| Mean Characters | {txtl.get('mean', 0)} |",
        f"| Median Characters | {txtl.get('median', 0)} |",
        f"| Max Characters | {txtl.get('max', 0)} |",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"  Exported overview Markdown: {output_path}")
    return output_path
