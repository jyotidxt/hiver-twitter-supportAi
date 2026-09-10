"""Dataset export module for the preprocessing pipeline.

Exports processed datasets in multiple formats for downstream use.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig

from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)


def export_datasets(
    df: pd.DataFrame,
    config: PipelineConfig,
    removal_counts: dict[str, int],
    raw_count: int,
) -> None:
    """Export processed datasets and summary statistics.

    Generates:
    1. processed_conversations.csv - Full processed dataset
    2. annotation_ready.csv - Simplified format for annotation
    3. preprocessing_summary.json - Pipeline run statistics

    Args:
        df: The final processed DataFrame.
        config: Pipeline configuration with export paths.
        removal_counts: Dictionary of removal counts from quality filters.
        raw_count: Total count of rows in the raw dataset.
    """
    log_section(logger, "EXPORTING DATASETS")

    # Ensure output directories exist
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.preprocessing_summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Export 1: Full processed conversations
    _export_processed_conversations(df, config)

    # Export 2: Annotation-ready dataset
    _export_annotation_ready(df, config)

    # Export 3: Preprocessing summary
    _export_summary(df, config, removal_counts, raw_count)

    log_section(logger, "EXPORT COMPLETE")


def _export_processed_conversations(
    df: pd.DataFrame, config: PipelineConfig
) -> None:
    """Export the full processed conversations dataset.

    Args:
        df: The processed DataFrame.
        config: Pipeline configuration.
    """
    output_path = config.processed_conversations_path

    # Select and order columns for export
    export_columns = [
        "conversation_id",
        "tweet_id",
        "author_id",
        "role",
        "inbound",
        "created_at",
        "text",
        "text_original",
        "in_response_to_tweet_id",
        "response_tweet_id",
    ]

    # Only include columns that exist
    available_columns = [c for c in export_columns if c in df.columns]
    export_df = df[available_columns]

    export_df.to_csv(output_path, index=False, encoding="utf-8")

    log_metric(logger, "Processed conversations", str(output_path))
    log_metric(logger, "  Rows", f"{len(export_df):,}")
    log_metric(logger, "  Columns", f"{len(available_columns)}")
    file_size = output_path.stat().st_size / 1e6
    log_metric(logger, "  File size", f"{file_size:.1f} MB")


def _export_annotation_ready(
    df: pd.DataFrame, config: PipelineConfig
) -> None:
    """Export a simplified annotation-ready dataset.

    This format groups messages by conversation and presents them in a
    format suitable for human annotation.

    Args:
        df: The processed DataFrame.
        config: Pipeline configuration.
    """
    output_path = config.annotation_ready_path

    if "conversation_id" not in df.columns:
        logger.warning("No conversation_id column found. Skipping annotation export.")
        return

    # Build annotation-ready format: one row per conversation
    annotation_rows: list[dict[str, Any]] = []

    for conv_id, group in df.groupby("conversation_id"):
        group = group.sort_values("created_at") if "created_at" in group.columns else group

        # Get opening customer message
        customer_msgs = group[group["role"] == "customer"]
        brand_msgs = group[group["role"] == "brand"]

        opening_message = (
            customer_msgs.iloc[0]["text"] if len(customer_msgs) > 0 else ""
        )

        # Build full conversation text
        conversation_text = []
        for _, msg in group.iterrows():
            role_label = "[CUSTOMER]" if msg["role"] == "customer" else "[BRAND]"
            conversation_text.append(f"{role_label} {msg['text']}")

        annotation_rows.append({
            "conversation_id": conv_id,
            "opening_message": opening_message,
            "full_conversation": "\n".join(conversation_text),
            "num_messages": len(group),
            "num_customer_messages": len(customer_msgs),
            "num_brand_replies": len(brand_msgs),
            "customer_id": (
                customer_msgs.iloc[0]["author_id"]
                if len(customer_msgs) > 0
                else ""
            ),
        })

    annotation_df = pd.DataFrame(annotation_rows)
    annotation_df.to_csv(output_path, index=False, encoding="utf-8")

    log_metric(logger, "Annotation-ready dataset", str(output_path))
    log_metric(logger, "  Conversations", f"{len(annotation_df):,}")
    file_size = output_path.stat().st_size / 1e6
    log_metric(logger, "  File size", f"{file_size:.1f} MB")


def _export_summary(
    df: pd.DataFrame,
    config: PipelineConfig,
    removal_counts: dict[str, int],
    raw_count: int,
) -> None:
    """Export preprocessing summary statistics as JSON.

    Args:
        df: The final processed DataFrame.
        config: Pipeline configuration.
        removal_counts: Dictionary of removal counts.
        raw_count: Original raw dataset row count.
    """
    output_path = config.preprocessing_summary_path

    # Compute summary statistics
    n_conversations = (
        df["conversation_id"].nunique() if "conversation_id" in df.columns else 0
    )

    thread_sizes = (
        df.groupby("conversation_id").size()
        if "conversation_id" in df.columns
        else pd.Series(dtype=int)
    )

    summary: dict[str, Any] = {
        "pipeline_run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "brand": config.selected_brand,
            "config_file": str(config.raw.get("_source", "pipeline_config.yaml")),
        },
        "dataset_statistics": {
            "raw_total_rows": raw_count,
            "final_total_messages": len(df),
            "final_total_conversations": n_conversations,
            "retention_rate": (
                round(len(df) / raw_count * 100, 2) if raw_count > 0 else 0
            ),
        },
        "removal_counts": removal_counts,
        "thread_statistics": {
            "mean_length": round(float(thread_sizes.mean()), 2) if len(thread_sizes) > 0 else 0,
            "median_length": round(float(thread_sizes.median()), 2) if len(thread_sizes) > 0 else 0,
            "max_length": int(thread_sizes.max()) if len(thread_sizes) > 0 else 0,
            "min_length": int(thread_sizes.min()) if len(thread_sizes) > 0 else 0,
            "std_length": round(float(thread_sizes.std()), 2) if len(thread_sizes) > 0 else 0,
        },
        "role_distribution": (
            df["role"].value_counts().to_dict() if "role" in df.columns else {}
        ),
        "export_files": {
            "processed_conversations": str(config.processed_conversations_path),
            "annotation_ready": str(config.annotation_ready_path),
            "preprocessing_summary": str(config.preprocessing_summary_path),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    log_metric(logger, "Preprocessing summary", str(output_path))
