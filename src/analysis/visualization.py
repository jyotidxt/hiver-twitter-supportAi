"""Visualization module for EDA.

Generates publication-quality charts using matplotlib.
All figures are saved to the EDA output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger, log_section

logger = get_logger(__name__)


def _setup_style(config: PipelineConfig) -> dict[str, str]:
    """Configure matplotlib style for consistent, clean charts.

    Args:
        config: Pipeline configuration with visualization settings.

    Returns:
        Dictionary of color values.
    """
    colors = config.viz_colors or {
        "primary": "#1976D2",
        "secondary": "#FF7043",
        "accent": "#4CAF50",
        "warning": "#FFC107",
        "danger": "#F44336",
        "neutral": "#78909C",
        "background": "#FAFAFA",
        "grid": "#E0E0E0",
    }

    plt.rcParams.update({
        "figure.facecolor": colors.get("background", "#FAFAFA"),
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": colors.get("grid", "#E0E0E0"),
        "axes.grid": True,
        "grid.color": colors.get("grid", "#E0E0E0"),
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": config.viz_dpi,
        "savefig.dpi": config.viz_dpi,
        "savefig.bbox": "tight",
        "savefig.facecolor": colors.get("background", "#FAFAFA"),
    })

    return colors


def generate_all_charts(
    overview_stats: dict[str, Any],
    conversation_stats: dict[str, Any],
    language_stats: dict[str, Any],
    intent_stats: dict[str, Any],
    escalation_stats: dict[str, Any],
    config: PipelineConfig,
) -> list[Path]:
    """Generate all EDA charts and save to disk.

    Args:
        overview_stats: From dataset_overview.compute_overview.
        conversation_stats: From conversation_analysis.analyze_conversations.
        language_stats: From intent_discovery.analyze_customer_language.
        intent_stats: From intent_discovery.discover_intents.
        escalation_stats: From intent_discovery.analyze_escalation_patterns.
        config: Pipeline configuration.

    Returns:
        List of paths to saved chart files.
    """
    log_section(logger, "GENERATING VISUALIZATIONS")

    colors = _setup_style(config)
    output_dir = config.eda_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = config.viz_format

    saved_files: list[Path] = []

    # Chart 1: Conversation Length Distribution
    path = _chart_conversation_length(
        conversation_stats, colors, output_dir, fmt
    )
    if path:
        saved_files.append(path)

    # Chart 2: Messages per Thread (box + violin)
    path = _chart_messages_per_thread(
        conversation_stats, colors, output_dir, fmt
    )
    if path:
        saved_files.append(path)

    # Chart 3: Customer vs Brand Message Count
    path = _chart_customer_vs_brand(
        overview_stats, colors, output_dir, fmt
    )
    if path:
        saved_files.append(path)

    # Chart 4: Top Customer Keywords
    path = _chart_top_keywords(
        language_stats, colors, output_dir, fmt
    )
    if path:
        saved_files.append(path)

    # Chart 5: Candidate Intent Frequency
    path = _chart_intent_frequency(
        intent_stats, colors, output_dir, fmt
    )
    if path:
        saved_files.append(path)

    # Chart 6: Multi-turn Conversation Distribution
    path = _chart_multiturn_distribution(
        conversation_stats, colors, output_dir, fmt
    )
    if path:
        saved_files.append(path)

    # Chart 7: Question vs Complaint ratio
    path = _chart_question_complaint(
        language_stats, colors, output_dir, fmt
    )
    if path:
        saved_files.append(path)

    # Chart 8: Escalation Signals
    path = _chart_escalation_signals(
        escalation_stats, colors, output_dir, fmt
    )
    if path:
        saved_files.append(path)

    logger.info(f"  Saved {len(saved_files)} charts to {output_dir}")
    return saved_files


# ══════════════════════════════════════════
#  Individual chart functions
# ══════════════════════════════════════════


def _chart_conversation_length(
    conv_stats: dict[str, Any],
    colors: dict[str, str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Chart 1: Conversation length distribution histogram."""
    dist = conv_stats.get("messages_per_conversation", {}).get("distribution", {})
    if not dist:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    lengths = sorted(dist.keys())[:20]  # Cap at 20 for readability
    counts = [dist[l] for l in lengths]

    bars = ax.bar(
        [str(l) for l in lengths],
        counts,
        color=colors.get("primary", "#1976D2"),
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )

    # Highlight the median
    percentiles = conv_stats.get("messages_per_conversation", {}).get("percentiles", {})
    median_val = percentiles.get("p50", 0)
    ax.axvline(
        x=str(int(median_val)) if int(median_val) in [int(l) for l in lengths] else -1,
        color=colors.get("danger", "#F44336"),
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.0f}",
    )

    ax.set_xlabel("Messages per Conversation")
    ax.set_ylabel("Number of Conversations")
    ax.set_title("Conversation Length Distribution")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    path = output_dir / f"01_conversation_length_distribution.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  \u2713 {path.name}")
    return path


def _chart_messages_per_thread(
    conv_stats: dict[str, Any],
    colors: dict[str, str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Chart 2: Messages per thread distribution with buckets."""
    buckets = conv_stats.get("length_buckets", {})
    if not buckets:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(buckets.keys())
    values = list(buckets.values())
    total = sum(values) or 1

    bar_colors = [
        colors.get("primary", "#1976D2"),
        colors.get("accent", "#4CAF50"),
        colors.get("warning", "#FFC107"),
        colors.get("secondary", "#FF7043"),
        colors.get("danger", "#F44336"),
    ]
    # Extend colors if needed
    while len(bar_colors) < len(labels):
        bar_colors.append(colors.get("neutral", "#78909C"))

    bars = ax.barh(
        labels[::-1],
        values[::-1],
        color=bar_colors[:len(labels)][::-1],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )

    # Add percentage labels
    for bar, val in zip(bars, values[::-1]):
        pct = val / total * 100
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({pct:.1f}%)",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Number of Conversations")
    ax.set_title("Messages per Thread Distribution")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    path = output_dir / f"02_messages_per_thread.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  \u2713 {path.name}")
    return path


def _chart_customer_vs_brand(
    overview_stats: dict[str, Any],
    colors: dict[str, str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Chart 3: Customer vs Brand message count."""
    bc = overview_stats.get("basic_counts", {})
    customer = bc.get("customer_messages", 0)
    brand = bc.get("brand_replies", 0)

    if customer == 0 and brand == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    categories = ["Customer\nMessages", "Brand\nReplies"]
    values = [customer, brand]
    bar_colors = [
        colors.get("secondary", "#FF7043"),
        colors.get("primary", "#1976D2"),
    ]

    bars = axes[0].bar(
        categories, values, color=bar_colors, edgecolor="white",
        linewidth=0.8, alpha=0.85, width=0.5
    )
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:,}",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Message Counts by Role")
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Pie chart
    axes[1].pie(
        values,
        labels=[f"Customer ({customer:,})", f"Brand ({brand:,})"],
        colors=bar_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Role Distribution")

    fig.suptitle("Customer vs Brand Messages", fontsize=16, fontweight="bold", y=1.02)
    path = output_dir / f"03_customer_vs_brand.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  \u2713 {path.name}")
    return path


def _chart_top_keywords(
    language_stats: dict[str, Any],
    colors: dict[str, str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Chart 4: Top customer keywords horizontal bar chart."""
    top_kws = language_stats.get("top_keywords", [])
    if not top_kws:
        return None

    # Show top 20
    display_kws = top_kws[:20]
    words = [kw["word"] for kw in display_kws][::-1]
    counts = [kw["count"] for kw in display_kws][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(
        words, counts,
        color=colors.get("primary", "#1976D2"),
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center", fontsize=9,
        )

    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Customer Keywords")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    path = output_dir / f"04_top_customer_keywords.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  \u2713 {path.name}")
    return path


def _chart_intent_frequency(
    intent_stats: dict[str, Any],
    colors: dict[str, str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Chart 5: Candidate intent frequency bar chart."""
    clusters = intent_stats.get("clusters", [])
    if not clusters:
        return None

    fig, ax = plt.subplots(figsize=(12, max(6, len(clusters) * 0.7)))

    labels = [c["theme_label"][:40] for c in clusters][::-1]
    sizes = [c["size"] for c in clusters][::-1]
    pcts = [c["frequency_pct"] for c in clusters][::-1]

    # Gradient colors from primary to accent
    n = len(labels)
    base_color = colors.get("primary", "#1976D2")
    bar_colors = [base_color] * n

    bars = ax.barh(
        labels, sizes,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )

    for bar, size, pct in zip(bars, sizes, pcts):
        ax.text(
            bar.get_width() + max(sizes) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{size:,} ({pct:.1f}%)",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Number of Messages")
    ax.set_title("Candidate Intent Themes (by Frequency)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    path = output_dir / f"05_candidate_intent_frequency.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  \u2713 {path.name}")
    return path


def _chart_multiturn_distribution(
    conv_stats: dict[str, Any],
    colors: dict[str, str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Chart 6: Multi-turn vs single-turn distribution."""
    turn = conv_stats.get("turn_structure", {})
    if not turn:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    values = [turn.get("single_turn_count", 0), turn.get("multi_turn_count", 0)]
    labels_pie = [
        f"Single-turn\n({values[0]:,})",
        f"Multi-turn\n({values[1]:,})",
    ]
    pie_colors = [
        colors.get("neutral", "#78909C"),
        colors.get("primary", "#1976D2"),
    ]

    axes[0].pie(
        values, labels=labels_pie, colors=pie_colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[0].set_title("Single-turn vs Multi-turn")

    # Response depth distribution
    depth_dist = conv_stats.get("response_depth", {}).get("depth_distribution", {})
    if depth_dist:
        depths = sorted(depth_dist.keys())[:12]
        depth_counts = [depth_dist[d] for d in depths]

        axes[1].bar(
            [str(d) for d in depths],
            depth_counts,
            color=colors.get("accent", "#4CAF50"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.85,
        )
        axes[1].set_xlabel("Response Depth (Exchange Rounds)")
        axes[1].set_ylabel("Conversations")
        axes[1].set_title("Response Depth Distribution")
        axes[1].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    fig.suptitle(
        "Multi-turn Conversation Analysis",
        fontsize=16, fontweight="bold", y=1.02,
    )
    path = output_dir / f"06_multiturn_distribution.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  \u2713 {path.name}")
    return path


def _chart_question_complaint(
    language_stats: dict[str, Any],
    colors: dict[str, str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Chart 7: Question vs complaint ratio."""
    qc = language_stats.get("question_vs_complaint", {})
    if not qc:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Questions (?)", "Complaint Words", "Exclamations (!)"]
    values = [
        qc.get("questions", 0),
        qc.get("complaints", 0),
        qc.get("exclamation_msgs", 0),
    ]
    pcts = [
        qc.get("questions_pct", 0),
        qc.get("complaints_pct", 0),
        qc.get("exclamation_pct", 0),
    ]
    bar_colors = [
        colors.get("primary", "#1976D2"),
        colors.get("danger", "#F44336"),
        colors.get("warning", "#FFC107"),
    ]

    bars = ax.bar(
        categories, values, color=bar_colors,
        edgecolor="white", linewidth=0.8, alpha=0.85, width=0.5,
    )

    for bar, val, pct in zip(bars, values, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:,}\n({pct}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Number of Messages")
    ax.set_title("Customer Message Types")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    path = output_dir / f"07_question_vs_complaint.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  \u2713 {path.name}")
    return path


def _chart_escalation_signals(
    escalation_stats: dict[str, Any],
    colors: dict[str, str],
    output_dir: Path,
    fmt: str,
) -> Path | None:
    """Chart 8: Escalation signals overview."""
    keyword_signals = escalation_stats.get("keyword_signals", {})
    if not keyword_signals:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Combine all signal types
    signal_labels = []
    signal_counts = []
    signal_pcts = []

    # Add keyword categories
    for cat, data in keyword_signals.items():
        signal_labels.append(cat.title())
        signal_counts.append(data.get("count", 0))
        signal_pcts.append(data.get("pct", 0))

    # Add structural signals
    for key, label in [
        ("repeated_followups", "Repeated\nFollowups"),
        ("unresolved", "Unresolved"),
    ]:
        if key in escalation_stats:
            signal_labels.append(label)
            signal_counts.append(escalation_stats[key].get("count", 0))
            signal_pcts.append(escalation_stats[key].get("pct", 0))

    if not signal_counts:
        plt.close(fig)
        return None

    bar_colors = [
        colors.get("danger", "#F44336"),
        colors.get("warning", "#FFC107"),
        colors.get("secondary", "#FF7043"),
        colors.get("primary", "#1976D2"),
        colors.get("accent", "#4CAF50"),
        colors.get("neutral", "#78909C"),
    ]
    while len(bar_colors) < len(signal_labels):
        bar_colors.append(colors.get("neutral", "#78909C"))

    bars = ax.bar(
        signal_labels, signal_counts,
        color=bar_colors[:len(signal_labels)],
        edgecolor="white", linewidth=0.8, alpha=0.85, width=0.6,
    )

    for bar, val, pct in zip(bars, signal_counts, signal_pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(signal_counts) * 0.02,
            f"{val:,}\n({pct}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Number of Conversations")
    ax.set_title("Escalation Signal Analysis")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    path = output_dir / f"08_escalation_signals.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  \u2713 {path.name}")
    return path
