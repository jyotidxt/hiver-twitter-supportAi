"""Human-LLM Judge Agreement Analysis module.

Computes statistical agreement metrics (Cohen's Kappa, Percentage Agreement, MAE)
between human evaluator scores and LLM Judge scores across all 5 rubric dimensions.
Generates publication-quality agreement plots using Matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from src.utils.logger import get_logger

logger = get_logger(__name__)

JUDGE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "judge"


def validate_human_annotations_schema(df: pd.DataFrame) -> list[str]:
    """Validate human annotations DataFrame schema.

    Args:
        df: Input human annotation DataFrame.

    Returns:
        List of validation error strings. Empty list means valid.
    """
    required_cols = [
        "conversation_id",
        "human_groundedness",
        "human_correctness",
        "human_empathy",
        "human_actionability",
        "human_brand_tone",
        "human_overall",
    ]
    errors = []
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required human annotation column: '{col}'")
    return errors


def compute_human_llm_agreement(
    human_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute statistical agreement metrics and generate visualization charts.

    Args:
        human_df: DataFrame containing human scores.
        judge_df: DataFrame containing LLM judge scores.
        output_dir: Target output directory for charts.

    Returns:
        Dictionary of agreement metrics.
    """
    out_dir = output_dir or JUDGE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate schema
    schema_errs = validate_human_annotations_schema(human_df)
    if schema_errs:
        raise ValueError(f"Human annotations schema invalid: {schema_errs}")

    # Merge on conversation_id
    merged = pd.merge(human_df, judge_df, on="conversation_id", suffixes=("_human", "_judge"))
    if len(merged) == 0:
        logger.warning("No overlapping conversation IDs found between human and judge data.")
        return {"overall": {}, "dimensions": {}}

    dimensions = [
        ("groundedness", "human_groundedness", "groundedness"),
        ("correctness", "human_correctness", "correctness"),
        ("empathy", "human_empathy", "empathy"),
        ("actionability", "human_actionability", "actionability"),
        ("brand_tone", "human_brand_tone", "brand_tone"),
    ]

    dim_metrics: dict[str, Any] = {}
    human_all_scores = []
    judge_all_scores = []

    for dim_key, h_col, j_col in dimensions:
        h_scores = merged[h_col].astype(int).values
        j_scores = merged[j_col].astype(int).values

        human_all_scores.extend(h_scores)
        judge_all_scores.extend(j_scores)

        exact_match = float(np.mean(h_scores == j_scores))
        within_1 = float(np.mean(np.abs(h_scores - j_scores) <= 1))
        mae = float(mean_absolute_error(h_scores, j_scores))
        mean_diff = float(np.mean(j_scores - h_scores))

        # Cohen's Kappa
        try:
            kappa_unweighted = float(cohen_kappa_score(h_scores, j_scores))
            kappa_quadratic = float(cohen_kappa_score(h_scores, j_scores, weights="quadratic"))
        except Exception:
            kappa_unweighted = 0.0
            kappa_quadratic = 0.0

        dim_metrics[dim_key] = {
            "human_mean": round(float(np.mean(h_scores)), 2),
            "judge_mean": round(float(np.mean(j_scores)), 2),
            "cohen_kappa_unweighted": round(kappa_unweighted, 4),
            "cohen_kappa_quadratic": round(kappa_quadratic, 4),
            "exact_match_pct": round(exact_match * 100, 2),
            "within_1_pct": round(within_1 * 100, 2),
            "mae": round(mae, 4),
            "mean_score_difference": round(mean_diff, 4),
        }

    # Overall metrics
    h_overall = merged["human_overall"].astype(float).values
    j_overall = merged["overall_score"].astype(float).values
    overall_mae = float(mean_absolute_error(h_overall, j_overall))
    overall_mean_diff = float(np.mean(j_overall - h_overall))

    overall_metrics = {
        "sample_count": len(merged),
        "human_overall_mean": round(float(np.mean(h_overall)), 2),
        "judge_overall_mean": round(float(np.mean(j_overall)), 2),
        "overall_mae": round(overall_mae, 4),
        "overall_mean_difference": round(overall_mean_diff, 4),
        "overall_exact_match_pct": round(float(np.mean(np.round(h_overall) == np.round(j_overall))) * 100, 2),
        "overall_within_1_pct": round(float(np.mean(np.abs(h_overall - j_overall) <= 1.0)) * 100, 2),
    }

    # Generate Visualizations
    _plot_dimension_average_scores(dim_metrics, out_dir / "dimension_average_scores.png")
    _plot_score_distribution(merged, out_dir / "score_distribution.png")
    _plot_agreement_heatmap(merged, out_dir / "agreement_heatmap.png")
    _plot_human_vs_llm_scatter(merged, out_dir / "human_vs_llm_scatter.png")

    results_all = {
        "overall": overall_metrics,
        "dimensions": dim_metrics,
    }

    return results_all


def _plot_dimension_average_scores(dim_metrics: dict[str, Any], output_path: Path) -> None:
    """Bar chart comparing Human vs LLM Judge mean scores per dimension."""
    labels = list(dim_metrics.keys())
    h_means = [dim_metrics[k]["human_mean"] for k in labels]
    j_means = [dim_metrics[k]["judge_mean"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.bar(x - width/2, h_means, width, label="Human Mean", color="#1976D2")
    ax.bar(x + width/2, j_means, width, label="LLM Judge Mean", color="#4CAF50")

    ax.set_ylabel("Average Score (1-5 Scale)")
    ax.set_title("Human vs. LLM Judge Mean Scores by Dimension")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", " ").title() for l in labels])
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)


def _plot_score_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Distribution of overall scores for Human vs LLM Judge."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.hist(df["human_overall"], bins=np.linspace(1, 5, 9), alpha=0.6, label="Human Overall", color="#1976D2", edgecolor="black")
    ax.hist(df["overall_score"], bins=np.linspace(1, 5, 9), alpha=0.6, label="LLM Judge Overall", color="#4CAF50", edgecolor="black")

    ax.set_xlabel("Overall Quality Score (1-5)")
    ax.set_ylabel("Frequency")
    ax.set_title("Overall Quality Score Distribution (Human vs. LLM Judge)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)


def _plot_agreement_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap showing agreement matrix across 1-5 score levels."""
    h_scores = np.round(df["human_overall"]).astype(int)
    j_scores = np.round(df["overall_score"]).astype(int)

    matrix = np.zeros((5, 5), dtype=int)
    for h, j in zip(h_scores, j_scores):
        if 1 <= h <= 5 and 1 <= j <= 5:
            matrix[h-1, j-1] += 1

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_yticklabels([1, 2, 3, 4, 5])

    ax.set_xlabel("LLM Judge Score")
    ax.set_ylabel("Human Score")
    ax.set_title("Human vs. LLM Judge Agreement Heatmap Matrix")

    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > matrix.max()/2 else "black")

    fig.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)


def _plot_human_vs_llm_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot comparing Human vs LLM Judge overall scores."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Add minor jitter for visualization clarity
    jitter_h = df["human_overall"] + np.random.uniform(-0.05, 0.05, len(df))
    jitter_j = df["overall_score"] + np.random.uniform(-0.05, 0.05, len(df))

    ax.scatter(jitter_h, jitter_j, alpha=0.7, color="#1976D2", edgecolors="none", s=50)

    # Perfect agreement line y = x
    ax.plot([1, 5], [1, 5], "r--", label="Perfect Agreement (y = x)")

    ax.set_xlabel("Human Overall Score")
    ax.set_ylabel("LLM Judge Overall Score")
    ax.set_title("Scatter Plot: Human vs. LLM Judge Overall Scores")
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)
