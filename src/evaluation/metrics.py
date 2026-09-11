"""Quantitative evaluation metrics module.

Computes classification metrics for intent prediction and escalation decisions,
including accuracy, precision, recall, F1 scores, FPR/FNR, and confusion matrix visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_intent_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict[str, Any]:
    """Compute classification metrics for intent prediction.

    Args:
        y_true: Ground truth intent labels.
        y_pred: Predicted intent labels.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and confusion matrix.
    """
    if not y_true or not y_pred:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_weighted": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "total_samples": 0,
        }

    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0)
    f1_weight = f1_score(y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_macro": round(float(rec_macro), 4),
        "recall_weighted": round(float(rec_weighted), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_weight), 4),
        "total_samples": len(y_true),
        "classes": unique_labels,
        "confusion_matrix": cm.tolist(),
    }


def compute_escalation_metrics(
    y_true: list[bool | str], y_pred: list[bool | str]
) -> dict[str, Any]:
    """Compute classification and error rate metrics for escalation decisions.

    Normalizes inputs to boolean (True = ESCALATE, False = AUTO_HANDLE).

    Args:
        y_true: Ground truth escalation labels.
        y_pred: Predicted escalation labels.

    Returns:
        Dictionary containing accuracy, precision, recall, F1, FPR, FNR, and business commentary.
    """
    if not y_true or not y_pred:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "total_samples": 0,
        }

    # Normalize to boolean
    def _to_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        s = str(val).strip().upper()
        return s in ("TRUE", "YES", "1", "ESCALATE", "HUMAN REQUIRED")

    b_true = [_to_bool(v) for v in y_true]
    b_pred = [_to_bool(v) for v in y_pred]

    cm = confusion_matrix(b_true, b_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(b_true, b_pred)
    prec = precision_score(b_true, b_pred, zero_division=0)
    rec = recall_score(b_true, b_pred, zero_division=0)
    f1 = f1_score(b_true, b_pred, zero_division=0)

    # FPR = FP / (FP + TN) (Auto-handle cases wrongly escalated to human)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # FNR = FN / (FN + TP) (Human-required cases wrongly auto-handled - DANGEROUS RISK)
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": len(b_true),
        "business_interpretation": (
            f"FPR ({fpr:.1%}) represents safe queries unnecessarily sent to agents (increased labor cost). "
            f"FNR ({fnr:.1%}) represents sensitive queries wrongly automated (risk of customer dissatisfaction or policy breach)."
        ),
    }


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str = "Intent Classification Confusion Matrix",
) -> Path:
    """Plot and save a publication-quality confusion matrix chart using Matplotlib.

    Args:
        cm: Confusion matrix array.
        labels: Class label names.
        output_path: Target PNG file path.
        title: Plot title text.

    Returns:
        The output file Path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(cm)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate matrix cells
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(
                j, i, format(val, "d"),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix saved: {output_path}")
    return output_path
