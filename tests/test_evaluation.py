"""Tests for the quantitative evaluation harness module."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baselines import Baseline1_Trivial, Baseline2_Simple
from src.evaluation.evaluator import EvaluationHarness
from src.evaluation.metrics import (
    compute_escalation_metrics,
    compute_intent_metrics,
    plot_confusion_matrix,
)


class TestEvaluationMetrics:
    """Test suite for evaluation metrics calculation."""

    def test_compute_intent_metrics(self) -> None:
        """Test calculation of intent metrics."""
        y_true = ["order_status", "order_status", "refund_request", "account_access"]
        y_pred = ["order_status", "refund_request", "refund_request", "account_access"]

        metrics = compute_intent_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.75
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert len(metrics["classes"]) == 3
        assert len(metrics["confusion_matrix"]) == 3

    def test_compute_escalation_metrics(self) -> None:
        """Test calculation of escalation decision metrics and error rates."""
        y_true = [False, False, True, True]
        y_pred = [False, True, True, False]

        metrics = compute_escalation_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.5
        assert metrics["true_positives"] == 1
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["false_positive_rate"] == 0.5
        assert metrics["false_negative_rate"] == 0.5
        assert "business_interpretation" in metrics

    def test_plot_confusion_matrix(self, tmp_path: Path) -> None:
        """Test confusion matrix visualization generation."""
        cm = [[5, 1], [2, 4]]
        labels = ["ClassA", "ClassB"]
        out_png = tmp_path / "test_cm.png"

        result_path = plot_confusion_matrix(cm, labels, out_png)
        assert result_path.exists()


class TestBaselines:
    """Test suite for baseline comparison models."""

    def test_baseline1_trivial(self) -> None:
        """Test Baseline 1 static majority predictions."""
        b1 = Baseline1_Trivial(majority_intent="order_status")
        pred = b1.predict("Random customer text")

        assert pred.predicted_intent == "order_status"
        assert pred.escalation_decision == "AUTO_HANDLE"
        assert isinstance(pred.generated_reply, str)

    def test_baseline2_simple(self) -> None:
        """Test Baseline 2 TF-IDF nearest-neighbor retrieval."""
        sample_df = pd.DataFrame([
            {"customer_text": "Where is my order?", "primary_intent": "order_status", "brand_reply": "Please DM us your order ID."},
            {"customer_text": "I want a refund", "primary_intent": "refund_request", "brand_reply": "Please DM us for refund details."},
        ])

        b2 = Baseline2_Simple()
        b2.fit(sample_df)
        assert b2.is_fitted is True

        pred = b2.predict("I need a refund for my order")
        assert isinstance(pred.predicted_intent, str)
        assert pred.escalation_decision in ("AUTO_HANDLE", "ESCALATE")


class TestEvaluationHarness:
    """Test suite for the full evaluation harness runner."""

    def test_run_evaluations(self, tmp_path: Path) -> None:
        """Test end-to-end quantitative evaluation harness execution."""
        harness = EvaluationHarness()
        harness.output_dir = tmp_path

        res = harness.run_evaluations()

        assert "predictions_df" in res
        assert "intent_metrics" in res
        assert "escalation_metrics" in res
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "intent_metrics.json").exists()
        assert (tmp_path / "escalation_metrics.json").exists()
        assert (tmp_path / "baseline_comparison.csv").exists()
