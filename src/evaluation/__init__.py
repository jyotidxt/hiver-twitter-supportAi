"""Evaluation harness package for Hiver Support AI."""

from src.evaluation.baselines import Baseline1_Trivial, Baseline2_Simple
from src.evaluation.evaluator import EvaluationHarness
from src.evaluation.metrics import (
    compute_escalation_metrics,
    compute_intent_metrics,
    plot_confusion_matrix,
)
from src.evaluation.report import generate_evaluation_summary

__all__ = [
    "compute_intent_metrics",
    "compute_escalation_metrics",
    "plot_confusion_matrix",
    "Baseline1_Trivial",
    "Baseline2_Simple",
    "EvaluationHarness",
    "generate_evaluation_summary",
]
