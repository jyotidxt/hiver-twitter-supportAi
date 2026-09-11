"""Tests for the LLM-as-a-Judge and Human Agreement evaluation module."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.judge.agreement import compute_human_llm_agreement, validate_human_annotations_schema
from src.judge.llm_judge import JudgeEvaluation, LLMJudge
from src.judge.report import generate_judge_report
from src.judge.rubric import RUBRIC_DIMENSIONS, EvaluationRubric


class TestRubric:
    """Test suite for rubric schema and validation."""

    def test_rubric_dimensions_count(self) -> None:
        """Test that all 5 required rubric dimensions are defined."""
        assert len(RUBRIC_DIMENSIONS) == 5
        for dim in ["groundedness", "correctness", "empathy", "actionability", "brand_tone"]:
            assert dim in RUBRIC_DIMENSIONS

    def test_validate_scores_valid(self) -> None:
        """Test score validation with valid scores."""
        scores = {
            "groundedness": 5,
            "correctness": 4,
            "empathy": 5,
            "actionability": 4,
            "brand_tone": 5,
        }
        errors = EvaluationRubric.validate_scores(scores)
        assert errors == []

    def test_validate_scores_invalid_range(self) -> None:
        """Test score validation with out-of-range score."""
        scores = {
            "groundedness": 6,  # Invalid > 5
            "correctness": 4,
            "empathy": 5,
            "actionability": 4,
            "brand_tone": 5,
        }
        errors = EvaluationRubric.validate_scores(scores)
        assert len(errors) > 0


class TestLLMJudge:
    """Test suite for LLM Judge evaluation."""

    def test_evaluate_reply_heuristic(self) -> None:
        """Test judge fallback evaluation."""
        judge = LLMJudge()
        res = judge.evaluate_reply(
            conversation_id="conv_test_01",
            customer_message="Where is my order #12345?",
            predicted_intent="order_status",
            generated_reply="We'd be glad to look into your order status! Please DM us your order number.",
            reference_reply="Please DM us your order ID for help.",
        )

        assert isinstance(res, JudgeEvaluation)
        assert res.conversation_id == "conv_test_01"
        assert 1 <= res.groundedness <= 5
        assert 1 <= res.correctness <= 5
        assert 1 <= res.empathy <= 5
        assert 1 <= res.actionability <= 5
        assert 1 <= res.brand_tone <= 5
        assert 1.0 <= res.overall_score <= 5.0
        assert isinstance(res.reasoning, str)


class TestHumanAgreement:
    """Test suite for Human-LLM agreement analysis."""

    def test_validate_human_annotations_schema(self) -> None:
        """Test human annotations schema validation."""
        valid_df = pd.DataFrame([{
            "conversation_id": "c1",
            "human_groundedness": 5,
            "human_correctness": 4,
            "human_empathy": 5,
            "human_actionability": 4,
            "human_brand_tone": 5,
            "human_overall": 4.6,
        }])
        assert validate_human_annotations_schema(valid_df) == []

        invalid_df = pd.DataFrame([{"conversation_id": "c1"}])
        assert len(validate_human_annotations_schema(invalid_df)) > 0

    def test_compute_human_llm_agreement(self, tmp_path: Path) -> None:
        """Test agreement metrics calculation and plot generation."""
        human_df = pd.DataFrame([
            {"conversation_id": "c1", "human_groundedness": 5, "human_correctness": 4, "human_empathy": 5, "human_actionability": 4, "human_brand_tone": 5, "human_overall": 4.6},
            {"conversation_id": "c2", "human_groundedness": 4, "human_correctness": 3, "human_empathy": 4, "human_actionability": 3, "human_brand_tone": 4, "human_overall": 3.6},
        ])
        judge_df = pd.DataFrame([
            {"conversation_id": "c1", "groundedness": 5, "correctness": 4, "empathy": 5, "actionability": 4, "brand_tone": 5, "overall_score": 4.6},
            {"conversation_id": "c2", "groundedness": 4, "correctness": 4, "empathy": 4, "actionability": 3, "brand_tone": 4, "overall_score": 3.8},
        ])

        res = compute_human_llm_agreement(human_df, judge_df, tmp_path)

        assert "overall" in res
        assert "dimensions" in res
        assert res["overall"]["sample_count"] == 2
        assert res["dimensions"]["groundedness"]["exact_match_pct"] == 100.0
        assert (tmp_path / "dimension_average_scores.png").exists()
        assert (tmp_path / "agreement_heatmap.png").exists()


class TestJudgeReport:
    """Test suite for judge report generation."""

    def test_generate_judge_report(self, tmp_path: Path) -> None:
        """Test end-to-end judge report generation."""
        res = generate_judge_report(output_dir=tmp_path)

        assert "summary" in res
        assert "agreement" in res
        assert (tmp_path / "judged_predictions.csv").exists()
        assert (tmp_path / "agreement_metrics.json").exists()
        assert (tmp_path / "judge_summary.json").exists()
        assert (tmp_path / "JUDGE_REPORT.md").exists()
