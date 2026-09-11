"""Tests for the unified AI Support Agent pipeline and inference engine.

Covers inference execution, JSON schema validation, latency measurement,
CLI runner, and configuration loading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.run_agent import format_pretty_terminal_output
from src.pipeline.inference import InferenceEngine, InferenceResult
from src.pipeline.support_agent import analyze_customer_message
from src.utils.config import load_config


class TestInferencePipeline:
    """Test suite for InferenceEngine and unified pipeline."""

    def test_config_loading_agent_fields(self) -> None:
        """Test that load_config populates agent configuration fields."""
        config = load_config()
        assert hasattr(config, "agent_classifier_model_type")
        assert hasattr(config, "agent_embedding_model")
        assert config.agent_top_k >= 1
        assert config.agent_confidence_threshold > 0.0

    def test_inference_engine_run(self) -> None:
        """Test executing inference engine on sample text."""
        engine = InferenceEngine()
        result = engine.run("Where is my package tracking?")

        assert isinstance(result, InferenceResult)
        assert result.customer_message == "Where is my package tracking?"
        assert isinstance(result.predicted_intent, str)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.generated_reply, str)
        assert result.escalation_decision in ("AUTO_HANDLE", "ESCALATE")
        assert isinstance(result.escalation_reason, str)
        assert result.processing_time_ms > 0.0

    def test_inference_result_json_schema(self) -> None:
        """Test InferenceResult JSON schema fields."""
        engine = InferenceEngine()
        result = engine.run("I want a refund for item #4410")
        json_str = result.to_json()
        data = json.loads(json_str)

        required_keys = [
            "customer_message",
            "predicted_intent",
            "confidence",
            "top_3_candidates",
            "generated_reply",
            "escalation_decision",
            "escalation_reason",
            "processing_time_ms",
            "retrieved_examples",
        ]
        for key in required_keys:
            assert key in data, f"Missing key in JSON schema: {key}"

    def test_analyze_customer_message_entry_point(self) -> None:
        """Test repository single entry point analyze_customer_message()."""
        res = analyze_customer_message("Can I change my delivery address?")
        assert isinstance(res, dict)
        assert "predicted_intent" in res
        assert "generated_reply" in res
        assert "escalation_decision" in res

    def test_format_pretty_terminal_output(self) -> None:
        """Test terminal formatting utility for CLI."""
        engine = InferenceEngine()
        result = engine.run("Test message")
        output = format_pretty_terminal_output(result)
        assert "INFERENCE RESULT" in output
        assert "Generated Support Reply" in output

    def test_audit_log_created(self) -> None:
        """Test that agent audit log file is created in logs/."""
        engine = InferenceEngine()
        _ = engine.run("Checking log audit")
        log_file = PROJECT_ROOT / "logs" / "agent.log"
        assert log_file.exists()
