"""Tests for the EscalationEngine module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.escalation.escalation import (
    ACTION_AUTO_HANDLE,
    ACTION_ESCALATE,
    EscalationDecision,
    EscalationEngine,
)


class TestEscalationEngine:
    """Test suite for EscalationEngine."""

    def test_engine_initialization(self) -> None:
        """Test default escalation engine initialization."""
        engine = EscalationEngine()
        assert engine.confidence_threshold > 0.0
        assert "account_access" in engine.auto_escalate_intents

    def test_high_confidence_auto_handle(self) -> None:
        """Test that high confidence standard queries are auto-handled."""
        engine = EscalationEngine()
        decision = engine.evaluate(
            customer_message="Where is my order status?",
            predicted_intent="order_status",
            classifier_confidence=0.92,
        )
        assert isinstance(decision, EscalationDecision)
        assert decision.decision == ACTION_AUTO_HANDLE
        assert decision.confidence > 0.0

    def test_low_confidence_escalation(self) -> None:
        """Test that low classifier confidence triggers escalation."""
        engine = EscalationEngine()
        decision = engine.evaluate(
            customer_message="Random ambiguous query",
            predicted_intent="order_status",
            classifier_confidence=0.40,
        )
        assert decision.decision == ACTION_ESCALATE
        assert "Low intent classification confidence" in decision.reason

    def test_sensitive_intent_escalation(self) -> None:
        """Test that sensitive intents like account_access trigger escalation."""
        engine = EscalationEngine()
        decision = engine.evaluate(
            customer_message="I cannot log into my account",
            predicted_intent="account_access",
            classifier_confidence=0.95,
        )
        assert decision.decision == ACTION_ESCALATE
        assert "account_access" in decision.reason

    def test_trigger_keyword_escalation(self) -> None:
        """Test that escalation keywords trigger human transfer."""
        engine = EscalationEngine()
        decision = engine.evaluate(
            customer_message="I will file a lawsuit and get an attorney for fraud",
            predicted_intent="general_inquiry",
            classifier_confidence=0.90,
        )
        assert decision.decision == ACTION_ESCALATE
        assert any(word in decision.reason for word in ["lawsuit", "attorney", "fraud", "keyword"])
