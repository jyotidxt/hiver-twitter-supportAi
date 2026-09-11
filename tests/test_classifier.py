"""Tests for the IntentClassifier module."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.intent_classifier import IntentClassifier, IntentPrediction


class TestIntentClassifier:
    """Test suite for IntentClassifier."""

    def test_classifier_initialization(self) -> None:
        """Test default classifier initialization."""
        clf = IntentClassifier()
        assert clf.is_trained is False
        assert clf.model is not None

    def test_classifier_training_and_prediction(self) -> None:
        """Test training on sample data and predicting intents."""
        sample_df = pd.DataFrame([
            {"text": "Where is my order #123?", "primary_intent": "order_status"},
            {"text": "Tracking number for my shipment", "primary_intent": "order_status"},
            {"text": "I want a refund for this item", "primary_intent": "refund_request"},
            {"text": "Give me my money back please", "primary_intent": "refund_request"},
            {"text": "I cannot log into my account", "primary_intent": "account_access"},
            {"text": "Password reset token expired", "primary_intent": "account_access"},
        ])

        clf = IntentClassifier()
        clf.train(sample_df)

        assert clf.is_trained is True

        pred = clf.predict("Where is my tracking number for order?")
        assert isinstance(pred, IntentPrediction)
        assert pred.predicted_intent == "order_status"
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.top_3_candidates) > 0

    def test_empty_input_prediction(self) -> None:
        """Test handling empty string input."""
        clf = IntentClassifier()
        pred = clf.predict("   ")
        assert pred.predicted_intent == "general_inquiry"
        assert pred.confidence == 1.0

    def test_bootstrap_training(self) -> None:
        """Test automatic bootstrap dataset training when no data supplied."""
        clf = IntentClassifier()
        pred = clf.predict("Please cancel my order right away")
        assert clf.is_trained is True
        assert isinstance(pred.predicted_intent, str)
        assert len(pred.top_3_candidates) <= 3
