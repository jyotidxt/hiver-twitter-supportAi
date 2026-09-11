"""Tests for the SemanticRetriever module."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import RetrievedEvidence, SemanticRetriever


class TestSemanticRetriever:
    """Test suite for SemanticRetriever."""

    def test_retriever_initialization(self) -> None:
        """Test default retriever initialization."""
        retriever = SemanticRetriever()
        assert retriever.is_indexed is False

    def test_index_building_and_retrieval(self) -> None:
        """Test index building and query retrieval."""
        sample_df = pd.DataFrame([
            {
                "conversation_id": "c1",
                "customer_text": "Where is my order package?",
                "brand_reply": "Please DM us your order ID to check tracking.",
            },
            {
                "conversation_id": "c2",
                "customer_text": "I was charged twice on my credit card",
                "brand_reply": "Please DM us so we can issue a billing refund.",
            },
            {
                "conversation_id": "c3",
                "customer_text": "My app keeps crashing when opening cart",
                "brand_reply": "Please try reinstalling the app or DM us error details.",
            },
        ])

        retriever = SemanticRetriever()
        retriever.build_index(sample_df)
        assert retriever.is_indexed is True

        evidence = retriever.retrieve("Where is my package tracking?", top_k=2)
        assert isinstance(evidence, RetrievedEvidence)
        assert evidence.query == "Where is my package tracking?"
        assert len(evidence.items) <= 2
        assert evidence.items[0].conversation_id == "c1"
        assert evidence.items[0].similarity_score >= 0.0

    def test_empty_query_retrieval(self) -> None:
        """Test handling empty string query."""
        retriever = SemanticRetriever()
        evidence = retriever.retrieve("  ")
        assert len(evidence.items) == 0

    def test_default_historical_evidence_fallback(self) -> None:
        """Test retrieval using default historical dataset when no file exists."""
        retriever = SemanticRetriever()
        evidence = retriever.retrieve("I want a refund for damaged item")
        assert retriever.is_indexed is True
        assert len(evidence.items) > 0
        assert hasattr(evidence.items[0], "brand_reply")
