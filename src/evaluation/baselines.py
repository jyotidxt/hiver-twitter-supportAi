"""Baseline models module for evaluation benchmarking.

Provides two comparison baselines required by the Hiver evaluation harness:
1. Baseline1_Trivial: Zero-intelligence majority class predictor (lower bound).
2. Baseline2_Simple: Non-LLM TF-IDF nearest-neighbor retrieval benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BaselinePrediction:
    """Dataclass holding baseline model outputs.

    Attributes:
        predicted_intent: Intent string.
        confidence: Confidence score.
        escalation_decision: "AUTO_HANDLE" or "ESCALATE".
        generated_reply: Response string.
    """

    predicted_intent: str
    confidence: float
    escalation_decision: str
    generated_reply: str


class Baseline1_Trivial:
    """Trivial Baseline 1: Majority Class & Static Template (Lower Bound).

    Rationale:
    Establishes an absolute lower-bound anchor. A system that cannot beat
    always predicting the most frequent intent and defaulting to 'AUTO_HANDLE'
    provides zero marginal value.
    """

    def __init__(self, majority_intent: str = "order_status") -> None:
        """Initialize trivial baseline.

        Args:
            majority_intent: Target majority intent string label.
        """
        self.majority_intent = majority_intent
        self.name = "Baseline 1 (Majority Class Trivial)"

    def predict(self, text: str) -> BaselinePrediction:
        """Predict static majority values.

        Args:
            text: Input customer text (ignored).

        Returns:
            BaselinePrediction with majority intent and static reply.
        """
        return BaselinePrediction(
            predicted_intent=self.majority_intent,
            confidence=0.35,  # Fixed low static confidence
            escalation_decision="AUTO_HANDLE",  # Always default to no escalation
            generated_reply=(
                "Thank you for reaching out to @AmazonHelp. "
                "Please send us a Direct Message with your order details so we can assist you."
            ),
        )


class Baseline2_Simple:
    """Simple Baseline 2: TF-IDF Nearest-Neighbor Retrieval (Non-LLM Benchmark).

    Rationale:
    Establishes a strong non-LLM machine learning baseline. Finds the most similar
    historical customer query using TF-IDF cosine similarity, reuses its intent
    and historical resolution reply, and applies rule-based keyword escalation.
    """

    def __init__(self) -> None:
        """Initialize simple baseline."""
        self.name = "Baseline 2 (TF-IDF Nearest Neighbor)"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix: Any = None
        self.train_df: pd.DataFrame | None = None
        self.is_fitted = False

        self.escalation_keywords = [
            "refund", "cancel", "damaged", "broken", "hacked",
            "stolen", "charged twice", "lawsuit", "attorney", "unauthorized",
        ]

    def fit(self, df: pd.DataFrame) -> None:
        """Fit TF-IDF index over historical reference dataset.

        Args:
            df: DataFrame containing 'customer_text'/'text', 'intent'/'primary_intent', and optional 'brand_reply'.
        """
        self.train_df = df.copy().reset_index(drop=True)

        text_col = "customer_text" if "customer_text" in self.train_df.columns else "text" if "text" in self.train_df.columns else "opening_message"
        intent_col = "primary_intent" if "primary_intent" in self.train_df.columns else "intent"

        texts = self.train_df[text_col].fillna("").astype(str).tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        logger.info(f"Baseline2_Simple fitted on {len(texts)} samples.")

    def predict(self, text: str) -> BaselinePrediction:
        """Predict intent, escalation, and reply using nearest-neighbor similarity.

        Args:
            text: Input customer query text.

        Returns:
            BaselinePrediction object.
        """
        if not self.is_fitted or self.train_df is None or len(self.train_df) == 0:
            # Fallback if un-fitted
            return Baseline1_Trivial().predict(text)

        clean_text = text.strip().lower()
        query_vec = self.vectorizer.transform([clean_text])
        sims = cosine_similarity(self.tfidf_matrix, query_vec).squeeze()

        if np.isscalar(sims):
            sims = np.array([sims])

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        matched_row = self.train_df.iloc[best_idx]

        intent_col = "primary_intent" if "primary_intent" in matched_row else "intent"
        reply_col = "brand_reply" if "brand_reply" in matched_row else "full_conversation"

        predicted_intent = str(matched_row.get(intent_col, "order_status"))
        brand_reply = str(matched_row.get(reply_col, "Please DM us your order ID for help."))

        # Rule-based escalation check
        needs_escalate = any(kw in clean_text for kw in self.escalation_keywords)
        escalation_decision = "ESCALATE" if needs_escalate else "AUTO_HANDLE"

        return BaselinePrediction(
            predicted_intent=predicted_intent,
            confidence=max(best_sim, 0.40),
            escalation_decision=escalation_decision,
            generated_reply=brand_reply[:280],
        )
