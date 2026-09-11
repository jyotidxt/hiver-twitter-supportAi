"""Intent classification module for customer support messages.

Trains a TF-IDF + Logistic Regression baseline model on labeled support intent data
and exposes a clean prediction interface returning predicted intent, confidence score,
and top 3 candidate intents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.annotation.schema import Intent
from src.utils.config import PipelineConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Seed dataset for bootstrap training if golden dataset is empty/missing
BOOTSTRAP_TRAINING_DATA = [
    ("Where is my package? Order #12345 hasn't arrived yet.", "order_status"),
    ("Can you give me tracking details for my order?", "order_status"),
    ("My delivery is expected today, what time will it arrive?", "order_status"),
    ("Check status of my order placed yesterday", "order_status"),
    ("Is my order still processing or shipped?", "order_status"),

    ("I want to cancel my order #9988", "order_modification"),
    ("Can I change the delivery address on my order?", "order_modification"),
    ("Please update the shipping address for order #4412", "order_modification"),
    ("I need to remove an item from my order", "order_modification"),
    ("Cancel order immediately please", "order_modification"),

    ("My package arrived but item is missing", "order_missing_wrong"),
    ("You sent me the wrong size shirt", "order_missing_wrong"),
    ("The box was delivered but only half of my order was inside", "order_missing_wrong"),
    ("Received blue shoes instead of red", "order_missing_wrong"),
    ("Item missing from multi-item shipment", "order_missing_wrong"),

    ("I want a full refund for this purchase", "refund_request"),
    ("How do I return a product and get my money back?", "refund_request"),
    ("Where is my refund? It's been 5 business days", "refund_request"),
    ("Please issue a refund to my credit card", "refund_request"),
    ("Requesting return label for order refund", "refund_request"),

    ("I was charged twice for order #5551", "billing_issue"),
    ("Why is my credit card bill higher than the checkout price?", "billing_issue"),
    ("Unauthorized charge on my account", "billing_issue"),
    ("Double payment billed for single order", "billing_issue"),
    ("Payment declined but money was deducted from bank", "billing_issue"),

    ("Driver left package in wrong apartment building", "delivery_problem"),
    ("Package marked delivered but not on my porch", "delivery_problem"),
    ("Package was severely damaged in transit", "delivery_problem"),
    ("Delivery failed and carrier left no notice", "delivery_problem"),
    ("Box crushed during delivery", "delivery_problem"),

    ("Do you offer express 1-day shipping?", "shipping_inquiry"),
    ("What are the international shipping rates to Canada?", "shipping_inquiry"),
    ("Is free shipping included with Prime?", "shipping_inquiry"),
    ("How much does standard shipping cost?", "shipping_inquiry"),
    ("Shipping timeframe for standard delivery", "shipping_inquiry"),

    ("I cannot log into my account, password reset link expired", "account_access"),
    ("My account has been locked due to security alert", "account_access"),
    ("Forgot password and 2FA code not working", "account_access"),
    ("Account hacked and unauthorized login detected", "account_access"),
    ("Unable to access my Amazon account", "account_access"),

    ("How do I cancel my Prime membership subscription?", "account_management"),
    ("I need to update my email address on my profile", "account_management"),
    ("Change primary phone number in account settings", "account_management"),
    ("How do I share Prime benefits with family?", "account_management"),
    ("Update notification preferences", "account_management"),

    ("The blender stopped working after two days", "product_issue"),
    ("Item arrived broken with cracked screen", "product_issue"),
    ("Defective electronics won't turn on", "product_issue"),
    ("Poor quality material, fell apart immediately", "product_issue"),
    ("Product is defective and not as described", "product_issue"),

    ("Amazon shopping app crashes every time I open cart", "technical_support"),
    ("Kindle device screen is frozen", "technical_support"),
    ("Website error when trying to checkout", "technical_support"),
    ("Alexa smart speaker not responding to voice", "technical_support"),
    ("Video stream keeps buffering on app", "technical_support"),

    ("What is your 30 day return policy for electronics?", "general_inquiry"),
    ("Thank you so much for the quick help!", "general_inquiry"),
    ("Great customer service support team", "general_inquiry"),
    ("How does gift card balance transfer work?", "general_inquiry"),
    ("General feedback regarding customer support", "general_inquiry"),
]


@dataclass
class IntentPrediction:
    """Dataclass holding intent prediction results.

    Attributes:
        predicted_intent: The top predicted intent string label.
        confidence: Probability score for the top predicted intent (0.0 to 1.0).
        top_3_candidates: List of (intent_label, probability) tuples for top 3 predictions.
    """

    predicted_intent: str
    confidence: float
    top_3_candidates: list[tuple[str, float]]


class IntentClassifier:
    """Intent classifier using TF-IDF feature extraction and Logistic Regression.

    Supports training on labeled dataset (or bootstrap training data)
    and provides probability-based intent prediction.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the intent classifier.

        Args:
            config: Optional PipelineConfig. Loaded automatically if None.
        """
        self.config = config or load_config()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )
        self.is_trained = False
        self.classes_: list[str] = []

    def train(self, df: pd.DataFrame | None = None) -> None:
        """Train the classifier on labeled intent data.

        If df is None, loads from golden dataset path or bootstrap data.

        Args:
            df: Optional DataFrame with 'text'/'customer_text' and 'primary_intent'/'intent'.
        """
        if df is None:
            df = self._load_training_data()

        # Handle column naming variations
        text_col = "text" if "text" in df.columns else "primary_text" if "primary_text" in df.columns else "customer_text"
        intent_col = "primary_intent" if "primary_intent" in df.columns else "intent"

        if text_col not in df.columns or intent_col not in df.columns:
            raise ValueError(f"Training data must contain text and intent columns. Found: {list(df.columns)}")

        texts = df[text_col].astype(str).tolist()
        labels = df[intent_col].astype(str).tolist()

        logger.info(f"Training IntentClassifier on {len(texts)} samples across {len(set(labels))} classes...")

        x_vec = self.vectorizer.fit_transform(texts)
        self.model.fit(x_vec, labels)
        self.classes_ = list(self.model.classes_)
        self.is_trained = True
        logger.info("IntentClassifier training completed successfully.")

    def _load_training_data(self) -> pd.DataFrame:
        """Load training dataset from golden CSV or bootstrap dataset."""
        golden_path = self.config.agent_golden_dataset_path
        if golden_path.exists():
            try:
                gdf = pd.read_csv(golden_path)
                if "primary_intent" in gdf.columns and len(gdf) >= 10:
                    # Merge with annotation_ready to get customer text if needed
                    ann_path = self.config.annotation_ready_path
                    if ann_path.exists():
                        adf = pd.read_csv(ann_path)
                        merged = pd.merge(gdf, adf, on="conversation_id", how="inner")
                        if "opening_message" in merged.columns:
                            merged["text"] = merged["opening_message"]
                            return merged
                    if "customer_text" in gdf.columns:
                        gdf["text"] = gdf["customer_text"]
                        return gdf
            except Exception as e:
                logger.warning(f"Could not load golden dataset from {golden_path}: {e}")

        # Fallback to bootstrap training data
        logger.info("Using bootstrap intent training dataset for baseline classifier...")
        return pd.DataFrame(BOOTSTRAP_TRAINING_DATA, columns=["text", "primary_intent"])

    def predict(self, text: str) -> IntentPrediction:
        """Predict the primary intent for an input customer message.

        Args:
            text: Customer input message text.

        Returns:
            IntentPrediction object with predicted_intent, confidence, and top_3_candidates.
        """
        if not self.is_trained:
            self.train()

        clean_text = text.strip()
        if not clean_text:
            return IntentPrediction(
                predicted_intent=Intent.GENERAL_INQUIRY.value,
                confidence=1.0,
                top_3_candidates=[(Intent.GENERAL_INQUIRY.value, 1.0)],
            )

        x_vec = self.vectorizer.transform([clean_text])
        probabilities = self.model.predict_proba(x_vec)[0]

        # Rank classes by probability
        ranked_indices = np.argsort(probabilities)[::-1]
        top_intent = self.classes_[ranked_indices[0]]
        top_conf = float(probabilities[ranked_indices[0]])

        top_3: list[tuple[str, float]] = [
            (self.classes_[idx], float(probabilities[idx]))
            for idx in ranked_indices[: min(3, len(ranked_indices))]
        ]

        return IntentPrediction(
            predicted_intent=top_intent,
            confidence=top_conf,
            top_3_candidates=top_3,
        )
