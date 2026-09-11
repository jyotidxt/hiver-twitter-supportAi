"""Unified AI Support Agent orchestrator module.

Integrates IntentClassifier, SemanticRetriever, ReplyGenerator, and EscalationEngine
into a single modular pipeline. Provides analyze_customer_message() as the primary
entry point for the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.classifier.intent_classifier import IntentClassifier
from src.escalation.escalation import EscalationEngine
from src.reply_generator.generator import ReplyGenerator
from src.retrieval.retriever import SemanticRetriever
from src.utils.config import PipelineConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SupportAgent:
    """Unified AI Support Agent combining intent classification, semantic retrieval,
    grounded reply generation, and escalation evaluation.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the Support Agent components using dependency injection.

        Args:
            config: Optional PipelineConfig. Loaded automatically if None.
        """
        self.config = config or load_config()
        self.classifier = IntentClassifier(self.config)
        self.retriever = SemanticRetriever(self.config)
        self.reply_generator = ReplyGenerator(self.config)
        self.escalation_engine = EscalationEngine(self.config)
        self._is_initialized = False

    def initialize(self) -> None:
        """Explicitly train classifier and build retrieval index if needed."""
        if not self._is_initialized:
            logger.info("Initializing Support Agent pipeline components...")
            self.classifier.train()
            self.retriever.build_index()
            self._is_initialized = True
            logger.info("Support Agent pipeline ready.")

    def process_message(self, customer_message: str) -> dict[str, Any]:
        """Process an incoming customer support message through the full pipeline.

        Args:
            customer_message: Raw incoming tweet or query from customer.

        Returns:
            Dictionary containing:
            - customer_message
            - predicted_intent
            - confidence
            - top_3_candidates
            - generated_reply
            - escalation_decision
            - escalation_reason
            - retrieved_examples
        """
        clean_text = customer_message.strip()

        # Step 1: Predict Primary Intent
        prediction = self.classifier.predict(clean_text)

        # Step 2: Retrieve Historical Evidence
        evidence = self.retriever.retrieve(clean_text, top_k=self.config.agent_top_k)

        # Step 3: Evaluate Escalation Decision
        escalation = self.escalation_engine.evaluate(
            customer_message=clean_text,
            predicted_intent=prediction.predicted_intent,
            classifier_confidence=prediction.confidence,
            evidence=evidence,
        )

        # Step 4: Generate Grounded Support Reply
        generated_reply = self.reply_generator.generate_reply(
            customer_message=clean_text,
            predicted_intent=prediction.predicted_intent,
            evidence=evidence,
        )

        # Format structured output
        output = {
            "customer_message": clean_text,
            "predicted_intent": prediction.predicted_intent,
            "confidence": round(prediction.confidence, 4),
            "top_3_candidates": [
                [intent, round(score, 4)]
                for intent, score in prediction.top_3_candidates
            ],
            "generated_reply": generated_reply,
            "escalation_decision": escalation.decision,
            "escalation_reason": escalation.reason,
            "retrieved_examples": evidence.to_dict()["items"],
        }

        return output


# Global instance for quick function access
_GLOBAL_AGENT: SupportAgent | None = None


def analyze_customer_message(
    customer_message: str,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Single repository entry point to analyze a customer support message.

    Runs the complete AI Support Agent pipeline (Intent -> Retrieval -> Reply -> Escalation).

    Args:
        customer_message: Customer message text.
        config_path: Optional path to custom YAML configuration file.

    Returns:
        Structured JSON-serializable dictionary with prediction, reply, and escalation decision.
    """
    global _GLOBAL_AGENT
    if _GLOBAL_AGENT is None or config_path is not None:
        cfg = load_config(config_path)
        _GLOBAL_AGENT = SupportAgent(cfg)

    return _GLOBAL_AGENT.process_message(customer_message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hiver Support AI Agent CLI")
    parser.add_argument(
        "message",
        nargs="?",
        default="Where is my order? Order #12345 has not arrived.",
        help="Customer message to analyze",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")

    args = parser.parse_args()

    result = analyze_customer_message(args.message, args.config)
    print(json.dumps(result, indent=2))
