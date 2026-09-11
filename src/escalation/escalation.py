"""Escalation decision engine module.

Evaluates customer messages, predicted intent, classification confidence,
and retrieved evidence against configurable rule-based policies to decide
whether a query can be AUTO_HANDLE or must ESCALATE to a human operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.retrieval.retriever import RetrievedEvidence

from src.utils.config import PipelineConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Action decision constants
ACTION_AUTO_HANDLE = "AUTO_HANDLE"
ACTION_ESCALATE = "ESCALATE"


@dataclass
class EscalationDecision:
    """Dataclass encapsulating escalation decision outcomes.

    Attributes:
        decision: "AUTO_HANDLE" or "ESCALATE".
        confidence: Decision confidence score (0.0 to 1.0).
        reason: Human-readable explanation for the decision.
    """

    decision: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


class EscalationEngine:
    """Configurable rule-based escalation engine for customer support queries.

    Checks intent thresholds, sensitive intent policies, and trigger keywords
    to ensure safety and policy compliance.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the escalation engine.

        Args:
            config: Optional PipelineConfig. Loaded automatically if None.
        """
        self.config = config or load_config()
        self.confidence_threshold = self.config.agent_confidence_threshold
        self.auto_escalate_intents = set(self.config.agent_auto_escalate_intents)
        self.trigger_keywords = self.config.agent_escalation_trigger_keywords

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        classifier_confidence: float,
        evidence: RetrievedEvidence | None = None,
    ) -> EscalationDecision:
        """Evaluate whether a message can be auto-handled or requires escalation.

        Args:
            customer_message: Raw customer message text.
            predicted_intent: Predicted intent string label.
            classifier_confidence: Classification probability score.
            evidence: Optional RetrievedEvidence containing historical items.

        Returns:
            EscalationDecision object.
        """
        clean_msg = customer_message.lower()

        # Rule 1: Classifier Confidence Threshold Check
        if classifier_confidence < self.confidence_threshold:
            return EscalationDecision(
                decision=ACTION_ESCALATE,
                confidence=0.90,
                reason=(
                    f"Low intent classification confidence "
                    f"({classifier_confidence:.2f} < threshold {self.confidence_threshold:.2f})"
                ),
            )

        # Rule 2: Sensitive Policy Intents (Account Access, Billing Dispute)
        if predicted_intent in self.auto_escalate_intents:
            return EscalationDecision(
                decision=ACTION_ESCALATE,
                confidence=0.95,
                reason=(
                    f"Intent '{predicted_intent}' involves sensitive operations "
                    f"and requires human verification"
                ),
            )

        # Rule 3: Configurable Trigger Keyword Categories
        for category, keywords in self.trigger_keywords.items():
            for kw in keywords:
                if kw.lower() in clean_msg:
                    cat_readable = category.replace("_", " ").title()
                    return EscalationDecision(
                        decision=ACTION_ESCALATE,
                        confidence=0.95,
                        reason=f"Detected {cat_readable} escalation keyword: '{kw}'",
                    )

        # Rule 4: Abusive or Harmful Language Signals
        abusive_terms = ["lawsuit", "attorney", "sue you", "scam", "fraud", "police", "report you"]
        for term in abusive_terms:
            if term in clean_msg:
                return EscalationDecision(
                    decision=ACTION_ESCALATE,
                    confidence=0.98,
                    reason=f"Detected high-risk legal/fraud term: '{term}'",
                )

        # Rule 5: Default Safe Automation Path
        return EscalationDecision(
            decision=ACTION_AUTO_HANDLE,
            confidence=round(classifier_confidence, 2),
            reason=f"High confidence intent '{predicted_intent}' eligible for automated response",
        )
