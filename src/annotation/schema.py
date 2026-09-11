"""Canonical annotation schema for the Golden Dataset.

Defines the data models used throughout the annotation pipeline.
All annotation records must conform to this schema before export.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Intent(str, Enum):
    """Valid primary intent labels.

    Derived from planning/03_INTENT_TAXONOMY.md.
    Each value maps to a specific customer support action.
    """

    ORDER_STATUS = "order_status"
    ORDER_MODIFICATION = "order_modification"
    ORDER_MISSING_WRONG = "order_missing_wrong"
    REFUND_REQUEST = "refund_request"
    BILLING_ISSUE = "billing_issue"
    DELIVERY_PROBLEM = "delivery_problem"
    SHIPPING_INQUIRY = "shipping_inquiry"
    ACCOUNT_ACCESS = "account_access"
    ACCOUNT_MANAGEMENT = "account_management"
    PRODUCT_ISSUE = "product_issue"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_INQUIRY = "general_inquiry"

    @classmethod
    def values(cls) -> list[str]:
        """Return all valid intent string values."""
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, value: str) -> Intent:
        """Create an Intent from a string, case-insensitive.

        Args:
            value: The intent string to parse.

        Returns:
            The matching Intent enum member.

        Raises:
            ValueError: If the string does not match any intent.
        """
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return cls(normalized)
        except ValueError:
            valid = ", ".join(cls.values())
            raise ValueError(
                f"Invalid intent '{value}'. Valid intents: {valid}"
            )


class Confidence(str, Enum):
    """Annotator confidence levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def values(cls) -> list[str]:
        """Return all valid confidence string values."""
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, value: str) -> Confidence:
        """Create a Confidence from a string, case-insensitive.

        Args:
            value: The confidence string to parse.

        Returns:
            The matching Confidence enum member.

        Raises:
            ValueError: If the string does not match any level.
        """
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            valid = ", ".join(cls.values())
            raise ValueError(
                f"Invalid confidence '{value}'. Valid levels: {valid}"
            )


class EscalationReason(str, Enum):
    """Controlled escalation reason codes.

    From planning/02_LABEL_GUIDELINES.md.
    """

    IDENTITY_VERIFICATION = "identity_verification"
    REFUND_PAYMENT = "refund_payment"
    LEGAL_PRIVACY = "legal_privacy"
    ACCOUNT_SECURITY = "account_security"
    DAMAGED_DEFECTIVE = "damaged_defective"
    ABUSIVE_CUSTOMER = "abusive_customer"
    MISSING_INFORMATION = "missing_information"
    POLICY_EXCEPTION = "policy_exception"
    NONE = "none"

    @classmethod
    def values(cls) -> list[str]:
        """Return all valid escalation reason string values."""
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, value: str) -> EscalationReason:
        """Create an EscalationReason from a string, case-insensitive.

        Args:
            value: The reason string to parse.

        Returns:
            The matching EscalationReason enum member.

        Raises:
            ValueError: If the string does not match any reason.
        """
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in ("", "n/a", "na"):
            return cls.NONE
        try:
            return cls(normalized)
        except ValueError:
            valid = ", ".join(v for v in cls.values() if v != "none")
            raise ValueError(
                f"Invalid escalation reason '{value}'. Valid reasons: {valid}"
            )


@dataclass
class AnnotationRecord:
    """A single annotation entry for the golden dataset.

    Attributes:
        conversation_id: Unique identifier for the conversation thread.
        primary_intent: The primary intent label from the taxonomy.
        escalation: Whether the conversation requires human escalation.
        escalation_reason: The reason for escalation (required if escalation is True).
        confidence: Annotator's confidence in the label.
        annotator_notes: Optional free-text notes about the annotation.
        annotated_at: ISO 8601 timestamp of when the annotation was created.
    """

    conversation_id: str
    primary_intent: str
    escalation: bool
    escalation_reason: str = "none"
    confidence: str = "high"
    annotator_notes: str = ""
    annotated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> list[str]:
        """Validate this annotation record.

        Returns:
            List of validation error messages. Empty list means valid.
        """
        errors: list[str] = []

        # Conversation ID
        if not self.conversation_id or not self.conversation_id.strip():
            errors.append("conversation_id is empty")

        # Intent validation
        if not self.primary_intent or not self.primary_intent.strip():
            errors.append("primary_intent is empty")
        elif self.primary_intent not in Intent.values():
            errors.append(
                f"Invalid intent '{self.primary_intent}'. "
                f"Valid: {Intent.values()}"
            )

        # Confidence validation
        if self.confidence not in Confidence.values():
            errors.append(
                f"Invalid confidence '{self.confidence}'. "
                f"Valid: {Confidence.values()}"
            )

        # Escalation validation
        if self.escalation:
            if (
                not self.escalation_reason
                or self.escalation_reason == "none"
                or not self.escalation_reason.strip()
            ):
                errors.append(
                    "escalation_reason is required when escalation is True"
                )
            elif self.escalation_reason not in EscalationReason.values():
                errors.append(
                    f"Invalid escalation_reason '{self.escalation_reason}'. "
                    f"Valid: {[v for v in EscalationReason.values() if v != 'none']}"
                )

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for CSV export.

        Returns:
            Dictionary with all fields.
        """
        return {
            "conversation_id": self.conversation_id,
            "primary_intent": self.primary_intent,
            "escalation": self.escalation,
            "escalation_reason": self.escalation_reason,
            "confidence": self.confidence,
            "annotator_notes": self.annotator_notes,
            "annotated_at": self.annotated_at,
        }


# Column order for CSV export
GOLDEN_DATASET_COLUMNS = [
    "conversation_id",
    "primary_intent",
    "escalation",
    "escalation_reason",
    "confidence",
    "annotator_notes",
    "annotated_at",
]

# Intent display helpers for the annotation tool
INTENT_CATEGORIES = {
    "Orders": [
        ("order_status", "Where is my order?"),
        ("order_modification", "Change or cancel an order"),
        ("order_missing_wrong", "Missing items or wrong product received"),
    ],
    "Payments & Refunds": [
        ("refund_request", "Requesting refund or return"),
        ("billing_issue", "Charges, pricing, payment problems"),
    ],
    "Shipping & Delivery": [
        ("delivery_problem", "Late, lost, damaged in transit"),
        ("shipping_inquiry", "Shipping options, costs, timeframes"),
    ],
    "Account & Access": [
        ("account_access", "Login, password, locked account"),
        ("account_management", "Profile, Prime, subscriptions"),
    ],
    "Product & Technical": [
        ("product_issue", "Defective, broken, quality complaints"),
        ("technical_support", "App, website, device issues"),
    ],
    "General": [
        ("general_inquiry", "Policy, praise, feedback, unclear"),
    ],
}

ESCALATION_REASON_DESCRIPTIONS = {
    "identity_verification": "Need to confirm customer identity",
    "refund_payment": "Monetary compensation or charge reversal",
    "legal_privacy": "Legal obligations or data privacy",
    "account_security": "Compromised account or unauthorized access",
    "damaged_defective": "Physical product damage or defect",
    "abusive_customer": "Profanity, harassment, or TOS violation",
    "missing_information": "Essential details absent for resolution",
    "policy_exception": "Situation outside standard policy",
}
