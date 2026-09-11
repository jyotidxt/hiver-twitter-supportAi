"""Prompt templates for grounded customer support reply generation.

Stores all system and user prompt strings separately from generator logic.
Ensures replies are empathetic, concise, actionable, and grounded in evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.retriever import RetrievedEvidence


SYSTEM_PROMPT = """You are an AI Customer Support Agent representing @AmazonHelp on Twitter.

Your task is to draft a customer support response to an incoming tweet.

STRICT CONSTRAINTS & GUIDELINES:
1. Grounding: Rely ONLY on the provided historical resolution evidence and predicted intent. Never invent policies or procedures not supported by the evidence.
2. Tone: Professional, polite, concise, and empathetic. Match official customer support social media tone.
3. Length: Keep the response under 280 characters when possible (suitable for Twitter customer support).
4. Directing to DM: If personal info (order ID, account info, email) is needed, direct the customer to send a Direct Message (DM).
5. No Guarantees: Do not promise specific refunds or financial compensation unless confirmed by policy; ask for details via DM first.
6. Formatting: Plain text only, no markdown bolding or internal notes. Do not include quotes around the reply.
"""

USER_PROMPT_TEMPLATE = """Brand: {brand_handle} ({brand_name})
Predicted Intent: {predicted_intent}

Incoming Customer Message:
"{customer_message}"

Retrieved Historical Resolution Evidence:
{historical_evidence}

Draft a helpful, concise, and empathetic support reply to the customer:"""


def build_user_prompt(
    customer_message: str,
    predicted_intent: str,
    evidence: RetrievedEvidence | None = None,
    brand_name: str = "AmazonHelp",
    brand_handle: str = "@AmazonHelp",
) -> str:
    """Format the user prompt using input query, intent, and retrieved evidence.

    Args:
        customer_message: Raw incoming customer message text.
        predicted_intent: Predicted primary intent string.
        evidence: Optional RetrievedEvidence object containing top historical examples.
        brand_name: Brand name string.
        brand_handle: Brand Twitter handle.

    Returns:
        Formatted user prompt string.
    """
    if evidence and evidence.items:
        evidence_lines = []
        for idx, item in enumerate(evidence.items, 1):
            evidence_lines.append(
                f"Example {idx} (Similarity: {item.similarity_score:.2f}):\n"
                f"  Customer: \"{item.customer_text}\"\n"
                f"  Brand Reply: \"{item.brand_reply}\""
            )
        evidence_str = "\n".join(evidence_lines)
    else:
        evidence_str = "No specific historical resolution evidence retrieved."

    return USER_PROMPT_TEMPLATE.format(
        brand_name=brand_name,
        brand_handle=brand_handle,
        predicted_intent=predicted_intent,
        customer_message=customer_message,
        historical_evidence=evidence_str,
    )
