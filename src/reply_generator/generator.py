"""Grounded reply generator module.

Uses prompt templates from prompts.py, predicted intent, and retrieved evidence
to synthesize brand-grounded customer support replies.

Supports OpenAI API, Gemini API, or fallback template-based generation.
"""

from __future__ import annotations

import os
from typing import Any

from src.reply_generator.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.retriever import RetrievedEvidence
from src.utils.config import PipelineConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Template-based fallback replies for offline / keyless execution
INTENT_TEMPLATE_REPLIES = {
    "order_status": (
        "We'd be glad to look into your order status! Please send us a DM with your "
        "order number and email address so we can check tracking for you."
    ),
    "order_modification": (
        "We can assist with updating or canceling your order if it hasn't shipped yet. "
        "Please send us a DM with your order ID and the requested changes."
    ),
    "order_missing_wrong": (
        "We're so sorry to hear an item was missing or incorrect! Please DM us your order "
        "number and details so we can arrange a replacement right away."
    ),
    "refund_request": (
        "We understand you're looking for a refund. Please DM us your order number and "
        "email so our support team can verify the return details and process this for you."
    ),
    "billing_issue": (
        "We apologize for the billing concern. Please send us a DM with your order number "
        "and transaction details so we can investigate and correct any charge errors."
    ),
    "delivery_problem": (
        "We're sorry to hear your package hasn't arrived safely! Please DM us your order ID "
        "and shipping address so we can investigate with the carrier."
    ),
    "shipping_inquiry": (
        "Thank you for reaching out! You can view standard shipping options and delivery "
        "timeframes during checkout or by DMing us your order details."
    ),
    "account_access": (
        "We take account security seriously. Please visit our online Account Help center "
        "or DM us for guidance on recovering your account safely."
    ),
    "account_management": (
        "We can guide you with your account or Prime settings! Please DM us if you need "
        "assistance updating your profile details."
    ),
    "product_issue": (
        "We're very sorry your product arrived in poor condition. Please send us a DM with "
        "your order number so we can help with a replacement or return."
    ),
    "technical_support": (
        "We're sorry for the technical issue! Please try restarting the app or device, "
        "and DM us if you continue experiencing errors so we can troubleshoot."
    ),
    "general_inquiry": (
        "Thank you for contacting support! Please let us know if you need assistance with "
        "an order or product and we'll be happy to help via DM."
    ),
}


class ReplyGenerator:
    """Grounded reply generator synthesizing support responses.

    Combines system prompt guidelines, predicted intent, and retrieved evidence
    to produce empathetic, concise, and policy-compliant replies.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the reply generator.

        Args:
            config: Optional PipelineConfig. Loaded automatically if None.
        """
        self.config = config or load_config()
        self.provider = self.config.agent_llm_provider.lower()
        self.api_key_env = self.config.agent_api_key_env
        self.model_name = self.config.agent_llm_model_name

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        evidence: RetrievedEvidence | None = None,
    ) -> str:
        """Generate a grounded customer support reply.

        Args:
            customer_message: Raw incoming customer tweet text.
            predicted_intent: Predicted intent string label.
            evidence: Optional RetrievedEvidence containing historical examples.

        Returns:
            Generated support reply string.
        """
        prompt = build_user_prompt(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            evidence=evidence,
            brand_name=self.config.selected_brand,
            brand_handle=self.config.brand_handle,
        )

        api_key = os.getenv(self.api_key_env, "")

        if self.provider == "openai" and api_key:
            return self._call_openai(prompt)

        if self.provider == "gemini" and (api_key or os.getenv("GEMINI_API_KEY")):
            return self._call_gemini(prompt)

        # Fallback to grounded template generator
        return self._generate_template_reply(predicted_intent, evidence)

    def _call_openai(self, prompt: str) -> str:
        """Invoke OpenAI API to generate response."""
        try:
            import openai

            client = openai.OpenAI(api_key=os.getenv(self.api_key_env))
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.agent_max_tokens,
                temperature=self.config.agent_temperature,
            )
            reply = response.choices[0].message.content or ""
            return reply.strip()
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}. Falling back to template generation.")
            return self._generate_template_reply("general_inquiry", None)

    def _call_gemini(self, prompt: str) -> str:
        """Invoke Gemini API to generate response."""
        try:
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv(self.api_key_env)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=SYSTEM_PROMPT,
            )
            response = model.generate_content(prompt)
            reply = response.text or ""
            return reply.strip()
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to template generation.")
            return self._generate_template_reply("general_inquiry", None)

    def _generate_template_reply(
        self, intent: str, evidence: RetrievedEvidence | None
    ) -> str:
        """Generate a deterministic, evidence-grounded template reply."""
        # Use evidence brand reply if top score is high (> 0.75)
        if evidence and evidence.items:
            top_item = evidence.items[0]
            if top_item.similarity_score >= 0.75 and top_item.brand_reply:
                return top_item.brand_reply

        # Fallback to intent-specific template reply
        return INTENT_TEMPLATE_REPLIES.get(
            intent, INTENT_TEMPLATE_REPLIES["general_inquiry"]
        )
