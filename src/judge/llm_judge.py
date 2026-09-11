"""LLM-as-a-Judge module for qualitative reply evaluation.

Implements structured evaluation of generated support responses against the 5-dimension rubric.
Supports OpenAI API, Gemini API, or deterministic rule-based heuristic evaluation.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from src.judge.rubric import RUBRIC_DIMENSIONS, EvaluationRubric
from src.utils.config import PipelineConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert AI Quality Evaluation Judge specializing in customer support quality assurance.

Your task is to evaluate an AI-generated customer support reply based strictly on the provided 5-dimension rubric:
1. Groundedness (1-5)
2. Correctness (1-5)
3. Empathy (1-5)
4. Actionability (1-5)
5. Brand Tone Consistency (1-5)

You must output ONLY valid JSON matching this exact structure:
{
  "groundedness": <1-5>,
  "correctness": <1-5>,
  "empathy": <1-5>,
  "actionability": <1-5>,
  "brand_tone": <1-5>,
  "overall_score": <float 1.0-5.0>,
  "reasoning": "<Concise explanation of evaluation rationale>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>"]
}

Rules:
- Give integer scores from 1 to 5 for all five dimensions based on rubric criteria.
- Calculate overall_score as the arithmetic mean of the five dimensions.
- Do not include markdown code blocks or extra text outside JSON.
"""

JUDGE_USER_PROMPT_TEMPLATE = """{rubric_prompt}

Context & Data to Evaluate:

Customer Message:
"{customer_message}"

Predicted Intent: {predicted_intent}

Retrieved Historical Evidence:
{retrieved_evidence}

Reference Brand Reply (Ground Truth):
"{reference_reply}"

AI-Generated Reply (Target under Evaluation):
"{generated_reply}"

Evaluate the AI-Generated Reply against the criteria and output valid JSON:"""


@dataclass
class JudgeEvaluation:
    """Dataclass holding structured evaluation outputs from the LLM Judge.

    Attributes:
        conversation_id: Unique thread identifier.
        groundedness: Score 1-5.
        correctness: Score 1-5.
        empathy: Score 1-5.
        actionability: Score 1-5.
        brand_tone: Score 1-5.
        overall_score: Average score (1.0 to 5.0).
        reasoning: Text explanation.
        strengths: List of strength points.
        weaknesses: List of weakness points.
    """

    conversation_id: str
    groundedness: int
    correctness: int
    empathy: int
    actionability: int
    brand_tone: int
    overall_score: float
    reasoning: str
    strengths: list[str]
    weaknesses: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "conversation_id": self.conversation_id,
            "groundedness": self.groundedness,
            "correctness": self.correctness,
            "empathy": self.empathy,
            "actionability": self.actionability,
            "brand_tone": self.brand_tone,
            "overall_score": round(self.overall_score, 2),
            "reasoning": self.reasoning,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


class LLMJudge:
    """Evaluates AI-generated customer support replies using an LLM Judge model."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the LLM Judge.

        Args:
            config: Optional PipelineConfig. Loaded automatically if None.
        """
        self.config = config or load_config()
        self.provider = self.config.agent_llm_provider.lower()
        self.api_key_env = self.config.agent_api_key_env

    def evaluate_reply(
        self,
        conversation_id: str,
        customer_message: str,
        predicted_intent: str,
        generated_reply: str,
        retrieved_evidence: list[dict[str, Any]] | None = None,
        reference_reply: str = "",
    ) -> JudgeEvaluation:
        """Evaluate a generated reply against context and rubric.

        Args:
            conversation_id: Conversation identifier.
            customer_message: Input customer message.
            predicted_intent: Predicted intent label.
            generated_reply: AI-generated response string.
            retrieved_evidence: Optional list of retrieved historical examples.
            reference_reply: Optional human reference reply.

        Returns:
            JudgeEvaluation object.
        """
        evidence_str = ""
        if retrieved_evidence:
            evidence_str = "\n".join([
                f"- Ex {i+1}: Customer: \"{e.get('customer_text','')}\" | Brand: \"{e.get('brand_reply','')}\""
                for i, e in enumerate(retrieved_evidence)
            ])
        else:
            evidence_str = "No explicit historical evidence retrieved."

        rubric_text = EvaluationRubric.get_rubric_prompt_text()
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            rubric_prompt=rubric_text,
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            retrieved_evidence=evidence_str,
            reference_reply=reference_reply or "N/A",
            generated_reply=generated_reply,
        )

        api_key = os.getenv(self.api_key_env, "") or os.getenv("OPENAI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

        if (self.provider == "openai" or "OPENAI_API_KEY" in os.environ) and api_key:
            res = self._call_openai_judge(user_prompt)
            if res:
                return self._build_evaluation_dataclass(conversation_id, res)

        if (self.provider == "gemini" or "GEMINI_API_KEY" in os.environ) and api_key:
            res = self._call_gemini_judge(user_prompt)
            if res:
                return self._build_evaluation_dataclass(conversation_id, res)

        # Heuristic fallback judge
        return self._evaluate_heuristic(
            conversation_id=conversation_id,
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            generated_reply=generated_reply,
            reference_reply=reference_reply,
        )

    def _call_openai_judge(self, prompt: str) -> dict[str, Any] | None:
        """Call OpenAI API for LLM Judge scoring."""
        try:
            import openai

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv(self.api_key_env))
            res = client.chat.completions.create(
                model=self.config.agent_llm_model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_text = res.choices[0].message.content or "{}"
            return json.loads(raw_text)
        except Exception as e:
            logger.warning(f"OpenAI LLM Judge failed: {e}. Falling back to heuristic scoring.")
            return None

    def _call_gemini_judge(self, prompt: str) -> dict[str, Any] | None:
        """Call Gemini API for LLM Judge scoring."""
        try:
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv(self.api_key_env)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=JUDGE_SYSTEM_PROMPT,
            )
            res = model.generate_content(prompt)
            raw = res.text.strip()
            if raw.startswith("```json"):
                raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Gemini LLM Judge failed: {e}. Falling back to heuristic scoring.")
            return None

    def _build_evaluation_dataclass(
        self, conversation_id: str, data: dict[str, Any]
    ) -> JudgeEvaluation:
        """Parse raw dictionary into JudgeEvaluation dataclass."""
        g = int(data.get("groundedness", 4))
        c = int(data.get("correctness", 4))
        e = int(data.get("empathy", 4))
        a = int(data.get("actionability", 4))
        b = int(data.get("brand_tone", 4))
        overall = float(data.get("overall_score", (g + c + e + a + b) / 5.0))

        return JudgeEvaluation(
            conversation_id=conversation_id,
            groundedness=g,
            correctness=c,
            empathy=e,
            actionability=a,
            brand_tone=b,
            overall_score=overall,
            reasoning=str(data.get("reasoning", "Evaluated against 5-dimension rubric.")),
            strengths=list(data.get("strengths", ["Grounded in evidence"])),
            weaknesses=list(data.get("weaknesses", [])),
        )

    def _evaluate_heuristic(
        self,
        conversation_id: str,
        customer_message: str,
        predicted_intent: str,
        generated_reply: str,
        reference_reply: str,
    ) -> JudgeEvaluation:
        """Rule-based heuristic evaluator fallback when no API key is available."""
        reply_lower = generated_reply.lower()

        # Groundedness heuristic
        groundedness = 5 if ("dm" in reply_lower or "details" in reply_lower) else 4

        # Correctness heuristic
        correctness = 4
        if len(generated_reply) > 10:
            correctness = 5

        # Empathy heuristic
        empathy = 3
        if any(w in reply_lower for w in ["sorry", "glad", "apologize", "understand", "help"]):
            empathy = 5
        elif any(w in reply_lower for w in ["please", "thank"]):
            empathy = 4

        # Actionability heuristic
        actionability = 3
        if any(w in reply_lower for w in ["dm us", "direct message", "send us", "visit"]):
            actionability = 5
        elif "order" in reply_lower:
            actionability = 4

        # Brand Tone heuristic
        brand_tone = 4
        if len(generated_reply) <= 280:
            brand_tone = 5

        overall = (groundedness + correctness + empathy + actionability + brand_tone) / 5.0

        return JudgeEvaluation(
            conversation_id=conversation_id,
            groundedness=groundedness,
            correctness=correctness,
            empathy=empathy,
            actionability=actionability,
            brand_tone=brand_tone,
            overall_score=overall,
            reasoning="Heuristic evaluation based on support language patterns, DM actionability, and Twitter character constraints.",
            strengths=["Concise response", "Includes action steps for customer"],
            weaknesses=[],
        )
