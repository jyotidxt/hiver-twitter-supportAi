"""Evaluation rubric module for qualitative LLM-as-Judge scoring.

Defines the 5 core quality dimensions (Groundedness, Correctness, Empathy, Actionability,
and Brand Tone Consistency) with explicit 1–5 score definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ScoreLevelDefinition:
    """Definition for a single 1-5 score level on a rubric dimension."""

    score: int
    label: str
    description: str


@dataclass
class RubricDimension:
    """Definition of a single evaluation dimension."""

    name: str
    key: str
    description: str
    score_levels: dict[int, ScoreLevelDefinition]


RUBRIC_DIMENSIONS: dict[str, RubricDimension] = {
    "groundedness": RubricDimension(
        name="Groundedness",
        key="groundedness",
        description="Assesses whether the response relies strictly on retrieved historical evidence without hallucinating unverified policies.",
        score_levels={
            1: ScoreLevelDefinition(1, "Severe Hallucination", "Contains fabricated policies or promises unsupported by evidence."),
            2: ScoreLevelDefinition(2, "Minor Hallucination", "Mostly grounded but includes speculative or unverified policy claims."),
            3: ScoreLevelDefinition(3, "Partially Grounded", "Supported by general support principles but lacks explicit evidence backing."),
            4: ScoreLevelDefinition(4, "Well Grounded", "Accurately reflects retrieved historical resolution patterns."),
            5: ScoreLevelDefinition(5, "Fully Grounded", "Strictly anchored in retrieved evidence with zero speculative statements."),
        },
    ),
    "correctness": RubricDimension(
        name="Correctness",
        key="correctness",
        description="Evaluates whether the response correctly addresses the customer's specific intent and core problem.",
        score_levels={
            1: ScoreLevelDefinition(1, "Incorrect", "Completely misunderstands the intent or gives wrong instructions."),
            2: ScoreLevelDefinition(2, "Mostly Incorrect", "Addresses a secondary topic while missing the core customer problem."),
            3: ScoreLevelDefinition(3, "Partially Correct", "Addresses the intent generally but misses specific details."),
            4: ScoreLevelDefinition(4, "Correct", "Accurately addresses the customer's intent and provides proper guidance."),
            5: ScoreLevelDefinition(5, "Perfectly Correct", "Flawlessly identifies and resolves the customer's specific query."),
        },
    ),
    "empathy": RubricDimension(
        name="Empathy",
        key="empathy",
        description="Measures politeness, tone, and appropriate expression of concern for customer frustration.",
        score_levels={
            1: ScoreLevelDefinition(1, "Rude / Hostile", "Dismissive, aggressive, or completely mechanical and cold."),
            2: ScoreLevelDefinition(2, "Cold / Robotic", "Lacks greeting or basic courtesy phrases; feels transactional."),
            3: ScoreLevelDefinition(3, "Neutral", "Standard polite greeting without explicit empathy for frustration."),
            4: ScoreLevelDefinition(4, "Empathetic", "Expresses clear apology/understanding for customer inconvenience."),
            5: ScoreLevelDefinition(5, "Highly Empathetic", "Exemplary warm, supportive, and reassuring customer support tone."),
        },
    ),
    "actionability": RubricDimension(
        name="Actionability",
        key="actionability",
        description="Checks if the response provides clear, unambiguous next steps (e.g. requesting DM with order details).",
        score_levels={
            1: ScoreLevelDefinition(1, "Unactionable", "Provides no next steps or leaves customer stranded."),
            2: ScoreLevelDefinition(2, "Vague", "Tells customer to get help without specifying how or where."),
            3: ScoreLevelDefinition(3, "Basic Action", "Provides general guidance (e.g., 'check website')."),
            4: ScoreLevelDefinition(4, "Clear Action", "Directs customer to send a DM with specific needed details (order ID, email)."),
            5: ScoreLevelDefinition(5, "Highly Actionable", "Provides immediate, step-by-step resolution instructions."),
        },
    ),
    "brand_tone": RubricDimension(
        name="Brand Tone Consistency",
        key="brand_tone",
        description="Assesses alignment with official @AmazonHelp social media voice and character limit constraints.",
        score_levels={
            1: ScoreLevelDefinition(1, "Off-Brand", "Unprofessional language, inappropriate formatting, or overly verbose."),
            2: ScoreLevelDefinition(2, "Inconsistent", "Deviates from social support conventions; includes markdown formatting."),
            3: ScoreLevelDefinition(3, "Acceptable", "Meets basic brand tone expectations."),
            4: ScoreLevelDefinition(4, "On-Brand", "Matches @AmazonHelp Twitter support voice well; concise and professional."),
            5: ScoreLevelDefinition(5, "Flawless Brand Tone", "Indistinguishable from official @AmazonHelp social support responses."),
        },
    ),
}


class EvaluationRubric:
    """Validation and utility helper for the LLM judge rubric."""

    @staticmethod
    def validate_scores(scores: dict[str, Any]) -> list[str]:
        """Validate that score dictionary contains valid 1-5 integer scores across all 5 dimensions.

        Args:
            scores: Dictionary mapping dimension keys to score values.

        Returns:
            List of validation error strings. Empty list means valid.
        """
        errors = []
        for key in RUBRIC_DIMENSIONS:
            if key not in scores:
                errors.append(f"Missing required dimension score: '{key}'")
            else:
                val = scores[key]
                try:
                    ival = int(val)
                    if not (1 <= ival <= 5):
                        errors.append(f"Score for '{key}' must be between 1 and 5 (got {val})")
                except (ValueError, TypeError):
                    errors.append(f"Score for '{key}' must be an integer (got {val})")

        return errors

    @staticmethod
    def get_rubric_prompt_text() -> str:
        """Format the full rubric definitions into prompt text for the LLM Judge."""
        lines = ["Rubric Scoring Criteria (Scale 1 to 5):"]
        for key, dim in RUBRIC_DIMENSIONS.items():
            lines.append(f"\n[{dim.name}] - {dim.description}")
            for lvl, defn in dim.score_levels.items():
                lines.append(f"  Score {lvl}: {defn.label} — {defn.description}")
        return "\n".join(lines)
