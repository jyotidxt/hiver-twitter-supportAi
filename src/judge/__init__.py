"""LLM-as-Judge and Human Agreement evaluation package for Hiver Support AI."""

from src.judge.agreement import compute_human_llm_agreement
from src.judge.llm_judge import LLMJudge, JudgeEvaluation
from src.judge.report import generate_judge_report
from src.judge.rubric import EvaluationRubric, RubricDimension

__all__ = [
    "EvaluationRubric",
    "RubricDimension",
    "LLMJudge",
    "JudgeEvaluation",
    "compute_human_llm_agreement",
    "generate_judge_report",
]
