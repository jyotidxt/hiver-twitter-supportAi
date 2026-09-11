"""Inference engine module for the AI Support Agent pipeline.

Loads configuration, trained classifier, retrieval index, and escalation engine
into a stateful, reproducible inference runner that measures latency and logs requests.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.classifier.intent_classifier import IntentClassifier, IntentPrediction
from src.escalation.escalation import EscalationDecision, EscalationEngine
from src.reply_generator.generator import ReplyGenerator
from src.retrieval.retriever import RetrievedEvidence, SemanticRetriever
from src.utils.config import PipelineConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Dedicated file logger for agent requests
AGENT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
AGENT_LOG_FILE = AGENT_LOG_DIR / "agent.log"


def _setup_agent_logger() -> logging.Logger:
    """Set up persistent log handler for logs/agent.log."""
    AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    agent_log = logging.getLogger("agent_audit")
    if not agent_log.handlers:
        agent_log.setLevel(logging.INFO)
        fh = logging.FileHandler(AGENT_LOG_FILE, encoding="utf-8")
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | intent=%(intent)s | conf=%(conf).2f | esc=%(esc)s | ms=%(ms).1f | msg=\"%(msg_snippet)s\"",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        agent_log.addHandler(fh)
    return agent_log


agent_audit_log = _setup_agent_logger()


@dataclass
class InferenceResult:
    """Typed result container for AI Support Agent inference.

    Attributes:
        customer_message: Input customer message.
        predicted_intent: Top predicted intent label string.
        confidence: Prediction confidence probability (0.0 to 1.0).
        top_3_candidates: List of (intent_label, confidence) pairs.
        retrieved_examples: Historical retrieved conversation evidence.
        generated_reply: Synthesized customer support response string.
        escalation_decision: "AUTO_HANDLE" or "ESCALATE".
        escalation_reason: Explanation string for escalation decision.
        processing_time_ms: End-to-end inference latency in milliseconds.
    """

    customer_message: str
    predicted_intent: str
    confidence: float
    top_3_candidates: list[tuple[str, float]]
    retrieved_examples: list[dict[str, Any]]
    generated_reply: str
    escalation_decision: str
    escalation_reason: str
    processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a clean, JSON-serializable dictionary."""
        return {
            "customer_message": self.customer_message,
            "predicted_intent": self.predicted_intent,
            "confidence": round(self.confidence, 4),
            "top_3_candidates": [
                [intent, round(score, 4)] for intent, score in self.top_3_candidates
            ],
            "generated_reply": self.generated_reply,
            "escalation_decision": self.escalation_decision,
            "escalation_reason": self.escalation_reason,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "retrieved_examples": self.retrieved_examples,
        }

    def to_json(self, indent: int = 2) -> str:
        """Format as indented JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class InferenceEngine:
    """Production inference engine for the AI Support Agent.

    Uses dependency injection for sub-modules and manages pipeline lifecycle.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        classifier: IntentClassifier | None = None,
        retriever: SemanticRetriever | None = None,
        reply_generator: ReplyGenerator | None = None,
        escalation_engine: EscalationEngine | None = None,
    ) -> None:
        """Initialize the inference engine with configurable components.

        Args:
            config: Pipeline configuration. Loaded automatically if None.
            classifier: Optional custom IntentClassifier instance.
            retriever: Optional custom SemanticRetriever instance.
            reply_generator: Optional custom ReplyGenerator instance.
            escalation_engine: Optional custom EscalationEngine instance.
        """
        self.config = config or load_config()
        self.classifier = classifier or IntentClassifier(self.config)
        self.retriever = retriever or SemanticRetriever(self.config)
        self.reply_generator = reply_generator or ReplyGenerator(self.config)
        self.escalation_engine = escalation_engine or EscalationEngine(self.config)
        self.is_loaded = False

    def load(self) -> None:
        """Train classifier and build retrieval index if not already loaded."""
        if not self.is_loaded:
            logger.info("Loading InferenceEngine models and retrieval index...")
            if not self.classifier.is_trained:
                self.classifier.train()
            if not self.retriever.is_indexed:
                self.retriever.build_index()
            self.is_loaded = True
            logger.info("InferenceEngine loaded successfully.")

    def run(self, customer_message: str) -> InferenceResult:
        """Execute end-to-end inference for a customer message.

        Args:
            customer_message: Raw incoming customer tweet text.

        Returns:
            InferenceResult object with structured predictions, reply, and latency.
        """
        if not self.is_loaded:
            self.load()

        start_time = time.perf_counter()
        clean_text = customer_message.strip()

        # Step 1: Classification
        prediction: IntentPrediction = self.classifier.predict(clean_text)

        # Step 2: Semantic Retrieval
        evidence: RetrievedEvidence = self.retriever.retrieve(
            clean_text, top_k=self.config.agent_top_k
        )

        # Step 3: Escalation Decision
        escalation: EscalationDecision = self.escalation_engine.evaluate(
            customer_message=clean_text,
            predicted_intent=prediction.predicted_intent,
            classifier_confidence=prediction.confidence,
            evidence=evidence,
        )

        # Step 4: Grounded Reply Synthesis
        generated_reply: str = self.reply_generator.generate_reply(
            customer_message=clean_text,
            predicted_intent=prediction.predicted_intent,
            evidence=evidence,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        retrieved_list = evidence.to_dict()["items"]

        result = InferenceResult(
            customer_message=clean_text,
            predicted_intent=prediction.predicted_intent,
            confidence=prediction.confidence,
            top_3_candidates=prediction.top_3_candidates,
            retrieved_examples=retrieved_list,
            generated_reply=generated_reply,
            escalation_decision=escalation.decision,
            escalation_reason=escalation.reason,
            processing_time_ms=elapsed_ms,
        )

        # Audit log to logs/agent.log (never logs secrets)
        msg_snippet = (clean_text[:40] + "...") if len(clean_text) > 40 else clean_text
        agent_audit_log.info(
            "",
            extra={
                "intent": result.predicted_intent,
                "conf": result.confidence,
                "esc": result.escalation_decision,
                "ms": result.processing_time_ms,
                "msg_snippet": msg_snippet.replace('"', "'"),
            },
        )

        return result
