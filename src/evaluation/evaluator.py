"""Automated evaluation harness module.

Orchestrates quantitative evaluation of Baseline 1, Baseline 2, and the Phase 4 AI Agent
on the golden dataset. Generates structured prediction CSVs, metric JSONs, confusion matrix PNGs,
and comparison tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.annotation.schema import Intent
from src.evaluation.baselines import Baseline1_Trivial, Baseline2_Simple
from src.evaluation.metrics import (
    compute_escalation_metrics,
    compute_intent_metrics,
    plot_confusion_matrix,
)
from src.pipeline.inference import InferenceEngine
from src.utils.config import PipelineConfig, load_config
from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)

EVAL_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "evaluation"


class EvaluationHarness:
    """Quantitative evaluation harness comparing AI Agent against baselines."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize evaluation harness.

        Args:
            config: Optional PipelineConfig. Loaded automatically if None.
        """
        self.config = config or load_config()
        self.output_dir = EVAL_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.b1 = Baseline1_Trivial()
        self.b2 = Baseline2_Simple()
        self.agent_engine = InferenceEngine(self.config)

    def load_golden_eval_data(self) -> pd.DataFrame:
        """Load and prepare golden dataset for evaluation.

        Combines golden intent annotations with customer query text.
        Falls back to synthetic evaluation set if golden dataset file is unpopulated.

        Returns:
            DataFrame containing 'conversation_id', 'customer_text', 'primary_intent', 'escalation'.
        """
        golden_path = self.config.agent_golden_dataset_path
        ann_path = self.config.annotation_ready_path

        if golden_path.exists():
            try:
                gdf = pd.read_csv(golden_path)
                if len(gdf) > 0 and "primary_intent" in gdf.columns:
                    if ann_path.exists():
                        adf = pd.read_csv(ann_path)
                        merged = pd.merge(gdf, adf, on="conversation_id", how="inner")
                        if len(merged) > 0 and "opening_message" in merged.columns:
                            merged["customer_text"] = merged["opening_message"]
                            return merged
                    if "customer_text" in gdf.columns:
                        return gdf
            except Exception as e:
                logger.warning(f"Could not load golden dataset from {golden_path}: {e}")

        logger.info("Generating evaluation dataset from synthetic benchmark suite...")
        return self._create_synthetic_eval_dataset()

    def _create_synthetic_eval_dataset(self) -> pd.DataFrame:
        """Create a diverse 30-sample evaluation dataset covering all 12 taxonomy intents."""
        eval_samples = [
            ("Where is my order #1001? Tracking says in transit.", "order_status", False),
            ("Can you give me an update on my package delivery?", "order_status", False),
            ("When will my order arrive? Expected date passed.", "order_status", False),
            ("Please cancel my order #2002 before it ships.", "order_modification", False),
            ("I need to change my shipping address on order #2003.", "order_modification", False),
            ("Remove item 2 from my order #2004", "order_modification", False),
            ("My order arrived but one item is missing from the box.", "order_missing_wrong", False),
            ("You sent me the wrong color shoes in order #3001.", "order_missing_wrong", False),
            ("Package arrived empty, no contents inside.", "order_missing_wrong", True),
            ("I want a full refund for my return on order #4001.", "refund_request", True),
            ("Where is my refund for returned item #4002?", "refund_request", True),
            ("Requesting return label for refund", "refund_request", False),
            ("I was charged twice on my credit card for order #5001.", "billing_issue", True),
            ("Unauthorized charge of $49.99 on my account.", "billing_issue", True),
            ("Double billing deduction on my bank statement", "billing_issue", True),
            ("Package marked delivered but not on my porch.", "delivery_problem", True),
            ("Driver left package in wrong building rain.", "delivery_problem", True),
            ("Box was crushed and stolen from porch", "delivery_problem", True),
            ("Do you offer express 1-day shipping to NY?", "shipping_inquiry", False),
            ("What are the international shipping options?", "shipping_inquiry", False),
            ("Cannot log into my account password reset link expired.", "account_access", True),
            ("Account locked due to suspicious activity alert.", "account_access", True),
            ("Account hacked someone ordered without my permission", "account_access", True),
            ("How do I cancel my Prime membership subscription?", "account_management", False),
            ("Update my profile email and phone number", "account_management", False),
            ("The blender stopped working after two uses.", "product_issue", True),
            ("Item arrived with broken cracked glass screen.", "product_issue", True),
            ("Amazon app keeps crashing when opening cart.", "technical_support", False),
            ("Kindle device screen is completely frozen.", "technical_support", False),
            ("What is your 30 day return policy?", "general_inquiry", False),
        ]

        df = pd.DataFrame(eval_samples, columns=["customer_text", "primary_intent", "escalation"])
        df["conversation_id"] = [f"eval_{i:03d}" for i in range(len(df))]
        df["brand_reply"] = [
            "Please DM us your order ID so we can assist you." for _ in range(len(df))
        ]
        return df

    def run_evaluations(self) -> dict[str, Any]:
        """Run all systems (Baseline 1, Baseline 2, AI Agent) on golden dataset.

        Returns:
            Dictionary of metrics and comparison results.
        """
        log_section(logger, "STARTING QUANTITATIVE EVALUATION HARNESS")

        df = self.load_golden_eval_data()
        logger.info(f"Loaded {len(df)} golden evaluation samples.")

        # Fit Baseline 2
        self.b2.fit(df)
        self.agent_engine.load()

        predictions_records = []

        # Run predictions
        for idx, row in df.iterrows():
            cid = str(row.get("conversation_id", f"conv_{idx}"))
            text = str(row.get("customer_text", ""))
            true_intent = str(row.get("primary_intent", "general_inquiry"))
            true_esc = bool(row.get("escalation", False))
            ref_reply = str(row.get("brand_reply", ""))

            # Baseline 1
            p1 = self.b1.predict(text)

            # Baseline 2
            p2 = self.b2.predict(text)

            # AI Agent
            p_agent = self.agent_engine.run(text)

            predictions_records.append({
                "conversation_id": cid,
                "customer_text": text,
                "true_intent": true_intent,
                "true_escalation": true_esc,
                "reference_reply": ref_reply,
                # Baseline 1
                "b1_intent": p1.predicted_intent,
                "b1_confidence": p1.confidence,
                "b1_escalation": p1.escalation_decision,
                "b1_reply": p1.generated_reply,
                # Baseline 2
                "b2_intent": p2.predicted_intent,
                "b2_confidence": p2.confidence,
                "b2_escalation": p2.escalation_decision,
                "b2_reply": p2.generated_reply,
                # AI Agent
                "agent_intent": p_agent.predicted_intent,
                "agent_confidence": p_agent.confidence,
                "agent_escalation": p_agent.escalation_decision,
                "agent_reply": p_agent.generated_reply,
                "agent_latency_ms": p_agent.processing_time_ms,
            })

        pred_df = pd.DataFrame(predictions_records)
        pred_csv = self.output_dir / "predictions.csv"
        pred_df.to_csv(pred_csv, index=False, encoding="utf-8")
        logger.info(f"Saved predictions CSV: {pred_csv}")

        # Compute Intent Metrics
        y_true_intent = pred_df["true_intent"].tolist()

        b1_intent_m = compute_intent_metrics(y_true_intent, pred_df["b1_intent"].tolist())
        b2_intent_m = compute_intent_metrics(y_true_intent, pred_df["b2_intent"].tolist())
        agent_intent_m = compute_intent_metrics(y_true_intent, pred_df["agent_intent"].tolist())

        intent_metrics_all = {
            "baseline_1_trivial": b1_intent_m,
            "baseline_2_simple": b2_intent_m,
            "ai_agent_final": agent_intent_m,
        }

        with open(self.output_dir / "intent_metrics.json", "w", encoding="utf-8") as f:
            json.dump(intent_metrics_all, f, indent=2)

        # Plot Confusion Matrix for AI Agent
        classes = agent_intent_m.get("classes", list(set(y_true_intent)))
        cm = agent_intent_m.get("confusion_matrix", [])
        if cm and classes:
            plot_confusion_matrix(
                cm, classes, self.output_dir / "confusion_matrix.png"
            )

        # Compute Escalation Metrics
        y_true_esc = pred_df["true_escalation"].tolist()

        b1_esc_m = compute_escalation_metrics(y_true_esc, pred_df["b1_escalation"].tolist())
        b2_esc_m = compute_escalation_metrics(y_true_esc, pred_df["b2_escalation"].tolist())
        agent_esc_m = compute_escalation_metrics(y_true_esc, pred_df["agent_escalation"].tolist())

        esc_metrics_all = {
            "baseline_1_trivial": b1_esc_m,
            "baseline_2_simple": b2_esc_m,
            "ai_agent_final": agent_esc_m,
        }

        with open(self.output_dir / "escalation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(esc_metrics_all, f, indent=2)

        # Build Comparison Table CSV
        comp_data = [
            {
                "System": "Baseline 1 (Majority Trivial)",
                "Intent Accuracy": b1_intent_m["accuracy"],
                "Intent F1 (Macro)": b1_intent_m["f1_macro"],
                "Escalation Accuracy": b1_esc_m["accuracy"],
                "Escalation F1": b1_esc_m["f1"],
                "False Positive Rate": b1_esc_m["false_positive_rate"],
                "False Negative Rate": b1_esc_m["false_negative_rate"],
            },
            {
                "System": "Baseline 2 (TF-IDF Nearest Neighbor)",
                "Intent Accuracy": b2_intent_m["accuracy"],
                "Intent F1 (Macro)": b2_intent_m["f1_macro"],
                "Escalation Accuracy": b2_esc_m["accuracy"],
                "Escalation F1": b2_esc_m["f1"],
                "False Positive Rate": b2_esc_m["false_positive_rate"],
                "False Negative Rate": b2_esc_m["false_negative_rate"],
            },
            {
                "System": "Phase 4 AI Support Agent",
                "Intent Accuracy": agent_intent_m["accuracy"],
                "Intent F1 (Macro)": agent_intent_m["f1_macro"],
                "Escalation Accuracy": agent_esc_m["accuracy"],
                "Escalation F1": agent_esc_m["f1"],
                "False Positive Rate": agent_esc_m["false_positive_rate"],
                "False Negative Rate": agent_esc_m["false_negative_rate"],
            },
        ]

        comp_df = pd.DataFrame(comp_data)
        comp_csv = self.output_dir / "baseline_comparison.csv"
        comp_df.to_csv(comp_csv, index=False, encoding="utf-8")
        logger.info(f"Saved baseline comparison CSV: {comp_csv}")

        log_section(logger, "QUANTITATIVE EVALUATION COMPLETE")
        log_metric(logger, "Golden Samples Evaluated", len(df))
        log_metric(logger, "AI Agent Intent Accuracy", f"{agent_intent_m['accuracy']:.2%}")
        log_metric(logger, "AI Agent Intent F1 (Macro)", f"{agent_intent_m['f1_macro']:.4f}")
        log_metric(logger, "AI Agent Escalation F1", f"{agent_esc_m['f1']:.4f}")
        log_metric(logger, "AI Agent False Negative Rate", f"{agent_esc_m['false_negative_rate']:.2%}")

        return {
            "predictions_df": pred_df,
            "intent_metrics": intent_metrics_all,
            "escalation_metrics": esc_metrics_all,
            "comparison_df": comp_df,
        }
