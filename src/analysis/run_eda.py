"""Main EDA pipeline orchestrator.

Runs the complete exploratory data analysis from processed data
to final report. This is the primary entry point for Phase 2.2.

Usage:
    python -m src.analysis.run_eda
    python -m src.analysis.run_eda --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analysis.conversation_analysis import analyze_conversations
from src.analysis.dataset_overview import (
    compute_overview,
    export_overview_json,
    export_overview_markdown,
)
from src.analysis.intent_discovery import (
    analyze_customer_language,
    analyze_escalation_patterns,
    discover_intents,
)
from src.analysis.report_generator import generate_report
from src.analysis.visualization import generate_all_charts
from src.utils.config import load_config
from src.utils.logger import Colors, get_logger, log_metric, log_section


def main(config_path: str | None = None) -> None:
    """Run the complete EDA pipeline.

    Args:
        config_path: Optional path to configuration file.
    """
    start_time = time.time()

    # Load configuration
    config = load_config(config_path)
    logger = get_logger(
        "hiver-eda",
        level=config.log_level,
        log_file=config.log_file,
    )

    # Banner
    logger.info("")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}  HIVER SUPPORT AI — EXPLORATORY DATA ANALYSIS{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}  Phase 2.2: EDA & Brand Intelligence{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    logger.info("")
    logger.info(f"  Brand:  {Colors.BOLD}{config.selected_brand}{Colors.RESET}")
    logger.info(f"  Input:  {config.processed_conversations_path}")
    logger.info(f"  Output: {config.eda_output_dir}")
    logger.info("")

    # ── Load processed data ──
    log_section(logger, "LOADING PROCESSED DATA")

    if not config.processed_conversations_path.exists():
        logger.error(
            f"Processed data not found: {config.processed_conversations_path}"
        )
        logger.error(
            "Run the preprocessing pipeline first:\n"
            "  python -m src.preprocessing.run_pipeline"
        )
        sys.exit(1)

    df = pd.read_csv(
        config.processed_conversations_path,
        dtype={"tweet_id": str, "author_id": str, "conversation_id": str},
        parse_dates=["created_at"],
        low_memory=False,
    )
    logger.info(f"  Loaded {len(df):,} messages from processed conversations")

    # Also load annotation-ready if available
    annotation_df = None
    if config.annotation_ready_path.exists():
        annotation_df = pd.read_csv(
            config.annotation_ready_path,
            dtype={"conversation_id": str},
        )
        logger.info(f"  Loaded {len(annotation_df):,} conversations from annotation-ready dataset")

    # Load preprocessing summary if available
    preprocess_summary = {}
    if config.preprocessing_summary_path.exists():
        with open(config.preprocessing_summary_path, "r", encoding="utf-8") as f:
            preprocess_summary = json.load(f)
        logger.info("  Loaded preprocessing summary")

    # ── Step 1: Dataset Overview ──
    overview_stats = compute_overview(df, config)
    export_overview_json(overview_stats, config.eda_output_dir)
    export_overview_markdown(overview_stats, config.eda_output_dir, config.selected_brand)

    # ── Step 2: Conversation Analysis ──
    conversation_stats = analyze_conversations(df, config)

    # ── Step 3: Customer Language Analysis ──
    language_stats = analyze_customer_language(df, config)

    # ── Step 4: Intent Discovery ──
    intent_stats = discover_intents(df, config)

    # ── Step 5: Escalation Patterns ──
    escalation_stats = analyze_escalation_patterns(df, config)

    # ── Step 6: Generate Visualizations ──
    chart_paths = generate_all_charts(
        overview_stats=overview_stats,
        conversation_stats=conversation_stats,
        language_stats=language_stats,
        intent_stats=intent_stats,
        escalation_stats=escalation_stats,
        config=config,
    )

    # ── Step 7: Generate Report ──
    report_path = generate_report(
        overview_stats=overview_stats,
        conversation_stats=conversation_stats,
        language_stats=language_stats,
        intent_stats=intent_stats,
        escalation_stats=escalation_stats,
        chart_paths=chart_paths,
        config=config,
    )

    # ── Final Summary ──
    elapsed = time.time() - start_time
    log_section(logger, "EDA PIPELINE COMPLETE")
    log_metric(logger, "Total runtime", f"{elapsed:.1f} seconds")
    log_metric(logger, "Charts generated", f"{len(chart_paths)}")
    log_metric(logger, "Report", str(report_path))
    log_metric(logger, "Output directory", str(config.eda_output_dir))
    logger.info("")
    logger.info(f"  {Colors.GREEN}{Colors.BOLD}✓ EDA complete{Colors.RESET}")
    logger.info(f"  {Colors.DIM}Report:   {report_path}{Colors.RESET}")
    logger.info(f"  {Colors.DIM}Charts:   {config.eda_output_dir}{Colors.RESET}")
    logger.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hiver Support AI — EDA & Brand Intelligence Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline configuration YAML file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
