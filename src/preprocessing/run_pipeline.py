"""Main preprocessing pipeline orchestrator.

Runs the complete data preprocessing pipeline from raw data to
exported datasets. This is the primary entry point for Phase 2.

Usage:
    python -m src.preprocessing.run_pipeline
    python -m src.preprocessing.run_pipeline --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.brand_filter import filter_brand
from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.exporter import export_datasets
from src.preprocessing.quality_filter import apply_quality_filters
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.thread_builder import build_threads
from src.utils.config import load_config
from src.utils.logger import Colors, get_logger, log_metric, log_section


def main(config_path: str | None = None) -> None:
    """Run the complete preprocessing pipeline.

    Args:
        config_path: Optional path to a custom configuration file.
    """
    start_time = time.time()

    # Load configuration
    config = load_config(config_path)
    logger = get_logger(
        "hiver-pipeline",
        level=config.log_level,
        log_file=config.log_file,
    )

    # Banner
    logger.info("")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}  HIVER SUPPORT AI - DATA PREPROCESSING PIPELINE{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}  Phase 2: Data Engineering{Colors.RESET}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    logger.info("")
    logger.info(f"  Brand: {Colors.BOLD}{config.selected_brand}{Colors.RESET}")
    logger.info(f"  Config: {config.raw.get('_source', 'config.yaml')}")
    logger.info("")

    # Step 1: Load raw data
    raw_df = load_raw_data(config)
    raw_count = len(raw_df)

    # Step 2: Filter for selected brand
    brand_df = filter_brand(raw_df, config)

    # Free memory
    del raw_df

    # Step 3: Reconstruct conversation threads
    threaded_df = build_threads(brand_df, config)

    # Free memory
    del brand_df

    # Step 4: Clean text
    cleaned_df = clean_text(threaded_df, config)

    # Free memory
    del threaded_df

    # Step 5: Apply quality filters
    final_df, removal_counts = apply_quality_filters(cleaned_df, config)

    # Free memory
    del cleaned_df

    # Step 6: Export datasets
    export_datasets(final_df, config, removal_counts, raw_count)

    # Final summary
    elapsed = time.time() - start_time
    log_section(logger, "PIPELINE COMPLETE")
    log_metric(logger, "Total runtime", f"{elapsed:.1f} seconds")
    log_metric(logger, "Final dataset size", f"{len(final_df):,} messages")
    log_metric(
        logger,
        "Final conversations",
        f"{final_df['conversation_id'].nunique():,}"
        if "conversation_id" in final_df.columns
        else "N/A",
    )
    logger.info("")
    logger.info(f"  {Colors.GREEN}{Colors.BOLD}✓ All exports saved successfully{Colors.RESET}")
    logger.info(f"  {Colors.DIM}Processed conversations: {config.processed_conversations_path}{Colors.RESET}")
    logger.info(f"  {Colors.DIM}Annotation-ready:       {config.annotation_ready_path}{Colors.RESET}")
    logger.info(f"  {Colors.DIM}Summary:                {config.preprocessing_summary_path}{Colors.RESET}")
    logger.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hiver Support AI - Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline configuration YAML file",
    )
    args = parser.parse_args()
    main(config_path=args.config)
