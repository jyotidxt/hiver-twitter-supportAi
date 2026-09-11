"""Golden dataset exporter module.

Exports validated annotation records to the final golden_dataset.csv.
Only validated, deduplicated records are included in the export.

Usage:
    python -m src.annotation.exporter
    python -m src.annotation.exporter --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.schema import (
    GOLDEN_DATASET_COLUMNS,
    Intent,
)
from src.annotation.validator import (
    generate_validation_report,
    validate_annotations,
)
from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)

PROGRESS_FILE = PROJECT_ROOT / "data" / "golden" / "annotation_progress.csv"
GOLDEN_OUTPUT = PROJECT_ROOT / "data" / "golden" / "golden_dataset.csv"
REPORT_OUTPUT = PROJECT_ROOT / "data" / "golden" / "validation_report.txt"


def export_golden_dataset(
    progress_path: Path = PROGRESS_FILE,
    output_path: Path = GOLDEN_OUTPUT,
    report_path: Path = REPORT_OUTPUT,
    validate_only: bool = False,
) -> Path | None:
    """Export the validated golden dataset.

    Reads the annotation progress file, validates all records,
    removes duplicates (keeping the latest annotation), and
    exports the clean dataset.

    Args:
        progress_path: Path to the annotation progress CSV.
        output_path: Path for the golden dataset output.
        report_path: Path for the validation report.
        validate_only: If True, only validate without exporting.

    Returns:
        Path to the exported file, or None if validation failed.
    """
    log_section(logger, "GOLDEN DATASET EXPORT")

    # Step 1: Validate
    result = validate_annotations(progress_path)
    report = generate_validation_report(result, report_path)

    if result.warnings:
        for warn in result.warnings:
            logger.warning(f"  \u26a0 {warn}")

    if validate_only:
        print()
        print(report)
        return None

    # Step 2: Load and clean
    df = pd.read_csv(progress_path, dtype={"conversation_id": str})

    # Remove duplicates \u2014 keep the last annotation for each conversation
    initial_count = len(df)
    df = df.drop_duplicates(subset=["conversation_id"], keep="last")
    deduped_count = initial_count - len(df)
    if deduped_count > 0:
        logger.info(f"  Removed {deduped_count} duplicate entries (kept latest)")

    # Remove records with invalid intents
    valid_intents = Intent.values()
    invalid_mask = ~df["primary_intent"].isin(valid_intents)
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        logger.warning(f"  Removed {invalid_count} records with invalid intents")
        df = df[~invalid_mask]

    # Remove records with empty conversation_id
    empty_mask = (
        df["conversation_id"].isna()
        | (df["conversation_id"].astype(str).str.strip() == "")
    )
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        logger.warning(f"  Removed {empty_count} records with empty conversation_id")
        df = df[~empty_mask]

    # Step 3: Sort deterministically
    df = df.sort_values("conversation_id").reset_index(drop=True)

    # Step 4: Select and order columns
    export_cols = [
        col for col in GOLDEN_DATASET_COLUMNS if col in df.columns
    ]
    df = df[export_cols]

    # Step 5: Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    log_section(logger, "EXPORT COMPLETE")
    log_metric(logger, "Exported records", f"{len(df):,}")
    log_metric(logger, "Output file", str(output_path))
    log_metric(logger, "Validation report", str(report_path))

    # Intent distribution summary
    logger.info("")
    logger.info("  Final Intent Distribution:")
    for intent, count in df["primary_intent"].value_counts().items():
        pct = count / len(df) * 100
        logger.info(f"    {intent:<25} {count:>4} ({pct:5.1f}%)")

    logger.info("")
    logger.info(f"  {Colors.GREEN}{Colors.BOLD}\u2713 Golden dataset exported successfully{Colors.RESET}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hiver Support AI \u2014 Golden Dataset Exporter"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate annotations without exporting",
    )
    parser.add_argument(
        "--progress", type=str, default=None,
        help="Path to annotation progress file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for golden dataset output",
    )
    args = parser.parse_args()

    progress = Path(args.progress) if args.progress else PROGRESS_FILE
    output = Path(args.output) if args.output else GOLDEN_OUTPUT

    export_golden_dataset(
        progress_path=progress,
        output_path=output,
        validate_only=args.validate_only,
    )
