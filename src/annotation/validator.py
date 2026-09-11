"""Annotation validation module.

Validates annotation records before they are exported to the golden dataset.
Checks for data quality issues including duplicates, missing fields,
invalid labels, and logical consistency.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.schema import (
    GOLDEN_DATASET_COLUMNS,
    AnnotationRecord,
    Confidence,
    EscalationReason,
    Intent,
)
from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating the annotation dataset.

    Attributes:
        is_valid: Whether the dataset passed all checks.
        total_records: Total number of annotation records.
        valid_records: Number of records that passed validation.
        invalid_records: Number of records that failed validation.
        errors: List of (row_index, error_message) tuples.
        warnings: List of warning messages.
        duplicate_ids: List of conversation IDs that appear more than once.
        intent_distribution: Distribution of intent labels.
    """

    is_valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    errors: list[tuple[int, str]]
    warnings: list[str]
    duplicate_ids: list[str]
    intent_distribution: dict[str, int]


def validate_annotations(progress_path: Path) -> ValidationResult:
    """Validate the annotation progress file.

    Performs comprehensive checks on the annotation data including:
    - Duplicate conversation IDs
    - Missing required fields
    - Invalid intent names
    - Invalid confidence levels
    - Missing escalation reasons when escalation is True
    - Empty records

    Args:
        progress_path: Path to the annotation progress CSV file.

    Returns:
        A ValidationResult with all check outcomes.
    """
    log_section(logger, "VALIDATING ANNOTATIONS")

    if not progress_path.exists():
        logger.error(f"  Progress file not found: {progress_path}")
        return ValidationResult(
            is_valid=False,
            total_records=0,
            valid_records=0,
            invalid_records=0,
            errors=[(0, f"File not found: {progress_path}")],
            warnings=[],
            duplicate_ids=[],
            intent_distribution={},
        )

    df = pd.read_csv(progress_path, dtype={"conversation_id": str})
    total = len(df)
    logger.info(f"  Loaded {total:,} annotation records")

    errors: list[tuple[int, str]] = []
    warnings: list[str] = []

    # Check 1: Duplicate conversation IDs
    dupes = df[df["conversation_id"].duplicated(keep=False)]
    duplicate_ids = dupes["conversation_id"].unique().tolist() if len(dupes) > 0 else []
    if duplicate_ids:
        for dup_id in duplicate_ids:
            dup_rows = df[df["conversation_id"] == dup_id].index.tolist()
            errors.append(
                (dup_rows[0], f"Duplicate conversation_id '{dup_id}' found at rows {dup_rows}")
            )
        logger.warning(f"  Found {len(duplicate_ids)} duplicate conversation IDs")

    # Check 2: Validate each record
    valid_count = 0
    for idx, row in df.iterrows():
        record = AnnotationRecord(
            conversation_id=str(row.get("conversation_id", "")),
            primary_intent=str(row.get("primary_intent", "")),
            escalation=bool(row.get("escalation", False)),
            escalation_reason=str(row.get("escalation_reason", "none")),
            confidence=str(row.get("confidence", "")),
            annotator_notes=str(row.get("annotator_notes", "")),
            annotated_at=str(row.get("annotated_at", "")),
        )

        record_errors = record.validate()
        if record_errors:
            for err in record_errors:
                errors.append((int(idx), f"Row {idx}: {err}"))
        else:
            valid_count += 1

    # Check 3: Empty records
    empty_mask = (
        df["conversation_id"].isna() | (df["conversation_id"].astype(str).str.strip() == "")
    )
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        warnings.append(f"{empty_count} records have empty conversation_id")

    # Check 4: Intent distribution
    intent_dist = {}
    if "primary_intent" in df.columns:
        intent_dist = df["primary_intent"].value_counts().to_dict()

    # Check 5: Low confidence warnings
    if "confidence" in df.columns:
        low_conf = int((df["confidence"] == "low").sum())
        if low_conf > 0:
            warnings.append(
                f"{low_conf} records have low confidence \u2014 consider review"
            )

    # Check 6: Minimum dataset size
    if valid_count < 150:
        warnings.append(
            f"Only {valid_count} valid records. Target is 150-250 for the golden dataset."
        )

    is_valid = len(errors) == 0

    # Log results
    log_metric(logger, "Total records", f"{total:,}")
    log_metric(logger, "Valid records", f"{valid_count:,}")
    log_metric(logger, "Invalid records", f"{total - valid_count:,}")
    log_metric(logger, "Duplicates", f"{len(duplicate_ids):,}")
    log_metric(logger, "Validation", f"{'PASSED' if is_valid else 'FAILED'}")

    if intent_dist:
        logger.info("")
        logger.info("  Intent Distribution:")
        for intent, count in sorted(intent_dist.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            bar = "\u2588" * min(int(pct), 40)
            logger.info(f"    {intent:<25} {count:>4}  ({pct:5.1f}%)  {bar}")

    return ValidationResult(
        is_valid=is_valid,
        total_records=total,
        valid_records=valid_count,
        invalid_records=total - valid_count,
        errors=errors,
        warnings=warnings,
        duplicate_ids=duplicate_ids,
        intent_distribution=intent_dist,
    )


def generate_validation_report(
    result: ValidationResult, output_path: Path | None = None
) -> str:
    """Generate a human-readable validation report.

    Args:
        result: The validation result to report.
        output_path: Optional path to save the report as a text file.

    Returns:
        The report as a string.
    """
    lines = [
        "Annotation Validation Report",
        "=" * 40,
        "",
        f"Status:           {'PASSED' if result.is_valid else 'FAILED'}",
        f"Total Records:    {result.total_records}",
        f"Valid Records:    {result.valid_records}",
        f"Invalid Records:  {result.invalid_records}",
        f"Duplicate IDs:    {len(result.duplicate_ids)}",
        "",
    ]

    if result.errors:
        lines.append("ERRORS:")
        lines.append("-" * 40)
        for row_idx, msg in result.errors[:50]:
            lines.append(f"  [{row_idx}] {msg}")
        if len(result.errors) > 50:
            lines.append(f"  ... and {len(result.errors) - 50} more errors")
        lines.append("")

    if result.warnings:
        lines.append("WARNINGS:")
        lines.append("-" * 40)
        for warn in result.warnings:
            lines.append(f"  \u26a0 {warn}")
        lines.append("")

    if result.intent_distribution:
        lines.append("Intent Distribution:")
        lines.append("-" * 40)
        for intent, count in sorted(
            result.intent_distribution.items(), key=lambda x: -x[1]
        ):
            pct = (
                count / result.total_records * 100
                if result.total_records > 0
                else 0
            )
            lines.append(f"  {intent:<25} {count:>4} ({pct:5.1f}%)")
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"  Report saved: {output_path}")

    return report
