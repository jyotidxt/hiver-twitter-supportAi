"""Tests for the Golden Dataset annotation system.

Covers schema validation, duplicate detection, export correctness,
and resume functionality.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.schema import (
    GOLDEN_DATASET_COLUMNS,
    AnnotationRecord,
    Confidence,
    EscalationReason,
    Intent,
)


# ══════════════════════════════════════════════════
#  Schema Tests
# ══════════════════════════════════════════════════


class TestIntent:
    """Tests for the Intent enum."""

    def test_all_intents_defined(self) -> None:
        """Test that all 12 intents are defined."""
        assert len(Intent.values()) == 12

    def test_from_string_valid(self) -> None:
        """Test parsing valid intent strings."""
        assert Intent.from_string("order_status") == Intent.ORDER_STATUS
        assert Intent.from_string("REFUND_REQUEST") == Intent.REFUND_REQUEST
        assert Intent.from_string("billing-issue") == Intent.BILLING_ISSUE

    def test_from_string_invalid(self) -> None:
        """Test that invalid intent strings raise ValueError."""
        with pytest.raises(ValueError):
            Intent.from_string("nonexistent_intent")

    def test_values_returns_strings(self) -> None:
        """Test that values() returns strings, not enum members."""
        values = Intent.values()
        assert all(isinstance(v, str) for v in values)
        assert "order_status" in values


class TestConfidence:
    """Tests for the Confidence enum."""

    def test_all_levels_defined(self) -> None:
        """Test that all 3 confidence levels are defined."""
        assert len(Confidence.values()) == 3

    def test_from_string(self) -> None:
        """Test parsing confidence strings."""
        assert Confidence.from_string("high") == Confidence.HIGH
        assert Confidence.from_string("LOW") == Confidence.LOW

    def test_from_string_invalid(self) -> None:
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError):
            Confidence.from_string("very_high")


class TestEscalationReason:
    """Tests for the EscalationReason enum."""

    def test_from_string_empty(self) -> None:
        """Test that empty strings map to NONE."""
        assert EscalationReason.from_string("") == EscalationReason.NONE
        assert EscalationReason.from_string("n/a") == EscalationReason.NONE

    def test_from_string_valid(self) -> None:
        """Test parsing valid reason strings."""
        assert (
            EscalationReason.from_string("refund_payment")
            == EscalationReason.REFUND_PAYMENT
        )


# ══════════════════════════════════════════════════
#  Annotation Record Tests
# ══════════════════════════════════════════════════


class TestAnnotationRecord:
    """Tests for the AnnotationRecord dataclass."""

    def test_valid_record(self) -> None:
        """Test that a valid record passes validation."""
        record = AnnotationRecord(
            conversation_id="conv_001",
            primary_intent="order_status",
            escalation=False,
            escalation_reason="none",
            confidence="high",
        )
        errors = record.validate()
        assert errors == []

    def test_valid_record_with_escalation(self) -> None:
        """Test that a valid escalated record passes validation."""
        record = AnnotationRecord(
            conversation_id="conv_002",
            primary_intent="billing_issue",
            escalation=True,
            escalation_reason="refund_payment",
            confidence="medium",
        )
        errors = record.validate()
        assert errors == []

    def test_empty_conversation_id(self) -> None:
        """Test that empty conversation_id fails validation."""
        record = AnnotationRecord(
            conversation_id="",
            primary_intent="order_status",
            escalation=False,
        )
        errors = record.validate()
        assert any("conversation_id" in e for e in errors)

    def test_invalid_intent(self) -> None:
        """Test that invalid intent fails validation."""
        record = AnnotationRecord(
            conversation_id="conv_003",
            primary_intent="fake_intent",
            escalation=False,
        )
        errors = record.validate()
        assert any("intent" in e.lower() for e in errors)

    def test_empty_intent(self) -> None:
        """Test that empty intent fails validation."""
        record = AnnotationRecord(
            conversation_id="conv_004",
            primary_intent="",
            escalation=False,
        )
        errors = record.validate()
        assert any("intent" in e.lower() for e in errors)

    def test_invalid_confidence(self) -> None:
        """Test that invalid confidence fails validation."""
        record = AnnotationRecord(
            conversation_id="conv_005",
            primary_intent="order_status",
            escalation=False,
            confidence="very_high",
        )
        errors = record.validate()
        assert any("confidence" in e.lower() for e in errors)

    def test_escalation_without_reason(self) -> None:
        """Test that escalation=True without reason fails."""
        record = AnnotationRecord(
            conversation_id="conv_006",
            primary_intent="billing_issue",
            escalation=True,
            escalation_reason="none",
            confidence="high",
        )
        errors = record.validate()
        assert any("escalation_reason" in e for e in errors)

    def test_escalation_with_empty_reason(self) -> None:
        """Test that escalation=True with empty reason fails."""
        record = AnnotationRecord(
            conversation_id="conv_007",
            primary_intent="account_access",
            escalation=True,
            escalation_reason="",
            confidence="high",
        )
        errors = record.validate()
        assert any("escalation_reason" in e for e in errors)

    def test_no_escalation_reason_not_required(self) -> None:
        """Test that reason is not required when escalation is False."""
        record = AnnotationRecord(
            conversation_id="conv_008",
            primary_intent="general_inquiry",
            escalation=False,
            escalation_reason="none",
            confidence="high",
        )
        errors = record.validate()
        assert errors == []

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        record = AnnotationRecord(
            conversation_id="conv_009",
            primary_intent="delivery_problem",
            escalation=True,
            escalation_reason="damaged_defective",
            confidence="low",
            annotator_notes="Package was crushed",
        )
        d = record.to_dict()
        assert d["conversation_id"] == "conv_009"
        assert d["primary_intent"] == "delivery_problem"
        assert d["escalation"] is True
        assert d["escalation_reason"] == "damaged_defective"
        assert d["confidence"] == "low"
        assert d["annotator_notes"] == "Package was crushed"
        assert "annotated_at" in d

    def test_golden_columns_complete(self) -> None:
        """Test that GOLDEN_DATASET_COLUMNS matches record fields."""
        record = AnnotationRecord(
            conversation_id="test",
            primary_intent="order_status",
            escalation=False,
        )
        d = record.to_dict()
        for col in GOLDEN_DATASET_COLUMNS:
            assert col in d, f"Missing column: {col}"


# ══════════════════════════════════════════════════
#  Validator Tests
# ══════════════════════════════════════════════════


class TestValidator:
    """Tests for the annotation validator."""

    def test_validate_valid_file(self, tmp_path: Path) -> None:
        """Test validation of a correct progress file."""
        from src.annotation.validator import validate_annotations

        progress = tmp_path / "progress.csv"
        df = pd.DataFrame([
            {
                "conversation_id": "conv_001",
                "primary_intent": "order_status",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01T00:00:00",
            },
            {
                "conversation_id": "conv_002",
                "primary_intent": "refund_request",
                "escalation": True,
                "escalation_reason": "refund_payment",
                "confidence": "medium",
                "annotator_notes": "Customer upset",
                "annotated_at": "2024-01-01T00:01:00",
            },
        ])
        df.to_csv(progress, index=False)

        result = validate_annotations(progress)
        assert result.total_records == 2
        assert result.valid_records == 2
        assert result.invalid_records == 0
        assert result.duplicate_ids == []

    def test_detect_duplicates(self, tmp_path: Path) -> None:
        """Test detection of duplicate conversation IDs."""
        from src.annotation.validator import validate_annotations

        progress = tmp_path / "progress.csv"
        df = pd.DataFrame([
            {
                "conversation_id": "conv_001",
                "primary_intent": "order_status",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01T00:00:00",
            },
            {
                "conversation_id": "conv_001",
                "primary_intent": "billing_issue",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01T00:02:00",
            },
        ])
        df.to_csv(progress, index=False)

        result = validate_annotations(progress)
        assert "conv_001" in result.duplicate_ids
        assert len(result.errors) > 0

    def test_detect_invalid_intent(self, tmp_path: Path) -> None:
        """Test detection of invalid intent labels."""
        from src.annotation.validator import validate_annotations

        progress = tmp_path / "progress.csv"
        df = pd.DataFrame([
            {
                "conversation_id": "conv_001",
                "primary_intent": "invalid_intent",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01",
            },
        ])
        df.to_csv(progress, index=False)

        result = validate_annotations(progress)
        assert result.invalid_records > 0

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test validation of a nonexistent file."""
        from src.annotation.validator import validate_annotations

        result = validate_annotations(tmp_path / "nonexistent.csv")
        assert result.is_valid is False
        assert result.total_records == 0


# ══════════════════════════════════════════════════
#  Exporter Tests
# ══════════════════════════════════════════════════


class TestExporter:
    """Tests for the golden dataset exporter."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """Test that export creates the output CSV."""
        from src.annotation.exporter import export_golden_dataset

        progress = tmp_path / "progress.csv"
        output = tmp_path / "golden.csv"
        report = tmp_path / "report.txt"

        df = pd.DataFrame([
            {
                "conversation_id": "conv_001",
                "primary_intent": "order_status",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01T00:00:00",
            },
            {
                "conversation_id": "conv_002",
                "primary_intent": "refund_request",
                "escalation": True,
                "escalation_reason": "refund_payment",
                "confidence": "medium",
                "annotator_notes": "",
                "annotated_at": "2024-01-01T00:01:00",
            },
        ])
        df.to_csv(progress, index=False)

        result_path = export_golden_dataset(
            progress_path=progress,
            output_path=output,
            report_path=report,
        )

        assert result_path is not None
        assert output.exists()
        exported = pd.read_csv(output)
        assert len(exported) == 2

    def test_export_removes_duplicates(self, tmp_path: Path) -> None:
        """Test that export deduplicates by conversation_id."""
        from src.annotation.exporter import export_golden_dataset

        progress = tmp_path / "progress.csv"
        output = tmp_path / "golden.csv"
        report = tmp_path / "report.txt"

        df = pd.DataFrame([
            {
                "conversation_id": "conv_001",
                "primary_intent": "order_status",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "first attempt",
                "annotated_at": "2024-01-01T00:00:00",
            },
            {
                "conversation_id": "conv_001",
                "primary_intent": "billing_issue",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "corrected",
                "annotated_at": "2024-01-01T00:05:00",
            },
            {
                "conversation_id": "conv_002",
                "primary_intent": "delivery_problem",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01T00:02:00",
            },
        ])
        df.to_csv(progress, index=False)

        export_golden_dataset(
            progress_path=progress,
            output_path=output,
            report_path=report,
        )

        exported = pd.read_csv(output)
        assert len(exported) == 2
        # Should keep the last (corrected) annotation
        conv001 = exported[exported["conversation_id"] == "conv_001"]
        assert conv001.iloc[0]["primary_intent"] == "billing_issue"

    def test_export_deterministic_order(self, tmp_path: Path) -> None:
        """Test that export is sorted by conversation_id."""
        from src.annotation.exporter import export_golden_dataset

        progress = tmp_path / "progress.csv"
        output = tmp_path / "golden.csv"
        report = tmp_path / "report.txt"

        df = pd.DataFrame([
            {
                "conversation_id": "conv_003",
                "primary_intent": "order_status",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01",
            },
            {
                "conversation_id": "conv_001",
                "primary_intent": "billing_issue",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01",
            },
        ])
        df.to_csv(progress, index=False)

        export_golden_dataset(
            progress_path=progress,
            output_path=output,
            report_path=report,
        )

        exported = pd.read_csv(output)
        ids = exported["conversation_id"].tolist()
        assert ids == sorted(ids)

    def test_validate_only(self, tmp_path: Path) -> None:
        """Test validate-only mode doesn't create output."""
        from src.annotation.exporter import export_golden_dataset

        progress = tmp_path / "progress.csv"
        output = tmp_path / "golden.csv"
        report = tmp_path / "report.txt"

        df = pd.DataFrame([
            {
                "conversation_id": "conv_001",
                "primary_intent": "order_status",
                "escalation": False,
                "escalation_reason": "none",
                "confidence": "high",
                "annotator_notes": "",
                "annotated_at": "2024-01-01",
            },
        ])
        df.to_csv(progress, index=False)

        result = export_golden_dataset(
            progress_path=progress,
            output_path=output,
            report_path=report,
            validate_only=True,
        )

        assert result is None
        assert not output.exists()


# ══════════════════════════════════════════════════
#  Resume Functionality Tests
# ══════════════════════════════════════════════════


class TestResume:
    """Tests for resume functionality."""

    def test_load_progress_empty(self, tmp_path: Path) -> None:
        """Test loading progress when file doesn't exist."""
        from src.annotation.annotation_tool import load_progress, PROGRESS_FILE
        import src.annotation.annotation_tool as tool_module

        # Temporarily override PROGRESS_FILE
        original = tool_module.PROGRESS_FILE
        tool_module.PROGRESS_FILE = tmp_path / "nonexistent.csv"

        result = load_progress()
        assert result == set()

        tool_module.PROGRESS_FILE = original

    def test_load_progress_existing(self, tmp_path: Path) -> None:
        """Test loading progress from existing file."""
        from src.annotation.annotation_tool import load_progress
        import src.annotation.annotation_tool as tool_module

        progress = tmp_path / "progress.csv"
        df = pd.DataFrame([
            {"conversation_id": "conv_001", "primary_intent": "order_status",
             "escalation": False, "escalation_reason": "none",
             "confidence": "high", "annotator_notes": "",
             "annotated_at": "2024-01-01"},
            {"conversation_id": "conv_002", "primary_intent": "billing_issue",
             "escalation": False, "escalation_reason": "none",
             "confidence": "high", "annotator_notes": "",
             "annotated_at": "2024-01-01"},
        ])
        df.to_csv(progress, index=False)

        original = tool_module.PROGRESS_FILE
        tool_module.PROGRESS_FILE = progress

        result = load_progress()
        assert "conv_001" in result
        assert "conv_002" in result
        assert len(result) == 2

        tool_module.PROGRESS_FILE = original

    def test_save_annotation_creates_file(self, tmp_path: Path) -> None:
        """Test that saving creates the progress file."""
        from src.annotation.annotation_tool import save_annotation
        import src.annotation.annotation_tool as tool_module

        progress = tmp_path / "golden" / "progress.csv"
        original = tool_module.PROGRESS_FILE
        tool_module.PROGRESS_FILE = progress

        record = AnnotationRecord(
            conversation_id="conv_001",
            primary_intent="order_status",
            escalation=False,
            escalation_reason="none",
            confidence="high",
        )
        save_annotation(record)

        assert progress.exists()
        df = pd.read_csv(progress)
        assert len(df) == 1
        assert df.iloc[0]["conversation_id"] == "conv_001"

        tool_module.PROGRESS_FILE = original

    def test_save_annotation_appends(self, tmp_path: Path) -> None:
        """Test that saving appends to existing file."""
        from src.annotation.annotation_tool import save_annotation
        import src.annotation.annotation_tool as tool_module

        progress = tmp_path / "golden" / "progress.csv"
        original = tool_module.PROGRESS_FILE
        tool_module.PROGRESS_FILE = progress

        for i in range(3):
            record = AnnotationRecord(
                conversation_id=f"conv_{i:03d}",
                primary_intent="order_status",
                escalation=False,
                escalation_reason="none",
                confidence="high",
            )
            save_annotation(record)

        df = pd.read_csv(progress)
        assert len(df) == 3

        tool_module.PROGRESS_FILE = original
