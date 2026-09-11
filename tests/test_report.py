"""Tests for the Phase 6 Report Builder and Failure Analysis modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.report.assets import generate_mermaid_architecture
from src.report.builder import ReportBuilder, generate_final_report
from src.report.failure_analysis import FailureAnalysisEngine, FailureMode


class TestReportAssets:
    """Test suite for report assets generation."""

    def test_generate_mermaid_architecture(self) -> None:
        """Test generating Mermaid architecture diagram."""
        diagram = generate_mermaid_architecture()
        assert "flowchart TD" in diagram
        assert "Intent Classifier" in diagram
        assert "Semantic Evidence Retriever" in diagram
        assert "Escalation Decision Engine" in diagram


class TestFailureAnalysis:
    """Test suite for failure analysis engine."""

    def test_extract_top_5_failures(self) -> None:
        """Test extracting top 5 failure modes."""
        engine = FailureAnalysisEngine()
        failures = engine.extract_top_5_failures()

        assert len(failures) == 5
        for f in failures:
            assert isinstance(f, FailureMode)
            assert f.customer_message != ""
            assert f.predicted_output != ""
            assert f.expected_output != ""
            assert f.why_failed != ""
            assert f.hypothesis != ""


class TestReportBuilder:
    """Test suite for ReportBuilder."""

    def test_build_report(self) -> None:
        """Test building full REPORT.md content."""
        builder = ReportBuilder()
        content = builder.build_report()

        assert "# Engineering Technical Report" in content
        assert "## 1. Executive Summary" in content
        assert "## 6. Failure Analysis" in content
        assert "What is misleading about my headline number?" in content
        assert "## 8. Future Engineering Roadmap" in content

    def test_generate_final_report_creates_file(self) -> None:
        """Test running generate_final_report writes REPORT.md."""
        report_path = generate_final_report()
        assert report_path.exists()
        assert report_path.name == "REPORT.md"
