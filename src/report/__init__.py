"""Report generation package for Hiver Support AI."""

from src.report.assets import generate_mermaid_architecture
from src.report.builder import ReportBuilder, generate_final_report
from src.report.failure_analysis import FailureAnalysisEngine, FailureMode

__all__ = [
    "generate_mermaid_architecture",
    "FailureAnalysisEngine",
    "FailureMode",
    "ReportBuilder",
    "generate_final_report",
]
