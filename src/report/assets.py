"""Report assets and diagram generation module.

Generates Mermaid architecture diagrams and visual markdown elements for REPORT.md.
"""

from __future__ import annotations

from pathlib import Path

REPORT_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "report"


def generate_mermaid_architecture() -> str:
    """Generate the Mermaid architecture diagram string.

    Returns:
        Mermaid flowchart syntax string.
    """
    return """```mermaid
flowchart TD
    IN["Customer Message (Tweet)"] --> CLF["Intent Classifier<br/>(TF-IDF + Logistic Regression)"]
    
    CLF -->|Predicted Intent & Confidence| RET["Semantic Evidence Retriever<br/>(SentenceEmbeddings / Cosine)"]
    CLF -->|Confidence & Intent| ESC["Escalation Decision Engine<br/>(Rule & Threshold Guardrails)"]
    
    RET -->|Top-K Historical Evidence| GEN["Grounded Reply Generator<br/>(OpenAI / Gemini / Grounded Template)"]
    RET -->|Retrieved Context| ESC
    
    GEN -->|Generated Response| OUT["Structured Pipeline Output (JSON)"]
    ESC -->|AUTO_HANDLE / ESCALATE| OUT
    
    OUT --> EVAL["Evaluation Harness & LLM Judge<br/>(Quantitative & Qualitative Benchmark)"]
```"""


def save_report_asset(filename: str, content: str) -> Path:
    """Save an asset string to results/report/.

    Args:
        filename: Target filename.
        content: Text content to save.

    Returns:
        Path to saved file.
    """
    REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = REPORT_ASSETS_DIR / filename
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    return target_path
