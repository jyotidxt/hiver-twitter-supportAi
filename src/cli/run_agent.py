"""Command Line Interface (CLI) runner for the Hiver AI Support Agent.

Provides interactive and batch modes for executing inference on customer queries.

Usage:
    # Interactive mode
    python -m src.cli.run_agent

    # Direct input mode
    python -m src.cli.run_agent --input "Where is my order #12345?"

    # Batch file mode
    python -m src.cli.run_agent --file examples/sample_messages.json

    # JSON output mode
    python -m src.cli.run_agent --input "Can I cancel my order?" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.inference import InferenceEngine, InferenceResult
from src.utils.config import load_config
from src.utils.logger import Colors, get_logger

logger = get_logger(__name__)


def format_pretty_terminal_output(result: InferenceResult) -> str:
    """Format an InferenceResult for rich terminal display.

    Args:
        result: The InferenceResult to format.

    Returns:
        Formatted multi-line terminal output string.
    """
    lines = []
    lines.append(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}")
    lines.append(f"{Colors.BOLD}{Colors.CYAN}  HIVER AI SUPPORT AGENT — INFERENCE RESULT{Colors.RESET}")
    lines.append(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}\n")

    lines.append(f"  {Colors.BOLD}Customer Message:{Colors.RESET}")
    lines.append(f"    \"{result.customer_message}\"\n")

    # Intent Classification
    lines.append(f"  {Colors.BOLD}1. Intent Classification:{Colors.RESET}")
    lines.append(f"     Predicted Intent: {Colors.BOLD}{Colors.GREEN}{result.predicted_intent}{Colors.RESET}")
    lines.append(f"     Confidence:       {result.confidence:.2%}")
    lines.append(f"     Top Candidates:   {', '.join([f'{c[0]} ({c[1]:.2f})' for c in result.top_3_candidates])}\n")

    # Escalation Decision
    lines.append(f"  {Colors.BOLD}2. Escalation Decision:{Colors.RESET}")
    if result.escalation_decision == "AUTO_HANDLE":
        lines.append(f"     Status: {Colors.BOLD}{Colors.GREEN}✓ AUTO_HANDLE{Colors.RESET}")
    else:
        lines.append(f"     Status: {Colors.BOLD}{Colors.RED}⚡ ESCALATE{Colors.RESET}")
    lines.append(f"     Reason: {result.escalation_reason}\n")

    # Grounded Reply
    lines.append(f"  {Colors.BOLD}3. Generated Support Reply:{Colors.RESET}")
    lines.append(f"     \"{result.generated_reply}\"\n")

    # Retrieved Evidence Summary
    lines.append(f"  {Colors.BOLD}4. Historical Evidence ({len(result.retrieved_examples)} retrieved):{Colors.RESET}")
    for idx, item in enumerate(result.retrieved_examples, 1):
        score = item.get("similarity_score", 0.0)
        cust = item.get("customer_text", "")
        reply = item.get("brand_reply", "")
        lines.append(f"     Ex {idx} (Sim: {score:.2f}): \"{cust[:60]}...\" -> \"{reply[:60]}...\"")

    lines.append(f"\n  {Colors.DIM}Latency: {result.processing_time_ms:.1f} ms{Colors.RESET}")
    lines.append(f"{Colors.CYAN}{'=' * 65}{Colors.RESET}\n")

    return "\n".join(lines)


def run_cli() -> None:
    """Parse CLI arguments and run the inference pipeline."""
    parser = argparse.ArgumentParser(
        description="Hiver AI Support Agent — Runnable CLI Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python -m src.cli.run_agent\n"
               "  python -m src.cli.run_agent --input \"Where is my order #12345?\"\n"
               "  python -m src.cli.run_agent --file examples/sample_messages.json\n",
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Single customer message to analyze",
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to JSON file containing sample customer messages",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Output raw JSON instead of pretty terminal formatting",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to custom pipeline_config YAML file",
    )

    args = parser.parse_args()

    # Load engine
    cfg = load_config(args.config)
    engine = InferenceEngine(cfg)
    engine.load()

    # Direct input mode
    if args.input:
        result = engine.run(args.input)
        if args.json:
            print(result.to_json())
        else:
            print(format_pretty_terminal_output(result))
        return

    # File batch mode
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"{Colors.RED}File not found: {file_path}{Colors.RESET}")
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        messages = data if isinstance(data, list) else data.get("messages", [])
        results = []

        print(f"\n{Colors.CYAN}Processing {len(messages)} sample messages from {file_path.name}...{Colors.RESET}\n")

        for msg_item in messages:
            msg_text = msg_item if isinstance(msg_item, str) else msg_item.get("text", "")
            res = engine.run(msg_text)
            results.append(res.to_dict())
            if not args.json:
                print(format_pretty_terminal_output(res))

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # Interactive mode
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  HIVER AI SUPPORT AGENT — INTERACTIVE CLI DEMO{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  Brand: {cfg.selected_brand} ({cfg.brand_handle}){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}")
    print(f"  {Colors.DIM}Type your question, or 'exit' / 'q' to quit.{Colors.RESET}\n")

    while True:
        try:
            user_input = input(f"{Colors.BOLD}Enter customer message > {Colors.RESET}").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print(f"\n{Colors.CYAN}Exiting AI Agent CLI. Goodbye!{Colors.RESET}\n")
                break

            res = engine.run(user_input)
            if args.json:
                print(res.to_json())
            else:
                print(format_pretty_terminal_output(res))

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{Colors.CYAN}Exiting AI Agent CLI. Goodbye!{Colors.RESET}\n")
            break


if __name__ == "__main__":
    run_cli()
