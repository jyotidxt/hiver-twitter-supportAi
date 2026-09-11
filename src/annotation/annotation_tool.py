"""Interactive terminal annotation tool for the Golden Dataset.

Provides a command-line interface for manually annotating customer
support conversations with intent labels and escalation decisions.

Usage:
    python -m src.annotation.annotation_tool
    python -m src.annotation.annotation_tool --limit 50
    python -m src.annotation.annotation_tool --resume
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.schema import (
    ESCALATION_REASON_DESCRIPTIONS,
    GOLDEN_DATASET_COLUMNS,
    INTENT_CATEGORIES,
    AnnotationRecord,
    Confidence,
    EscalationReason,
    Intent,
)
from src.utils.config import load_config
from src.utils.logger import Colors, get_logger, log_metric, log_section

logger = get_logger(__name__)

# Progress file path
PROGRESS_FILE = PROJECT_ROOT / "data" / "golden" / "annotation_progress.csv"


def load_conversations(config: object) -> pd.DataFrame:
    """Load annotation-ready conversations.

    Args:
        config: Pipeline configuration with path attributes.

    Returns:
        DataFrame of conversations ready for annotation.

    Raises:
        SystemExit: If the annotation-ready file doesn't exist.
    """
    path = config.annotation_ready_path
    if not path.exists():
        logger.error(f"Annotation-ready file not found: {path}")
        logger.error("Run the preprocessing pipeline first:")
        logger.error("  python -m src.preprocessing.run_pipeline")
        sys.exit(1)

    df = pd.read_csv(path, dtype={"conversation_id": str})
    logger.info(f"  Loaded {len(df):,} conversations from {path.name}")
    return df


def load_progress() -> set[str]:
    """Load previously annotated conversation IDs.

    Returns:
        Set of conversation IDs that have already been annotated.
    """
    if not PROGRESS_FILE.exists():
        return set()

    try:
        df = pd.read_csv(PROGRESS_FILE, dtype={"conversation_id": str})
        annotated = set(df["conversation_id"].dropna().tolist())
        logger.info(f"  Resuming: {len(annotated)} conversations already annotated")
        return annotated
    except Exception as e:
        logger.warning(f"  Could not load progress file: {e}")
        return set()


def save_annotation(record: AnnotationRecord) -> None:
    """Append a single annotation record to the progress file.

    Args:
        record: The validated annotation record to save.
    """
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = PROGRESS_FILE.exists()

    with open(PROGRESS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GOLDEN_DATASET_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record.to_dict())


def display_conversation(row: pd.Series, index: int, total: int) -> None:
    """Display a conversation for annotation.

    Args:
        row: A row from the annotation-ready DataFrame.
        index: Current conversation number (1-indexed).
        total: Total conversations to annotate.
    """
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}  Conversation {index}/{total}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print()

    # Conversation metadata
    conv_id = row.get("conversation_id", "N/A")
    num_msgs = row.get("num_messages", "?")
    num_cust = row.get("num_customer_messages", "?")
    num_brand = row.get("num_brand_replies", "?")

    print(f"  {Colors.DIM}ID:{Colors.RESET}       {conv_id}")
    print(f"  {Colors.DIM}Messages:{Colors.RESET} {num_msgs} total ({num_cust} customer, {num_brand} brand)")
    print()

    # Opening message
    opening = row.get("opening_message", "")
    if opening and str(opening) != "nan":
        print(f"  {Colors.BOLD}{Colors.YELLOW}Opening Message:{Colors.RESET}")
        print(f"  {opening}")
        print()

    # Full conversation
    full_conv = row.get("full_conversation", "")
    if full_conv and str(full_conv) != "nan":
        print(f"  {Colors.BOLD}Full Thread:{Colors.RESET}")
        print(f"  {'-' * 60}")
        for line in str(full_conv).split("\n"):
            line = line.strip()
            if line.startswith("[CUSTOMER]"):
                print(f"  {Colors.GREEN}{line}{Colors.RESET}")
            elif line.startswith("[BRAND]"):
                print(f"  {Colors.BLUE}{line}{Colors.RESET}")
            elif line:
                print(f"  {line}")
        print(f"  {'-' * 60}")
    print()


def display_intent_menu() -> None:
    """Display the intent selection menu."""
    print(f"  {Colors.BOLD}Select Primary Intent:{Colors.RESET}")
    print()
    idx = 1
    for category, intents in INTENT_CATEGORIES.items():
        print(f"  {Colors.CYAN}{category}:{Colors.RESET}")
        for intent_id, description in intents:
            print(f"    {Colors.BOLD}{idx:>2}{Colors.RESET}. {intent_id:<25} {Colors.DIM}{description}{Colors.RESET}")
            idx += 1
    print()


def get_intent_by_number(number: int) -> str | None:
    """Map a menu number to an intent ID.

    Args:
        number: The 1-indexed menu selection.

    Returns:
        The intent ID string, or None if invalid.
    """
    idx = 1
    for _category, intents in INTENT_CATEGORIES.items():
        for intent_id, _desc in intents:
            if idx == number:
                return intent_id
            idx += 1
    return None


def prompt_intent() -> str:
    """Prompt the annotator to select a primary intent.

    Returns:
        The selected intent ID string.
    """
    display_intent_menu()

    while True:
        choice = input(f"  Enter intent number (1-12) or name: ").strip()

        if not choice:
            print(f"  {Colors.RED}Please select an intent.{Colors.RESET}")
            continue

        # Try as number
        try:
            num = int(choice)
            intent = get_intent_by_number(num)
            if intent:
                print(f"  {Colors.GREEN}\u2713 {intent}{Colors.RESET}")
                return intent
            else:
                print(f"  {Colors.RED}Invalid number. Enter 1-12.{Colors.RESET}")
                continue
        except ValueError:
            pass

        # Try as intent name
        try:
            intent_enum = Intent.from_string(choice)
            print(f"  {Colors.GREEN}\u2713 {intent_enum.value}{Colors.RESET}")
            return intent_enum.value
        except ValueError:
            print(f"  {Colors.RED}Invalid intent. Try a number (1-12) or intent name.{Colors.RESET}")


def prompt_escalation() -> tuple[bool, str]:
    """Prompt the annotator for escalation decision.

    Returns:
        Tuple of (escalation_bool, escalation_reason_string).
    """
    while True:
        esc = input(f"  Requires escalation? (y/n): ").strip().lower()
        if esc in ("y", "yes", "1", "true"):
            # Get reason
            print()
            print(f"  {Colors.BOLD}Escalation Reason:{Colors.RESET}")
            reasons = [
                (k, v) for k, v in ESCALATION_REASON_DESCRIPTIONS.items()
            ]
            for i, (reason_id, desc) in enumerate(reasons, 1):
                print(f"    {Colors.BOLD}{i}{Colors.RESET}. {reason_id:<25} {Colors.DIM}{desc}{Colors.RESET}")
            print()

            while True:
                reason_choice = input(f"  Enter reason number (1-{len(reasons)}) or name: ").strip()
                if not reason_choice:
                    print(f"  {Colors.RED}Please select a reason.{Colors.RESET}")
                    continue

                try:
                    num = int(reason_choice)
                    if 1 <= num <= len(reasons):
                        reason = reasons[num - 1][0]
                        print(f"  {Colors.GREEN}\u2713 {reason}{Colors.RESET}")
                        return True, reason
                    else:
                        print(f"  {Colors.RED}Invalid number.{Colors.RESET}")
                        continue
                except ValueError:
                    pass

                try:
                    reason_enum = EscalationReason.from_string(reason_choice)
                    if reason_enum == EscalationReason.NONE:
                        print(f"  {Colors.RED}Please select a specific reason.{Colors.RESET}")
                        continue
                    print(f"  {Colors.GREEN}\u2713 {reason_enum.value}{Colors.RESET}")
                    return True, reason_enum.value
                except ValueError:
                    print(f"  {Colors.RED}Invalid reason.{Colors.RESET}")

        elif esc in ("n", "no", "0", "false"):
            return False, "none"
        else:
            print(f"  {Colors.RED}Enter 'y' or 'n'.{Colors.RESET}")


def prompt_confidence() -> str:
    """Prompt the annotator for confidence level.

    Returns:
        The confidence level string.
    """
    while True:
        conf = input(f"  Confidence (h=high, m=medium, l=low): ").strip().lower()
        mapping = {
            "h": "high", "high": "high",
            "m": "medium", "medium": "medium", "med": "medium",
            "l": "low", "low": "low",
        }
        if conf in mapping:
            result = mapping[conf]
            print(f"  {Colors.GREEN}\u2713 {result}{Colors.RESET}")
            return result
        print(f"  {Colors.RED}Enter 'h', 'm', or 'l'.{Colors.RESET}")


def prompt_notes() -> str:
    """Prompt for optional annotator notes.

    Returns:
        The notes string (may be empty).
    """
    notes = input(f"  Notes (optional, press Enter to skip): ").strip()
    return notes


def annotate_conversation(
    row: pd.Series, index: int, total: int
) -> AnnotationRecord | None:
    """Run the full annotation workflow for one conversation.

    Args:
        row: A row from the annotation-ready DataFrame.
        index: Current conversation number (1-indexed).
        total: Total conversations to annotate.

    Returns:
        An AnnotationRecord, or None if the user chose to skip/quit.
    """
    display_conversation(row, index, total)

    # Check for skip/quit
    print(f"  {Colors.DIM}Commands: 's' = skip, 'q' = quit{Colors.RESET}")
    print()

    # Intent
    intent = prompt_intent()
    if intent in ("q", "quit"):
        return None
    print()

    # Escalation
    escalation, reason = prompt_escalation()
    print()

    # Confidence
    confidence = prompt_confidence()
    print()

    # Notes
    notes = prompt_notes()
    print()

    record = AnnotationRecord(
        conversation_id=str(row["conversation_id"]),
        primary_intent=intent,
        escalation=escalation,
        escalation_reason=reason,
        confidence=confidence,
        annotator_notes=notes,
    )

    # Validate
    errors = record.validate()
    if errors:
        print(f"  {Colors.RED}Validation errors:{Colors.RESET}")
        for err in errors:
            print(f"    - {err}")
        print(f"  {Colors.YELLOW}Skipping this conversation.{Colors.RESET}")
        return None

    # Confirm
    print(f"  {Colors.BOLD}Summary:{Colors.RESET}")
    print(f"    Intent:     {record.primary_intent}")
    print(f"    Escalation: {'Yes - ' + record.escalation_reason if record.escalation else 'No'}")
    print(f"    Confidence: {record.confidence}")
    if record.annotator_notes:
        print(f"    Notes:      {record.annotator_notes}")
    print()

    confirm = input(f"  Save this annotation? (y/n): ").strip().lower()
    if confirm not in ("y", "yes", ""):
        print(f"  {Colors.YELLOW}Discarded.{Colors.RESET}")
        return None

    return record


def main(
    config_path: str | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> None:
    """Run the interactive annotation workflow.

    Args:
        config_path: Optional path to configuration file.
        limit: Maximum number of conversations to annotate.
        resume: Whether to skip already-annotated conversations.
    """
    config = load_config(config_path)

    # Banner
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  HIVER SUPPORT AI \u2014 GOLDEN DATASET ANNOTATION{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  Phase 3: Manual Annotation Tool{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print()
    print(f"  Brand:    {Colors.BOLD}{config.selected_brand}{Colors.RESET}")
    print(f"  Progress: {PROGRESS_FILE}")
    print()

    # Load data
    conversations = load_conversations(config)

    # Resume support
    annotated_ids = load_progress() if resume else set()
    remaining = conversations[
        ~conversations["conversation_id"].isin(annotated_ids)
    ].copy()

    if limit:
        remaining = remaining.head(limit)

    total = len(remaining)
    if total == 0:
        print(f"  {Colors.GREEN}All conversations have been annotated!{Colors.RESET}")
        print(f"  Total annotated: {len(annotated_ids):,}")
        print(f"  Run the exporter: python -m src.annotation.exporter")
        return

    print(f"  Conversations to annotate: {total}")
    print(f"  Already completed: {len(annotated_ids)}")
    print()
    print(f"  {Colors.DIM}Tip: You can quit anytime with 'q'. Progress is saved.{Colors.RESET}")
    print()

    input(f"  Press Enter to begin annotation...")

    # Annotation loop
    completed = 0
    for idx, (_, row) in enumerate(remaining.iterrows(), 1):
        try:
            record = annotate_conversation(row, idx, total)

            if record is None:
                # Check if quit
                quit_check = input(f"  Continue? (y=yes, q=quit): ").strip().lower()
                if quit_check in ("q", "quit", "n", "no"):
                    break
                continue

            save_annotation(record)
            completed += 1
            print(f"  {Colors.GREEN}\u2713 Saved ({completed + len(annotated_ids)} total){Colors.RESET}")

        except KeyboardInterrupt:
            print(f"\n\n  {Colors.YELLOW}Annotation interrupted. Progress saved.{Colors.RESET}")
            break
        except Exception as e:
            logger.error(f"Error annotating conversation: {e}")
            continue

    # Summary
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"  Session complete!")
    print(f"  Annotated this session: {completed}")
    print(f"  Total annotated: {completed + len(annotated_ids)}")
    print(f"  Progress file: {PROGRESS_FILE}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hiver Support AI \u2014 Golden Dataset Annotation Tool"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of conversations to annotate",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh without resuming previous progress",
    )
    args = parser.parse_args()
    main(config_path=args.config, limit=args.limit, resume=not args.no_resume)
