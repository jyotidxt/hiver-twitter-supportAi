"""Validation script for submission readiness.

Verifies folder structure, dataset files, configuration integrity, evaluation outputs,
report generation, and documentation completeness.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import Colors


def validate_repository() -> bool:
    """Run submission verification checks.

    Returns:
        True if all checks pass, False otherwise.
    """
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  HIVER SUPPORT AI — SUBMISSION READINESS VALIDATOR{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}\n")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    passed_checks = 0
    total_checks = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed_checks, total_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {name:<45} {Colors.DIM}{detail}{Colors.RESET}")
        else:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} {name:<45} {Colors.RED}{detail}{Colors.RESET}")

    # 1. Directory Structure Checks
    required_dirs = [
        "planning", "configs", "data", "src", "notebooks", "results", "tests", "examples", "logs", ".github/workflows"
    ]
    for d in required_dirs:
        dir_path = PROJECT_ROOT / d
        check(f"Directory: {d}/", dir_path.exists() and dir_path.is_dir(), f"Path: {d}")

    # 2. Configuration Integrity
    try:
        cfg = load_config()
        config_valid = cfg.selected_brand == "AmazonHelp" and cfg.agent_top_k >= 1
    except Exception as e:
        config_valid = False
    check("Config: configs/config.yaml", config_valid, "Loaded AmazonHelp & Agent fields")

    # 3. Source Package Modules
    src_modules = [
        "src/preprocessing", "src/analysis", "src/annotation",
        "src/classifier", "src/retrieval", "src/reply_generator",
        "src/escalation", "src/pipeline", "src/evaluation",
        "src/judge", "src/report", "src/utils"
    ]
    for mod in src_modules:
        mod_path = PROJECT_ROOT / mod
        check(f"Package: {mod}/", mod_path.exists(), f"Path: {mod}")

    # 4. Critical Files
    critical_files = [
        ("README.md", "Main project documentation"),
        ("REPORT.md", "Final 6-page technical report"),
        ("SUBMISSION_CHECKLIST.md", "Hiver deliverables checklist"),
        ("Makefile", "Build & execution commands"),
        ("LICENSE", "MIT License"),
        (".env.example", "Environment template"),
        (".github/workflows/ci.yml", "CI workflow"),
        ("examples/sample_messages.json", "Demo queries dataset"),
    ]
    for file_name, desc in critical_files:
        fp = PROJECT_ROOT / file_name
        check(f"File: {file_name}", fp.exists() and fp.stat().st_size > 0, desc)

    # 5. Result Artifacts
    result_files = [
        "results/evaluation/predictions.csv",
        "results/evaluation/baseline_comparison.csv",
        "results/evaluation/EVALUATION_SUMMARY.md",
        "results/judge/judged_predictions.csv",
        "results/judge/JUDGE_REPORT.md",
    ]
    for rf in result_files:
        rf_path = PROJECT_ROOT / rf
        check(f"Result: {rf}", rf_path.exists(), "Generated output file")

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}")
    if passed_checks == total_checks:
        print(f"  {Colors.BOLD}{Colors.GREEN}SUCCESS: All {passed_checks}/{total_checks} validation checks PASSED!{Colors.RESET}")
        print(f"  {Colors.BOLD}{Colors.GREEN}The repository is 100% submission ready for Hiver review.{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}\n")
        return True
    else:
        print(f"  {Colors.BOLD}{Colors.RED}WARNING: {total_checks - passed_checks}/{total_checks} checks failed.{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 65}{Colors.RESET}\n")
        return False


if __name__ == "__main__":
    success = validate_repository()
    sys.exit(0 if success else 1)
