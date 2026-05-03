"""
Parse JUnit XML results and print a weighted grade summary.

Phase weights (matches project rubric):
    Phase 1 (MLP Architecture + Training Strategy)  — 40 pts
    Phase 2 (Regularization)                        — 20 pts
    Phase 3 (Transfer Learning)                     — 25 pts
    Phase 4 (Model Analysis + Documentation)        — 15 pts
    Total                                           — 100 pts
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PHASE_WEIGHTS = {
    "phase1": 40,
    "phase2": 20,
    "phase3": 25,
    "phase4": 15,
}


def parse_junit(xml_path: str) -> tuple[int, int]:
    """Return (passed, total) test counts from a JUnit XML file."""
    path = Path(xml_path)
    if not path.exists():
        return 0, 0
    tree = ET.parse(path)
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return 0, 0
    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failures - errors - skipped
    return passed, total


def main():
    xml_files = sys.argv[1:]
    if not xml_files:
        print("Usage: grade_summary.py phase1.xml phase2.xml ...")
        sys.exit(1)

    total_score = 0.0
    max_score = 0.0

    print("\n" + "=" * 60)
    print("  AUTOGRADER — IMAGE CLASSIFICATION PROJECT")
    print("=" * 60)
    print(f"{'Phase':<30} {'Tests':>10} {'Score':>10} {'Max':>6}")
    print("-" * 60)

    for xml_file in xml_files:
        phase_key = Path(xml_file).stem          # e.g. "phase1"
        weight = PHASE_WEIGHTS.get(phase_key, 0)
        passed, total = parse_junit(xml_file)

        if total == 0:
            score = 0.0
        else:
            score = (passed / total) * weight

        total_score += score
        max_score += weight

        status = "✓" if passed == total else f"{passed}/{total}"
        print(f"  {phase_key:<28} {status:>10} {score:>9.1f} {weight:>6}")

    print("-" * 60)
    grade_pct = (total_score / max_score * 100) if max_score > 0 else 0
    print(f"  {'TOTAL':<28} {'':>10} {total_score:>9.1f} {int(max_score):>6}")
    print(f"\n  Final Grade: {grade_pct:.1f} / 100")
    print("=" * 60 + "\n")

    # Fail CI only if nothing at all passes
    if total_score == 0 and max_score > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
