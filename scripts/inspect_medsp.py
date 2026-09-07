#!/usr/bin/env python3
"""Inventory MedSP1000 cases and print role-specific source files for review."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


ROLES = ("environment_controller", "evaluator", "examinee", "sp_actor")


def case_directories(root: Path) -> list[Path]:
    cases = [path for path in root.glob("*/scenario*") if path.is_dir()]
    return sorted(cases, key=lambda path: path.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "medsp1000",
    )
    parser.add_argument("--case", help="Relative case path, e.g. mededportal_10009/scenario1")
    parser.add_argument("--preview-chars", type=int, default=1200)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {root}")
    cases = case_directories(root)
    if args.case:
        target = (root / args.case).resolve()
        if target not in cases:
            raise SystemExit(f"Case not found: {args.case}")
        cases = [target]

    print(f"cases: {len(cases)}")
    for role in ROLES:
        print(f"{role}: {sum((case / role).is_dir() for case in cases)} cases")

    for case in cases:
        print(f"\n## {case.relative_to(root)}")
        for role in ROLES:
            files = sorted((case / role).rglob("*.md")) if (case / role).is_dir() else []
            print(f"\n### {role}: {len(files)} markdown files")
            for file in files:
                print(f"- {file.relative_to(case)}")
            if files:
                text = files[0].read_text(encoding="utf-8", errors="replace")
                print(f"\n{files[0].name} preview:\n{text[:args.preview_chars].rstrip()}")


if __name__ == "__main__":
    main()
