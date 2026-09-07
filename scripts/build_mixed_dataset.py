#!/usr/bin/env python3
"""Combine the legacy and direct-pair pilots while preserving case-level splits."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "processed"
    train = (
        read_jsonl(processed / "medsp_intake_train.jsonl")
        + read_jsonl(processed / "medsp_direct_pairs_v1_train.jsonl")
        + read_jsonl(processed / "medsp_expanded_v1_train.jsonl")
        + read_jsonl(processed / "medsp_pediatric_expansion_v1_train.jsonl")
    )
    evaluation = (
        read_jsonl(processed / "medsp_intake_eval.jsonl")
        + read_jsonl(processed / "medsp_direct_pairs_v1_eval.jsonl")
        + read_jsonl(processed / "medsp_expanded_v1_eval.jsonl")
    )
    train_cases = {row["source_case_id"] for row in train}
    eval_cases = {row["source_case_id"] for row in evaluation}
    assert not train_cases & eval_cases, "source case leaked across splits"
    assert len({row["id"] for row in train}) == len(train), "duplicate training IDs"
    assert len({row["id"] for row in evaluation}) == len(evaluation), "duplicate evaluation IDs"
    write_jsonl(processed / "medsp_mixed_v1_train.jsonl", train)
    write_jsonl(processed / "medsp_mixed_v1_eval.jsonl", evaluation)
    (processed / "medsp_mixed_v1_manifest.json").write_text(
        json.dumps(
            {
                "method": "legacy source-grounded conversations, verbatim direct evaluator pairs, and expanded source-grounded cases",
                "train_count": len(train),
                "eval_count": len(evaluation),
                "train_case_ids": sorted(train_cases),
                "eval_case_ids": sorted(eval_cases),
                "case_overlap": sorted(train_cases & eval_cases),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(train)} mixed train and {len(evaluation)} mixed eval records")


if __name__ == "__main__":
    main()
