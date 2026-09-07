#!/usr/bin/env python3
"""Build a small, verbatim MedSP question/answer dataset where checklists allow it.

This is intentionally separate from ``build_training_data.py``.  It uses only
explicit ``response -> Student: question`` pairs from evaluator checklists and
records source line numbers for every training turn.
"""

from __future__ import annotations

import json
from pathlib import Path


SYSTEM = (
    "Conduct a structured clinical intake interview. Ask one concise, relevant "
    "question at a time. Gather relevant history before summarizing. Do not diagnose "
    "or prescribe treatment."
)
CASES = [
    {
        "case_id": "mededportal_10373/scenario1",
        "split": "train",
        "opening": "I can't seem to get rid of this cough.",
        "source": "mededportal_10373/scenario1/evaluator/B. Cough Case Standardized Patient Master Encounter Checklist.md",
    },
    {
        "case_id": "mededportal_10373/scenario2",
        "split": "eval",
        "opening": "My back hurts.",
        "source": "mededportal_10373/scenario2/evaluator/G. Back-Pain Case Standardized Patient Master Encounter Checklist.md",
    },
]


def clean_question(line: str) -> str:
    question = line.strip()
    assert question.startswith("*Student:") and question.endswith("*"), question
    return question.removeprefix("*Student:").removesuffix("*").strip()


def extract_pairs(path: Path) -> list[dict[str, object]]:
    """Extract adjacent one-line response / student-question checklist pairs."""
    lines = path.read_text(encoding="utf-8").splitlines()
    pairs = []
    for response_index, line in enumerate(lines):
        if not line.startswith("1. "):
            continue
        for question_index in range(response_index + 1, min(response_index + 5, len(lines))):
            candidate = lines[question_index].strip()
            if candidate.startswith("*Student:"):
                answer = line.removeprefix("1. ").strip()
                question = clean_question(candidate)
                # Preserve only checklist targets that are literally one question.
                if question.count("?") == 1 and question.endswith("?"):
                    pairs.append(
                        {
                            "question": question,
                            "answer": answer,
                            "response_line": response_index + 1,
                            "question_line": question_index + 1,
                        }
                    )
                break
            if candidate:
                break
    return pairs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_root = root / "data" / "medsp1000"
    output_root = root / "processed"
    records = []
    for case in CASES:
        source = source_root / case["source"]
        if not source.is_file():
            raise SystemExit(f"Missing source checklist: {source}")
        pairs = extract_pairs(source)
        if not pairs:
            raise SystemExit(f"No direct pairs extracted from: {source}")
        for number, pair in enumerate(pairs, start=1):
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": case["opening"]},
                {"role": "assistant", "content": pair["question"]},
                {"role": "user", "content": pair["answer"]},
            ]
            assert messages[2]["content"].count("?") == 1
            records.append(
                {
                    "id": f"{case['case_id'].replace('/', '__')}__direct_{number:02d}",
                    "source_case_id": case["case_id"],
                    "source_quality": "verbatim evaluator response/question pair",
                    "source_file": case["source"],
                    "source_response_line": pair["response_line"],
                    "source_question_line": pair["question_line"],
                    "messages": messages,
                }
            )
    train = [record for record in records if record["source_case_id"] == CASES[0]["case_id"]]
    evaluation = [record for record in records if record["source_case_id"] == CASES[1]["case_id"]]
    assert {row["source_case_id"] for row in train}.isdisjoint(
        {row["source_case_id"] for row in evaluation}
    )
    for split, rows in (("train", train), ("eval", evaluation)):
        destination = output_root / f"medsp_direct_pairs_v1_{split}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    manifest = {
        "dataset_repo": "byrLLCC/MedSP1000",
        "method": "verbatim adjacent evaluator response/question extraction",
        "conversation_count": len(records),
        "train_count": len(train),
        "eval_count": len(evaluation),
        "train_case_ids": [CASES[0]["case_id"]],
        "eval_case_ids": [CASES[1]["case_id"]],
        "excluded": "Checklist entries containing more than one literal question or non-adjacent response/question text.",
    }
    (output_root / "medsp_direct_pairs_v1_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(train)} train and {len(evaluation)} eval direct-pair records")


if __name__ == "__main__":
    main()
