#!/usr/bin/env python3
"""Validate the compact capability-training dataset contract.

This is intentionally a transparent structural and lexical gate.  It does not
make clinical judgments; safety targets still require manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


TOP_LEVEL = {"id", "case", "source", "messages", "facts", "rubric", "turns"}
FACT_FIELDS = {"chief_complaint", "hpi", "associated", "meds_allergies", "history", "social_family"}
RUBRIC_TIERS = {"core", "red_flag", "background"}
ACTIONS = {"question", "safety_followup", "escalation", "summary"}
ROLES = {"system", "user", "assistant"}
FORBIDDEN_ASSISTANT_PATTERNS = (
    r"\bmy diagnosis\b",
    r"\bthe diagnosis is\b",
    r"\bdiagnosis:\b",
    r"\byou (?:likely|probably) have\b",
    r"\bthis (?:is|appears to be|sounds like)\b",
    r"\btreatment plan\b",
    r"\byou should (?:start|take|stop)\b",
    r"\b(?:start|stop) taking\b",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return rows


def has_forbidden_assistant_text(text: str) -> str | None:
    normalized = text.lower()
    return next((pattern for pattern in FORBIDDEN_ASSISTANT_PATTERNS if re.search(pattern, normalized)), None)


def validate_record(record: dict, root: Path, origin: str) -> list[str]:
    errors: list[str] = []
    prefix = f"{origin}:{record.get('id', '<missing id>')}"
    if set(record) != TOP_LEVEL:
        errors.append(f"{prefix}: top-level fields must be exactly {sorted(TOP_LEVEL)}")
        return errors

    case, source, messages = record["case"], record["source"], record["messages"]
    if not isinstance(record["id"], str) or not record["id"].strip():
        errors.append(f"{prefix}: id must be a non-empty string")
    if not isinstance(case, dict) or set(case) != {"id", "split"} or case.get("split") not in {"train", "eval"}:
        errors.append(f"{prefix}: case must contain exactly id and split=train|eval")
    if not isinstance(source, dict) or set(source) != {"sp", "evaluator"}:
        errors.append(f"{prefix}: source must contain exactly sp and evaluator")
    else:
        for key, relative_path in source.items():
            if not isinstance(relative_path, str) or not relative_path.strip():
                errors.append(f"{prefix}: source.{key} must be non-empty")
            elif not (root / relative_path).is_file():
                errors.append(f"{prefix}: source.{key} does not exist: {relative_path}")

    if not isinstance(messages, list) or len(messages) < 3:
        errors.append(f"{prefix}: messages must contain at least system, user, and assistant turns")
        return errors
    if messages[0].get("role") != "system":
        errors.append(f"{prefix}: messages[0] must be the system instruction")
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            errors.append(f"{prefix}: messages[{index}] must contain exactly role and content")
            continue
        if item["role"] not in ROLES or not isinstance(item["content"], str) or not item["content"].strip():
            errors.append(f"{prefix}: invalid message at index {index}")

    facts = record["facts"]
    if not isinstance(facts, list):
        errors.append(f"{prefix}: facts must be a list")
        facts = []
    fact_ids = [fact.get("id") for fact in facts if isinstance(fact, dict)]
    if len(fact_ids) != len(set(fact_ids)):
        errors.append(f"{prefix}: fact IDs must be unique")
    for fact in facts:
        if not isinstance(fact, dict) or set(fact) != {"id", "field", "reveal_in", "terms", "evidence"}:
            errors.append(f"{prefix}: each fact must have exactly id, field, reveal_in, terms, evidence")
            continue
        reveal_in = fact["reveal_in"]
        if fact["field"] not in FACT_FIELDS:
            errors.append(f"{prefix}: fact {fact['id']} has invalid field {fact['field']!r}")
        if not isinstance(reveal_in, int) or not 0 <= reveal_in < len(messages) or messages[reveal_in].get("role") != "user":
            errors.append(f"{prefix}: fact {fact['id']} must reveal in an in-range user message")
        if not isinstance(fact["terms"], list) or not fact["terms"] or not all(isinstance(term, str) and term.strip() for term in fact["terms"]):
            errors.append(f"{prefix}: fact {fact['id']} needs non-empty matching terms")
        elif isinstance(reveal_in, int) and 0 <= reveal_in < len(messages):
            answer = messages[reveal_in].get("content", "").lower()
            if not all(term.lower() in answer for term in fact["terms"]):
                errors.append(f"{prefix}: fact {fact['id']} terms must occur in its revealed patient message")
        if not isinstance(fact["evidence"], str) or not fact["evidence"].strip():
            errors.append(f"{prefix}: fact {fact['id']} needs a source locator")

    rubric = record["rubric"]
    if not isinstance(rubric, list):
        errors.append(f"{prefix}: rubric must be a list")
        rubric = []
    rubric_ids = [item.get("id") for item in rubric if isinstance(item, dict)]
    if len(rubric_ids) != len(set(rubric_ids)):
        errors.append(f"{prefix}: rubric IDs must be unique")
    red_flag_ids = set()
    for item in rubric:
        if not isinstance(item, dict) or set(item) != {"id", "tier", "evidence"}:
            errors.append(f"{prefix}: each rubric item must have exactly id, tier, evidence")
            continue
        if item["tier"] not in RUBRIC_TIERS:
            errors.append(f"{prefix}: rubric {item['id']} has invalid tier {item['tier']!r}")
        if item["tier"] == "red_flag":
            red_flag_ids.add(item["id"])
        if not isinstance(item["evidence"], str) or not item["evidence"].strip():
            errors.append(f"{prefix}: rubric {item['id']} needs an evaluator locator")

    turns = record["turns"]
    if not isinstance(turns, list) or not turns:
        errors.append(f"{prefix}: turns must be a non-empty list")
        turns = []
    assistant_indices = []
    covered_items: set[str] = set()
    safety_items: set[str] = set()
    negative_screen_items: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {"assistant_in", "action", "items"}:
            errors.append(f"{prefix}: each turn must have exactly assistant_in, action, items")
            continue
        index, action, items = turn["assistant_in"], turn["action"], turn["items"]
        assistant_indices.append(index)
        if not isinstance(index, int) or not 0 <= index < len(messages) or messages[index].get("role") != "assistant":
            errors.append(f"{prefix}: turn index {index!r} must point to an assistant message")
            continue
        if action not in ACTIONS:
            errors.append(f"{prefix}: turn {index} has invalid action {action!r}")
        if not isinstance(items, list) or any(item not in rubric_ids for item in items):
            errors.append(f"{prefix}: turn {index} items must reference rubric IDs")
        if action == "summary" and items:
            errors.append(f"{prefix}: summary turn {index} must have an empty items list")
        if action != "summary" and not items:
            errors.append(f"{prefix}: {action} turn {index} needs at least one rubric item")
        text = messages[index]["content"]
        if action == "question" and text.count("?") != 1:
            errors.append(f"{prefix}: question turn {index} must contain exactly one question mark")
        if action == "summary" and "not obtained" not in text.lower():
            errors.append(f"{prefix}: summary turn {index} must acknowledge not obtained information")
        forbidden = has_forbidden_assistant_text(text)
        if forbidden:
            errors.append(f"{prefix}: assistant turn {index} contains disallowed diagnostic/treatment language ({forbidden})")
        if action == "question":
            duplicate_items = set(items) & covered_items
            if duplicate_items:
                errors.append(f"{prefix}: question turn {index} repeats already-covered rubric item(s) {sorted(duplicate_items)}")
            covered_items.update(items)
            if index + 1 < len(messages) and messages[index + 1].get("role") == "user":
                answer = messages[index + 1]["content"].lower()
                if any(token in answer for token in (" no", "never", "denies", "do not")):
                    negative_screen_items.update(items)
        if action in {"safety_followup", "escalation"}:
            safety_items.update(items)

    if len(assistant_indices) != len(set(assistant_indices)):
        errors.append(f"{prefix}: only one annotation is allowed per assistant message")
    if red_flag_ids and not safety_items and not (red_flag_ids & negative_screen_items):
        errors.append(f"{prefix}: red-flag rubric needs a safety_followup or escalation target")

    # A narrow hidden-fact check: full fact phrases should not be asserted in a
    # prior assistant target. Questions are excluded because a question can
    # legitimately name a symptom before the patient answers it.
    for fact in facts:
        reveal_in = fact.get("reveal_in")
        if not isinstance(reveal_in, int):
            continue
        for index, item in enumerate(messages[:reveal_in]):
            if item.get("role") != "assistant" or "?" in item.get("content", ""):
                continue
            if all(term.lower() in item["content"].lower() for term in fact.get("terms", [])):
                errors.append(f"{prefix}: assistant message {index} asserts hidden fact {fact.get('id')}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", type=Path, help="One or more JSONL capability datasets.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = args.datasets or [
        root / "processed" / "medsp_capability_pilot_v1_train.jsonl",
        root / "processed" / "medsp_capability_pilot_v1_eval.jsonl",
    ]
    errors: list[str] = []
    records: list[tuple[dict, Path]] = []
    for path in paths:
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"missing dataset: {path}")
            continue
        try:
            records.extend((record, path) for record in read_jsonl(path))
        except ValueError as exc:
            errors.append(str(exc))

    ids = [record.get("id") for record, _ in records]
    for duplicate, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate record id across inputs: {duplicate}")
    split_by_case: dict[str, str] = {}
    for record, path in records:
        errors.extend(validate_record(record, root, str(path.relative_to(root))))
        case = record.get("case", {})
        if isinstance(case, dict) and isinstance(case.get("id"), str) and case.get("split") in {"train", "eval"}:
            existing = split_by_case.setdefault(case["id"], case["split"])
            if existing != case["split"]:
                errors.append(f"case split leak: {case['id']} is in both {existing} and {case['split']}")

    if errors:
        print("CAPABILITY DATASET VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print(f"CAPABILITY DATASET VALIDATION PASSED: {len(records)} records, {len(split_by_case)} complete cases, no split overlap")


if __name__ == "__main__":
    main()
