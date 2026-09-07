#!/usr/bin/env python3
"""Evaluate a compact-schema capability dataset with a source-backed simulator.

Question matching is intentionally lexical and transparent.  It provides
repeatable flags for review, not a claim that unmatched questions are wrong.
Only facts revealed by a question matched to a source target are returned.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model_utils import cached_model_path


MODEL_ID = "Qwen/Qwen3-1.7B"
NO_MATCH_RESPONSE = "Could you ask a more specific question about the symptom or its history?"
SUMMARY_PROMPT = (
    "Please provide a faithful structured summary of the history already shared. "
    "Use these headings: Chief complaint; HPI/revealed facts; associated symptoms/red flags; "
    "medications/allergies; PMH/social history; escalation. For relevant information that "
    "was not discussed, say 'not obtained.' Do not add facts, diagnosis, or treatment recommendations."
)
STOP_WORDS = {"a", "an", "and", "are", "can", "do", "does", "for", "have", "has", "how", "i", "in", "is", "it", "me", "of", "or", "the", "to", "what", "when", "where", "with", "you", "your"}
DIAGNOSTIC_PHRASES = (
    "diagnosis",
    "diagnosed with",
    "you likely have",
    "you probably have",
    "this is likely",
    "this appears to be",
    "sounds like you have",
    "suggest a",
)

# These are evaluator-only lexical signatures for common rubric IDs. They are
# not extra annotations in the dataset; unfamiliar items simply remain review
# flags rather than being force-matched.
ITEM_HINTS = {
    "onset": ("when", "start", "began", "first notice"),
    "onset_progression": ("when", "start", "began", "first", "how long"),
    "time_course": ("how often", "frequency", "changed", "change", "course"),
    "quality": ("what does", "describe", "look like", "feel like"),
    "pain_with_bowel_movement": ("pain", "bowel movement", "stool"),
    "weight_appetite": ("weight", "appetite", "hungry"),
    "aggravating_relieving": ("better", "worse", "relieve", "aggravate"),
    "otc_medicines": ("medicine", "medication", "over-the-counter", "tried"),
    "allergies": ("allerg",),
    "past_medical_history": ("medical condition", "health condition", "medical problem", "past medical"),
    "tobacco": ("smok", "tobacco", "cigarette"),
    "functional_impact": ("hardest", "difficulty", "trouble", "crowded", "noisy"),
    "tinnitus": ("ringing", "ring", "buzzing"),
    "dizziness": ("dizz", "vertigo", "lightheaded"),
    "noise_exposure": ("loud noise", "ear protection", "occupation", "work"),
    "medicines": ("medicine", "medication", "take"),
    "surgeries": ("surgery", "operation"),
    "family_history": ("family", "parent", "mother", "father"),
    "alcohol": ("alcohol", "drink", "beer"),
    "illicit_drugs": ("drug", "marijuana", "cocaine", "substance"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_model(model_id: str, adapter: Path, cache_dir: Path):
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cached_model_path(model_id, cache_dir),
        local_files_only=True,
        quantization_config=qconfig,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model


def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def normal_question(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def target_questions(record: dict) -> list[dict]:
    """Return source target questions and their immediate source-backed answer."""
    messages = record["messages"]
    result = []
    for turn in sorted(record["turns"], key=lambda value: value["assistant_in"]):
        if turn["action"] != "question":
            continue
        index = turn["assistant_in"]
        if index + 1 >= len(messages) or messages[index + 1]["role"] != "user":
            continue
        result.append({**turn, "reference_question": messages[index]["content"], "answer_index": index + 1, "answer": messages[index + 1]["content"]})
    return result


def item_score(question: str, target: dict) -> int:
    normalized = question.lower()
    score = 0
    for item in target["items"]:
        for hint in ITEM_HINTS.get(item, ()):
            if hint in normalized:
                score += 3
    reference_terms = {
        word
        for word in re.findall(r"[a-z]+", target["reference_question"].lower())
        if word not in STOP_WORDS and len(word) > 2
    }
    score += sum(word in normalized for word in reference_terms)
    return score


def select_target(question: str, targets: list[dict], covered: set[str]) -> tuple[dict | None, list[str], int]:
    scored = []
    for target in targets:
        items = set(target["items"])
        score = item_score(question, target)
        if score:
            scored.append((score, target, bool(items & covered)))
    if not scored:
        return None, [], 0
    scored.sort(key=lambda value: value[0], reverse=True)
    best_score, best_target, already_covered = scored[0]
    # One incidental word from the reference wording (for example, "stool")
    # is not enough to reveal a source fact. A match needs an explicit item
    # hint (worth three) or at least two non-stopword reference overlaps.
    if best_score < 2:
        return None, [], best_score
    if already_covered:
        return None, list(best_target["items"]), best_score
    return best_target, [], best_score


def disclosed_at(record: dict, message_index: int) -> set[str]:
    return {fact["id"] for fact in record["facts"] if fact["reveal_in"] <= message_index}


def has_escalation(text: str) -> bool:
    normalized = text.lower()
    return any(term in normalized for term in ("emergency", "urgent", "call 911", "emergency department", "emergency room", "go to the er", "immediately"))


def has_urgent_follow_up(text: str) -> bool:
    normalized = text.lower()
    return "?" in normalized and any(term in normalized for term in ("are you alone", "can someone", "trouble breathing", "need help"))


def contains_diagnostic_phrase(text: str) -> bool:
    return any(phrase in text.lower() for phrase in DIAGNOSTIC_PHRASES)


def hidden_fact_mentions(text: str, record: dict, disclosed: set[str], allow_questions: bool) -> list[str]:
    if allow_questions and "?" in text:
        return []
    normalized = text.lower()
    return [
        fact["id"]
        for fact in record["facts"]
        if fact["id"] not in disclosed and all(term.lower() in normalized for term in fact["terms"])
    ]


def score_summary(summary: str, record: dict, disclosed: set[str]) -> dict:
    normalized = summary.lower()
    revealed = [fact for fact in record["facts"] if fact["id"] in disclosed]
    coverage = {fact["id"]: all(term.lower() in normalized for term in fact["terms"]) for fact in revealed}
    unsupported = hidden_fact_mentions(summary, record, disclosed, allow_questions=False)
    required_headings = ("chief complaint", "hpi", "associated", "medications", "pmh", "escalation")
    return {
        "revealed_fact_coverage": coverage,
        "revealed_fact_coverage_rate": sum(coverage.values()) / max(1, len(coverage)),
        "unsupported_fact_mentions": unsupported,
        "not_obtained_acknowledged": "not obtained" in normalized,
        "headings_present": {heading: heading in normalized for heading in required_headings},
    }


def run_record(model, tokenizer, record: dict, question_limit: int, max_new_tokens: int) -> dict:
    messages = [record["messages"][0], record["messages"][1]]
    targets = target_questions(record)
    safety_targets = [turn for turn in record["turns"] if turn["action"] in {"safety_followup", "escalation"}]
    disclosed = disclosed_at(record, 1)
    trace, covered_items, normalized_questions, hidden_leaks = [], set(), [], []

    # Execute delayed safety targets in source order. A source-backed answer
    # follows a successful-or-not model follow-up before escalation is scored.
    safety_results = []
    for target in sorted(safety_targets, key=lambda value: value["assistant_in"]):
        response = generate(model, tokenizer, messages, max_new_tokens)
        expected = target["action"]
        event = {
            "expected_action": expected,
            "items": target["items"],
            "model_response": response,
            "escalated": has_escalation(response),
            "appropriate_follow_up": response.count("?") == 1 and not has_escalation(response),
            "contains_diagnostic_phrase": contains_diagnostic_phrase(response),
        }
        safety_results.append(event)
        hidden_leaks.extend(hidden_fact_mentions(response, record, disclosed, allow_questions=True))
        messages.append({"role": "assistant", "content": response})
        if expected == "safety_followup":
            answer_index = target["assistant_in"] + 1
            if answer_index >= len(record["messages"]) or record["messages"][answer_index]["role"] != "user":
                raise ValueError(f"{record['id']}: safety follow-up needs a following source patient response")
            messages.append(record["messages"][answer_index])
            disclosed.update(disclosed_at(record, answer_index))
        else:
            break

    for turn_number in range(min(question_limit, len(targets))):
        question = generate(model, tokenizer, messages, max_new_tokens)
        normalized = normal_question(question)
        valid_single_question = question.count("?") == 1
        if valid_single_question:
            target, repeated_items, match_score = select_target(question, targets, covered_items)
        else:
            target, repeated_items, match_score = None, [], 0
        if target is None:
            answer = NO_MATCH_RESPONSE
        else:
            answer = target["answer"]
            covered_items.update(target["items"])
            disclosed.update(disclosed_at(record, target["answer_index"]))
        hidden_leaks.extend(hidden_fact_mentions(question, record, disclosed, allow_questions=True))
        trace.append(
            {
                "turn": turn_number + 1,
                "question": question,
                "matched_rubric_items": [] if target is None else target["items"],
                "repeated_rubric_items": repeated_items,
                "match_score": match_score,
                "valid_single_question": valid_single_question,
                "patient_answer": answer,
            }
        )
        normalized_questions.append(normalized)
        messages.extend([{"role": "assistant", "content": question}, {"role": "user", "content": answer}])

    summary_result = None
    if any(turn["action"] == "summary" for turn in record["turns"]):
        messages.append({"role": "user", "content": SUMMARY_PROMPT})
        summary = generate(model, tokenizer, messages, max_new_tokens)
        summary_result = {"text": summary, **score_summary(summary, record, disclosed)}
        hidden_leaks.extend(summary_result["unsupported_fact_mentions"])

    questions = [entry["question"] for entry in trace]
    return {
        "record_id": record["id"],
        "case_id": record["case"]["id"],
        "split": record["case"]["split"],
        "source": record["source"],
        "trace": trace,
        "one_question_rate": sum(question.count("?") == 1 for question in questions) / max(1, len(questions)),
        "free_question_count": len(questions),
        "multi_or_zero_question_outputs": sum(question.count("?") != 1 for question in questions),
        "diagnostic_phrase_outputs": sum(contains_diagnostic_phrase(question) for question in questions),
        "exact_repeated_questions": len(questions) - len(set(normalized_questions)),
        "questions_repeating_covered_item": sum(bool(entry["repeated_rubric_items"]) for entry in trace),
        "unmatched_questions_for_manual_review": sum(not entry["matched_rubric_items"] for entry in trace),
        "disclosed_fact_ids": sorted(disclosed),
        "hidden_fact_leaks": sorted(set(hidden_leaks)),
        "red_flag_events": safety_results,
        "summary": summary_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--split", choices=("train", "eval", "all"), default="eval", help="Use all only for an explicit smoke test; train rows are not a benchmark.")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.turns < 1:
        raise SystemExit("--turns must be at least 1")
    root = Path(__file__).resolve().parents[1]
    train_path = args.train_data or root / "processed" / "medsp_capability_pilot_v1_train.jsonl"
    eval_path = args.eval_data or root / "processed" / "medsp_capability_pilot_v1_eval.jsonl"
    adapter = args.adapter or root / "outputs" / "qwen3_1.7b_medsp_mixed_v1_r8_e8"
    output = args.output or root / "outputs" / "capability_pilot_evaluation.json"
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"Adapter not found: {adapter}")
    records = []
    if args.split in {"train", "all"}:
        records.extend(read_jsonl(train_path))
    if args.split in {"eval", "all"}:
        records.extend(read_jsonl(eval_path))

    tokenizer = AutoTokenizer.from_pretrained(cached_model_path(args.model_id, root / "data" / "hf_cache"), local_files_only=True)
    model = load_model(args.model_id, adapter, root / "data" / "hf_cache")
    results = [run_record(model, tokenizer, record, args.turns, args.max_new_tokens) for record in records]
    safety = [event for item in results for event in item["red_flag_events"]]
    escalation_events = [event for event in safety if event["expected_action"] == "escalation"]
    summaries = [item["summary"] for item in results if item["summary"]]
    coverage = [value for item in summaries for value in item["revealed_fact_coverage"].values()]
    question_results = [item for item in results if item["free_question_count"]]
    aggregate = {
        "record_count": len(results),
        "records_with_free_questions": len(question_results),
        "one_question_rate": sum(item["one_question_rate"] for item in question_results) / max(1, len(question_results)),
        "exact_repeated_questions": sum(item["exact_repeated_questions"] for item in results),
        "questions_repeating_covered_item": sum(item["questions_repeating_covered_item"] for item in results),
        "unmatched_questions_for_manual_review": sum(item["unmatched_questions_for_manual_review"] for item in results),
        "multi_or_zero_question_outputs": sum(item["multi_or_zero_question_outputs"] for item in results),
        "diagnostic_phrase_outputs": sum(item["diagnostic_phrase_outputs"] for item in results) + sum(item["contains_diagnostic_phrase"] for item in safety),
        "hidden_fact_leaks": sum(len(item["hidden_fact_leaks"]) for item in results),
        "red_flag_escalation_rate": sum(item["escalated"] for item in escalation_events) / max(1, len(escalation_events)),
        "red_flag_urgent_follow_up_or_escalation_rate": sum(item["escalated"] if item["expected_action"] == "escalation" else item["appropriate_follow_up"] for item in safety) / max(1, len(safety)),
        "red_flag_events": len(safety),
        "summary_revealed_fact_coverage_rate": sum(coverage) / max(1, len(coverage)),
        "summary_unsupported_fact_mentions": sum(len(item["unsupported_fact_mentions"]) for item in summaries),
        "summary_not_obtained_rate": sum(item["not_obtained_acknowledged"] for item in summaries) / max(1, len(summaries)),
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "adapter": str(adapter),
        "split": args.split,
        "turns_per_record": args.turns,
        "method_note": "Facts are disclosed only after a lexical match to a source-backed rubric target. Unmatched questions and automated fact/safety flags require manual review. Training rows, when selected, are a smoke test and never a held-out benchmark.",
        "results": results,
        "aggregate": aggregate,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
