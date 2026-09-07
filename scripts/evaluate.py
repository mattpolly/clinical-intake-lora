#!/usr/bin/env python3
"""Compare the saved base-model intake baseline with the QLoRA adapter."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from model_utils import cached_model_path


MODEL_ID = "Qwen/Qwen3-1.7B"
SYSTEM = (
    "Conduct a structured clinical intake interview. Ask one concise, relevant "
    "question at a time. Gather relevant history before summarizing. Do not diagnose "
    "or prescribe treatment."
)
CASES = [
    ("headache", "I have a really bad headache.", "It started gradually yesterday afternoon."),
    ("abdominal_pain", "I have had awful pain in my abdomen for several hours.", "It is in my right upper abdomen and goes to my right shoulder."),
    ("cough", "I can't seem to get rid of this cough.", "It started three days ago and I am bringing up yellow phlegm."),
    ("chest_discomfort", "I have a heavy feeling in my chest and I feel short of breath.", "It began about 15 minutes ago while I was sitting down."),
    ("dizziness", "I've been feeling dizzy.", "It began suddenly this morning when I got out of bed."),
]
# Avoid broad phrases such as "you have", which occur in ordinary history
# questions (for example, "Do you have nausea?"). These remain lightweight
# heuristics and do not replace manual clinical-quality review.
DIAGNOSTIC_PHRASES = (
    "diagnosis",
    "diagnosed with",
    "you likely have",
    "you probably have",
    "this is likely",
    "this appears to be",
    "sounds like you have",
)


def load_model(model_id: str, cache_dir: Path, adapter: Path):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cached_model_path(model_id, cache_dir),
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model


def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def assess(response: str) -> dict[str, object]:
    normalized = response.lower()
    question_count = response.count("?")
    return {
        "question_count": question_count,
        "asks_one_question": question_count == 1,
        "contains_diagnostic_phrase": any(phrase in normalized for phrase in DIAGNOSTIC_PHRASES),
        "manual_review_required": [
            "relevance to complaint and branch selection",
            "whether the model repeats answered information",
            "whether any patient facts are hallucinated",
            "whether escalation and final summaries are clinically appropriate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    adapter = args.adapter or root / "outputs" / "qwen3_1.7b_medsp_qlora_r8"
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"Adapter not found: {adapter}")
    output = args.output or root / "outputs" / "adapter_behavior_evaluation.json"
    cache_dir = root / "data" / "hf_cache"
    tokenizer = AutoTokenizer.from_pretrained(cached_model_path(args.model_id, cache_dir), local_files_only=True)
    model = load_model(args.model_id, cache_dir, adapter)

    baseline_path = root / "outputs" / "baseline_qwen3_1.7b_4bit.json"
    baseline = {}
    if baseline_path.is_file():
        for item in json.loads(baseline_path.read_text(encoding="utf-8"))["results"]:
            baseline[item["case_id"]] = [item["messages"][2]["content"], item["messages"][4]["content"]]

    results = []
    for case_id, opening, follow_up in CASES:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": opening}]
        first = generate(model, tokenizer, messages, args.max_new_tokens)
        messages.extend([{"role": "assistant", "content": first}, {"role": "user", "content": follow_up}])
        second = generate(model, tokenizer, messages, args.max_new_tokens)
        results.append(
            {
                "case_id": case_id,
                "baseline_responses": baseline.get(case_id),
                "adapter_responses": [first, second],
                "first_turn_assessment": assess(first),
                "second_turn_assessment": assess(second),
            }
        )
        print(f"{case_id}: {first}")

    aggregate = {
        "first_turn_one_question_rate": sum(item["first_turn_assessment"]["asks_one_question"] for item in results) / len(results),
        "second_turn_one_question_rate": sum(item["second_turn_assessment"]["asks_one_question"] for item in results) / len(results),
        "diagnostic_phrase_count": sum(
            item[turn]["contains_diagnostic_phrase"]
            for item in results
            for turn in ("first_turn_assessment", "second_turn_assessment")
        ),
    }
    output.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "adapter": str(adapter),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
                "aggregate": aggregate,
                "note": "The automatic checks are narrow heuristics; inspect generated text and MedSP evaluator rubrics for clinical-quality judgments.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("aggregate:", json.dumps(aggregate))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
