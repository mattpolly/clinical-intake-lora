#!/usr/bin/env python3
"""Save deterministic pre-fine-tuning Qwen3 intake responses for comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
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
    ("dyspnea", "I'm really short of breath. I could barely make it back from the bathroom.", "It woke me from sleep and gets worse with any activity."),
]


def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "outputs" / "baseline_qwen3_1.7b_4bit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "data" / "hf_cache"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cached_model_path(args.model_id, cache_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        cached_model_path(args.model_id, cache_dir),
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    results = []
    for case_id, opening, follow_up in CASES:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": opening},
        ]
        first = generate(model, tokenizer, messages, args.max_new_tokens)
        messages.extend([{"role": "assistant", "content": first}, {"role": "user", "content": follow_up}])
        second = generate(model, tokenizer, messages, args.max_new_tokens)
        messages.append({"role": "assistant", "content": second})
        results.append({"case_id": case_id, "messages": messages})
        print(f"{case_id}: {first}")

    output.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "quantization": "4-bit NF4 with fp16 compute",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
