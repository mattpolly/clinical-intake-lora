#!/usr/bin/env python3
"""Minimal local multi-turn terminal chat with the saved QLoRA adapter."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--max-history-messages",
        type=int,
        default=12,
        help="Maximum non-system messages retained in the prompt.",
    )
    parser.add_argument(
        "--save-transcript",
        type=Path,
        help="Optional local JSONL path; use only synthetic test conversations, never PHI.",
    )
    args = parser.parse_args()
    if args.max_history_messages < 2:
        raise SystemExit("--max-history-messages must be at least 2")
    root = Path(__file__).resolve().parents[1]
    adapter = args.adapter or root / "outputs" / "qwen3_1.7b_medsp_mixed_v1_r8_e8"
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"Adapter not found: {adapter}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for local 4-bit inference")

    cache_dir = root / "data" / "hf_cache"
    tokenizer = AutoTokenizer.from_pretrained(cached_model_path(args.model_id, cache_dir), local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        cached_model_path(args.model_id, cache_dir),
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    messages = [{"role": "system", "content": SYSTEM}]
    saved_turns = []
    print("Research-only clinical-intake chat — use synthetic test cases only; do not enter PHI.")
    print("Type /reset to clear history, /summary after six patient turns, or /quit to exit.")

    def retain_history() -> None:
        nonlocal messages
        messages = [messages[0], *messages[1:][-args.max_history_messages :]]

    def respond(user_text: str) -> str:
        messages.append({"role": "user", "content": user_text})
        retain_history()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": reply})
        retain_history()
        saved_turns.append({"user": user_text, "assistant": reply})
        return reply

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/reset":
            messages = [{"role": "system", "content": SYSTEM}]
            print("history cleared")
            continue
        if user_text == "/summary":
            patient_turns = sum(message["role"] == "user" for message in messages[1:])
            if patient_turns < 6:
                print(f"summary unavailable: collect more history first ({patient_turns}/6 patient turns)")
                continue
            user_text = (
                "Please provide a faithful structured summary of the history already shared. "
                "Do not add facts, diagnosis, or treatment recommendations."
            )
        reply = respond(user_text)
        print(f"assistant> {reply}")

    if args.save_transcript and saved_turns:
        args.save_transcript.parent.mkdir(parents=True, exist_ok=True)
        args.save_transcript.write_text(
            json.dumps({"model_id": args.model_id, "adapter": str(adapter), "turns": saved_turns}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"saved synthetic transcript: {args.save_transcript}")


if __name__ == "__main__":
    main()
