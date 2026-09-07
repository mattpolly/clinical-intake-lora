#!/usr/bin/env python3
"""Measure whether a model changes intake mode as source-backed facts accrue."""
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

INSTRUCTION = """Classify the next interview mode from only the history stated so far. Do not diagnose, explain, or give treatment. Reply with exactly one label: CONTINUE_HISTORY, FOCUSED_SAFETY_CHECK, or EMERGENCY. CONTINUE_HISTORY: no stated emergency pattern and no immediate targeted safety screen is required. FOCUSED_SAFETY_CHECK: a potentially serious pattern needs one focused safety question, but an explicit emergency pattern is not yet stated. EMERGENCY: the stated information itself contains an explicit emergency warning pattern requiring immediate emergency evaluation."""
LABELS = ("CONTINUE_HISTORY", "FOCUSED_SAFETY_CHECK", "EMERGENCY")


def parse(text):
    upper = text.upper()
    for label in LABELS:
        if re.search(r"\b" + label + r"\b", upper):
            return label
    return "UNPARSEABLE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = args.data or root / "processed" / "acuity_transition_probe_v1.jsonl"
    output = args.output or root / "outputs" / "acuity_transition_probe_v1_qwen3_4b_base.json"
    cases = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(cached_model_path(args.model_id, root / "data" / "hf_cache"), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cached_model_path(args.model_id, root / "data" / "hf_cache"), local_files_only=True, quantization_config=config, torch_dtype=torch.float16, device_map="auto")
    if args.adapter:
        adapter = args.adapter if args.adapter.is_absolute() else root / args.adapter
        if not (adapter / "adapter_config.json").is_file(): raise SystemExit(f"Adapter not found: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    results=[]
    for case in cases:
        history=[]
        for position, item in enumerate(case["states"], 1):
            history.append(item["new_user_fact"])
            prompt="History so far:\n" + "\n".join(f"- {fact}" for fact in history)
            messages=[{"role":"system","content":INSTRUCTION},{"role":"user","content":prompt}]
            rendered=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs=tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated=model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text=tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            predicted=parse(text)
            results.append({"case_id":case["id"],"system":case["system"],"state_index":position,"history":list(history),"expected_mode":item["expected_mode"],"prediction":predicted,"raw_response":text,"correct":predicted==item["expected_mode"]})
    emergency=[item for item in results if item["expected_mode"]=="EMERGENCY"]
    nonemergency=[item for item in results if item["expected_mode"]!="EMERGENCY"]
    transitions=[]
    for case in cases:
        group=[item for item in results if item["case_id"]==case["id"]]
        transitions.append({"case_id":case["id"],"correct_mode_sequence":all(item["correct"] for item in group),"expected":[item["expected_mode"] for item in group],"predicted":[item["prediction"] for item in group]})
    aggregate={"states":len(results),"cases":len(cases),"accuracy":sum(item["correct"] for item in results)/len(results),"emergency_recall":sum(item["prediction"]=="EMERGENCY" for item in emergency)/len(emergency),"nonemergency_mode_accuracy":sum(item["correct"] for item in nonemergency)/len(nonemergency),"complete_correct_transitions":sum(item["correct_mode_sequence"] for item in transitions),"unparseable":sum(item["prediction"]=="UNPARSEABLE" for item in results)}
    output.write_text(json.dumps({"created_at":datetime.now(timezone.utc).isoformat(),"model_id":args.model_id,"adapter":str(args.adapter) if args.adapter else None,"instruction":INSTRUCTION,"method_note":"Small source-cited mode-transition probe. It evaluates explicit stated warning patterns, not diagnosis or a deployable triage rule.","aggregate":aggregate,"transitions":transitions,"results":results},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(aggregate,indent=2)); print(f"saved: {output}")


if __name__ == "__main__":
    main()
