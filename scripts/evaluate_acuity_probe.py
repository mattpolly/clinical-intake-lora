#!/usr/bin/env python3
"""Run a constrained emergency-pattern recognition probe against a base model."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model_utils import cached_model_path


INSTRUCTION = """You are classifying whether the stated symptoms contain an explicit emergency warning pattern. Do not diagnose, explain, or give treatment. Reply with exactly one label: EMERGENCY or NOT_EMERGENCY. Use EMERGENCY only when the stated information itself indicates immediate emergency evaluation; otherwise use NOT_EMERGENCY."""


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_label(text):
    normalized = text.upper()
    # NOT must be checked before EMERGENCY because it contains that token.
    if re.search(r"\bNOT_EMERGENCY\b|\bNOT EMERGENCY\b", normalized):
        return "not_emergency"
    if re.search(r"\bEMERGENCY\b", normalized):
        return "emergency"
    return "unparseable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter", type=Path, help="Optional PEFT adapter to compare with the untouched base model.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_path = args.data or root / "processed" / "acuity_probe_v1.jsonl"
    output = args.output or root / "outputs" / "acuity_probe_v1_qwen3_4b_base.json"
    rows = load_rows(data_path)
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(cached_model_path(args.model_id, root / "data" / "hf_cache"), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cached_model_path(args.model_id, root / "data" / "hf_cache"), local_files_only=True, quantization_config=config, torch_dtype=torch.float16, device_map="auto")
    if args.adapter:
        adapter = args.adapter if args.adapter.is_absolute() else root / args.adapter
        if not (adapter / "adapter_config.json").is_file():
            raise SystemExit(f"Adapter not found: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    results=[]
    for row in rows:
        messages=[{"role":"system","content":INSTRUCTION},{"role":"user","content":row["presentation"]}]
        prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs=tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated=model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text=tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        predicted=parse_label(text)
        results.append({**row,"prediction":predicted,"raw_response":text,"correct":predicted == row["label"]})
    labels=Counter(item["label"] for item in results)
    by_system={}
    for system in sorted({item["system"] for item in results}):
        group=[item for item in results if item["system"] == system]
        by_system[system]={"count":len(group),"correct":sum(item["correct"] for item in group),"accuracy":sum(item["correct"] for item in group)/len(group)}
    emergency=[item for item in results if item["label"] == "emergency"]
    controls=[item for item in results if item["label"] == "not_emergency"]
    aggregate={"count":len(results),"labels":dict(labels),"accuracy":sum(item["correct"] for item in results)/len(results),"emergency_recall":sum(item["prediction"] == "emergency" for item in emergency)/len(emergency),"control_specificity":sum(item["prediction"] == "not_emergency" for item in controls)/len(controls),"unparseable":sum(item["prediction"] == "unparseable" for item in results),"by_system":by_system}
    output.write_text(json.dumps({"created_at":datetime.now(timezone.utc).isoformat(),"model_id":args.model_id,"adapter":str(args.adapter) if args.adapter else None,"instruction":INSTRUCTION,"method_note":"Small manually curated cited probe. Labels indicate whether an explicit emergency pattern is stated, not a diagnosis or clinical decision rule.","aggregate":aggregate,"results":results},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(aggregate,indent=2)); print(f"saved: {output}")


if __name__ == "__main__":
    main()
