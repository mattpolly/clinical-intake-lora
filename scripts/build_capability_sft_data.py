#!/usr/bin/env python3
"""Export compact-schema capability records as assistant-target SFT prefixes."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from model_utils import cached_model_path

MODEL_ID="Qwen/Qwen3-4B"
def read(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
def write(p,rows): p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
def export(records, tokenizer, max_length):
 rows=[]; excluded=[]
 for record in records:
  for turn in record["turns"]:
   index=turn["assistant_in"]; messages=record["messages"][:index+1]
   assert messages[-1]["role"]=="assistant"
   rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
   tokens=len(tokenizer(rendered,add_special_tokens=False)["input_ids"])
   item={"id":f"{record['id']}__assistant_{index}","source_case_id":record["case"]["id"],"target_action":turn["action"],"rubric_items":turn["items"],"token_count":tokens,"messages":messages}
   if tokens<=max_length: rows.append(item)
   else: excluded.append({"id":item["id"],"source_case_id":item["source_case_id"],"target_action":item["target_action"],"token_count":tokens})
 return rows,excluded


def oversample_escalations(rows, factor):
 """Duplicate only training escalation targets, with traceable copy IDs.

 This changes sampling frequency, not the compact source dataset or the held-out
 evaluation set.  Keeping the original source_case_id lets training/auditing
 tools distinguish weighting copies from new patient scenarios.
 """
 if factor < 1:
  raise ValueError("escalation oversample factor must be at least 1")
 weighted=[]
 for row in rows:
  copies=factor if row["target_action"] == "escalation" else 1
  for copy_number in range(1, copies + 1):
   item=dict(row)
   item["sampling_copy"]=copy_number
   if copy_number > 1:
    item["id"]=f"{row['id']}__weight_{copy_number}"
   weighted.append(item)
 return weighted
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--model-id",default=MODEL_ID); ap.add_argument("--max-length",type=int,default=512); ap.add_argument("--train-data",type=Path); ap.add_argument("--eval-data",type=Path); ap.add_argument("--output-prefix",default="medsp_capability_v3_sft"); ap.add_argument("--escalation-oversample",type=int,default=1,help="Training-only weighting factor for escalation targets.") ; a=ap.parse_args()
 if a.max_length<128: raise SystemExit("--max-length must be at least 128")
 if a.escalation_oversample<1: raise SystemExit("--escalation-oversample must be at least 1")
 root=Path(__file__).resolve().parents[1]; train_path=a.train_data or root/"processed"/"medsp_capability_v3_train.jsonl"; eval_path=a.eval_data or root/"processed"/"medsp_capability_v3_eval.jsonl"
 if not train_path.is_absolute(): train_path=root/train_path
 if not eval_path.is_absolute(): eval_path=root/eval_path
 train,ev=read(train_path),read(eval_path); train_cases={x["case"]["id"] for x in train}; eval_cases={x["case"]["id"] for x in ev}; assert not train_cases&eval_cases,"case split leak"
 tokenizer=AutoTokenizer.from_pretrained(cached_model_path(a.model_id,root/"data"/"hf_cache"),local_files_only=True)
 train_rows,train_excluded=export(train,tokenizer,a.max_length); eval_rows,eval_excluded=export(ev,tokenizer,a.max_length)
 if not train_rows or not eval_rows: raise SystemExit("context limit excluded every train or eval target")
 unweighted_train_rows=len(train_rows)
 train_rows=oversample_escalations(train_rows,a.escalation_oversample)
 out=root/"processed"; prefix=a.output_prefix; write(out/f"{prefix}_train.jsonl",train_rows); write(out/f"{prefix}_eval.jsonl",eval_rows)
 manifest={"source_train":str(train_path),"source_eval":str(eval_path),"model_id_for_length_check":a.model_id,"max_length":a.max_length,"train_cases":sorted(train_cases),"eval_cases":sorted(eval_cases),"train_rows":len(train_rows),"unweighted_train_rows":unweighted_train_rows,"eval_rows":len(eval_rows),"escalation_oversample_train_only":a.escalation_oversample,"train_target_actions":dict(Counter(x["target_action"] for x in train_rows)),"eval_target_actions":dict(Counter(x["target_action"] for x in eval_rows)),"excluded_train_targets":train_excluded,"excluded_eval_targets":eval_excluded,"case_overlap":sorted(train_cases&eval_cases),"loss":"assistant-only; train_qlora.py masks user and system tokens"}
 (out/f"{prefix}_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"train_rows":len(train_rows),"eval_rows":len(eval_rows),"excluded_train":len(train_excluded),"excluded_eval":len(eval_excluded),"train_actions":manifest["train_target_actions"],"eval_actions":manifest["eval_target_actions"]},indent=2))
if __name__=="__main__": main()
