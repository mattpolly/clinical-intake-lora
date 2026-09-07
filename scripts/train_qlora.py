#!/usr/bin/env python3
"""Run a deliberately small assistant-only QLoRA experiment on MedSP1000 data."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from model_utils import cached_model_path


MODEL_ID = "Qwen/Qwen3-1.7B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assistant_content_spans(rendered: str) -> list[tuple[int, int]]:
    """Return character spans for assistant content in Qwen's rendered chat text."""
    start_marker = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = rendered.find(start_marker, cursor)
        if start < 0:
            break
        content_start = start + len(start_marker)
        content_end = rendered.find(end_marker, content_start)
        if content_end < 0:
            raise ValueError("Unterminated assistant message in rendered chat")
        # Include Qwen's explicit end-of-turn token in the target.  Without it,
        # the model is taught question text but never receives a loss signal to
        # stop its assistant turn before producing another question.
        spans.append((content_start, content_end + len(end_marker)))
        cursor = content_end + len(end_marker)
    if not spans:
        raise ValueError("Conversation has no assistant messages")
    return spans


def tokenize_assistant_only(tokenizer, messages: list[dict[str, str]], max_length: int) -> dict[str, list[int]]:
    """Tokenize chat messages and mask loss everywhere except assistant content.

    Qwen3's current chat template does not expose generation masks, so this uses
    offsets over the template-rendered text rather than training on user tokens.
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    spans = assistant_content_spans(rendered)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    labels = []
    for token_id, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        in_assistant = any(start >= left and end <= right and end > start for left, right in spans)
        labels.append(token_id if in_assistant else -100)
    if not any(label != -100 for label in labels):
        raise ValueError("Truncation removed every assistant target token")
    return {"input_ids": encoded["input_ids"], "labels": labels}


class ConversationDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not records:
            raise ValueError(f"No records in {path}")
        self.rows = [tokenize_assistant_only(tokenizer, row["messages"], max_length) for row in records]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class AssistantOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{"input_ids": item["input_ids"]} for item in features],
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(batch["input_ids"], -100)
        for row, item in enumerate(features):
            labels[row, : len(item["labels"])] = torch.tensor(item["labels"], dtype=torch.long)
        batch["labels"] = labels
        return batch


def load_base_model(model_id: str, cache_dir: Path):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        cached_model_path(model_id, cache_dir),
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-data", type=Path, help="JSONL training records, relative to project root by default.")
    parser.add_argument("--eval-data", type=Path, help="JSONL evaluation records, relative to project root by default.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this QLoRA experiment")
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output or root / "outputs" / "qwen3_1.7b_medsp_qlora_r8"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "data" / "hf_cache"
    set_seed(args.seed)

    def project_path(path: Path | None, default: Path) -> Path:
        if path is None:
            return default
        return path if path.is_absolute() else root / path

    tokenizer = AutoTokenizer.from_pretrained(cached_model_path(args.model_id, cache_dir), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_path = project_path(args.train_data, root / "processed" / "medsp_intake_train.jsonl")
    eval_path = project_path(args.eval_data, root / "processed" / "medsp_intake_eval.jsonl")
    train_dataset = ConversationDataset(train_path, tokenizer, args.max_length)
    eval_dataset = ConversationDataset(eval_path, tokenizer, args.max_length)

    model = load_base_model(args.model_id, cache_dir)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    trainable_count = sum(count for _, count in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable parameters: {trainable_count:,} / {total_count:,} ({100 * trainable_count / total_count:.3f}%)")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    result = trainer.train()
    metrics = trainer.evaluate()
    peak_vram_mib = torch.cuda.max_memory_allocated() / 1024**2
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Reload from disk in a fresh 4-bit base model to prove the saved adapter is usable.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_base = load_base_model(args.model_id, cache_dir)
    reloaded = PeftModel.from_pretrained(reloaded_base, output_dir)
    reloaded.eval()
    reloaded_lora = [name for name, _ in reloaded.named_parameters() if "lora_" in name]
    assert reloaded_lora, "Reloaded adapter has no LoRA parameters"
    summary = {
        "model_id": args.model_id,
        "seed": args.seed,
        "eval_steps": args.eval_steps,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "lora_rank": args.rank,
        "train_data": str(train_path),
        "eval_data": str(eval_path),
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "peak_vram_mib": round(peak_vram_mib, 1),
        "train_metrics": result.metrics,
        "eval_metrics": metrics,
        "adapter_reload": "PASS",
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
