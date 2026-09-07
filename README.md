# Clinical Intake LoRA

Local research proof of concept for teaching `Qwen/Qwen3-1.7B` disciplined,
non-diagnostic clinical history-taking with QLoRA. It uses synthetic,
standardized-patient scenarios only.

The initial experiment is intentionally small: inspect MedSP1000, construct a
case-separated supervised dataset, establish a base-model baseline, then train
and reload a small adapter on an 8 GB RTX 2060 Super.

## Environment

Create the project environment once, then install the training dependencies:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/inspect_environment.py
```

`requirements.txt` intentionally does not list PyTorch: this machine already
has the CUDA 12.1 PyTorch build, which the project environment reuses.

The project does not use Ollama for training. Transformers and PEFT are the
initial inference and training path.

## Current experiment

The initial pipeline is complete: five MedSP1000 cases were inspected, then
expanded into 15 case-separated train and 10 evaluation conversations. The
baseline is saved at `outputs/baseline_qwen3_1.7b_4bit.json`.

`processed/medsp_intake_*` is a manually authored, source-grounded pilot. A
separate high-provenance pilot, `processed/medsp_direct_pairs_v1_*`, contains
verbatim evaluator response/question pairs with source line numbers; build it
with `python scripts/build_direct_pair_dataset.py`. See
`docs/medsp_inspection_report.md` for the role mapping and limitations. It is
not yet broad enough to replace the original training set.

The expanded mixed corpus is `processed/medsp_mixed_v1_*` (37 train / 25 eval
records, with 18 / 8 whole source cases). Its manifest documents the separated
case IDs. The 8-epoch expanded-data candidate was retained for comparison but
not promoted because it overfit and regressed on held-out context/summary
checks; see `outputs/adapter_comparison_mixed_v1_vs_v2.md`.

`Qwen/Qwen3-4B` was also verified as a 512-token QLoRA candidate on this GPU
(5.28 GiB peak allocated in a one-epoch run). Its pilot adapter is retained but
not promoted because the small corpus did not reliably enforce one-question
turns; see `outputs/qwen3_4b_pilot_comparison.md`.

The current best adapter is `outputs/qwen3_1.7b_medsp_mixed_v1_r8_e8`. It uses
NF4 4-bit QLoRA, rank 8, fp16, batch size 1, gradient accumulation 4, and a
512-token context. It trained and reloaded successfully within the 8 GB GPU
budget. Behavioral evaluation is saved at
`outputs/adapter_behavior_evaluation_mixed_v1.json`; it remains a research
artifact, not a clinical tool.

The fixed held-out simulated-intake benchmark is documented in
`docs/evaluation_protocol.md`. It measures source-backed fact disclosure,
repetition, red-flag response, and faithful summaries, and it requires a short
manual trace review before an adapter is promoted.

The next capability-focused corpus is intentionally separate from that mixed
corpus. Its compact, source-traceable contract is in
`docs/capability_dataset_schema.md`. Capability v3 has 12 complete,
case-separated records (eight train and four held-out evaluation). Its first
one-epoch 4B QLoRA run improved held-out summary fidelity but did not pass the
red-flag escalation gate; it is not the CLI default. See
`outputs/qwen3_4b_capability_v3_comparison.md`.

Run the reproducible steps after activating the environment:

```bash
python scripts/baseline.py
python scripts/build_mixed_dataset.py
python scripts/train_qlora.py --epochs 8 --train-data processed/medsp_mixed_v1_train.jsonl --eval-data processed/medsp_mixed_v1_eval.jsonl --output outputs/qwen3_1.7b_medsp_mixed_v1_r8_e8
python scripts/evaluate.py --adapter outputs/qwen3_1.7b_medsp_mixed_v1_r8_e8
python scripts/evaluate_simulated_intakes.py --adapter outputs/qwen3_1.7b_medsp_mixed_v1_r8_e8 --output outputs/simulated_intake_evaluation_mixed_v1.json
python scripts/chat.py
```

Build and check the capability pilot separately:

```bash
python scripts/build_capability_pilot.py
python scripts/validate_capability_dataset.py
# Held-out Bill Thompson summary case only (the default):
python scripts/evaluate_capability_pilot.py --adapter outputs/qwen3_1.7b_medsp_mixed_v1_r8_e8
# Explicit pipeline smoke test, including training rows; not a benchmark:
python scripts/evaluate_capability_pilot.py --split all --adapter outputs/qwen3_1.7b_medsp_mixed_v1_r8_e8 --output outputs/capability_pilot_smoke.json
# Build and evaluate the current six-case v1 corpus:
python scripts/build_capability_dataset.py
python scripts/validate_capability_dataset.py processed/medsp_capability_v1_train.jsonl processed/medsp_capability_v1_eval.jsonl
python scripts/evaluate_capability_pilot.py --train-data processed/medsp_capability_v1_train.jsonl --eval-data processed/medsp_capability_v1_eval.jsonl --split eval --adapter outputs/qwen3_1.7b_medsp_mixed_v1_r8_e8 --output outputs/capability_v1_heldout.json
# Current v2 corpus:
python scripts/build_capability_dataset_v2.py
python scripts/validate_capability_dataset.py processed/medsp_capability_v2_train.jsonl processed/medsp_capability_v2_eval.jsonl
# Current v3 capability training export:
python scripts/build_capability_dataset_v3.py
python scripts/build_capability_sft_data.py --max-length 1024 --output-prefix medsp_capability_v3_sft_1024
```
