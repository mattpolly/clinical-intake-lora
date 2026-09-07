# Clinical Intake LoRA

An inspectable research prototype for adapting local Qwen models to conduct
disciplined, non-diagnostic clinical intake interviews. It explores how QLoRA,
source-grounded standardized-patient data, and behavioral evaluation can improve
the mechanics of history-taking without presenting the system as a diagnostic or
treatment tool.

**Repository:** <https://github.com/mattpolly/clinical-intake-lora>

> **Research only.** This project uses synthetic/standardized-patient scenarios,
> not PHI. It is not validated for clinical use, does not provide diagnoses or
> treatment recommendations, and must not be used to make patient-care decisions.

## What it explores

- QLoRA fine-tuning of `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` with Transformers,
  PEFT, TRL, and BitsAndBytes—not Ollama.
- Interview behavior: one concise question at a time, complaint characterization,
  relevant history, medication/allergy/PMH/social history, and faithful summaries.
- Source-grounded training records derived from role-specific MedSP1000
  standardized-patient material, with complete-case train/evaluation splits.
- Patient-state simulation that reveals source-backed facts only after relevant
  questions, enabling checks for unsupported assertions and repetition.
- Safety research: recognizing when an interview should pause for a focused
  safety question or a non-diagnostic emergency-evaluation message.
- A planned latency-conscious architecture using activated LoRA (aLoRA): a shared
  base-model KV cache forks into intake and safety interpretations, with safety
  retaining veto authority.

## Why this is interesting

Clinical dialogue is not just a generic chat task. A useful intake assistant has
to maintain conversation state, avoid inventing patient facts, ask questions in
a clinically coherent order, and know when routine questioning is no longer the
right action. This project treats those as separate, measurable behaviors rather
than relying on an unstructured “helpful assistant” prompt.

The current work deliberately favors small, reviewable datasets and transparent
evaluation over claims of medical capability. It is an engineering investigation
into training and serving patterns, not a finished clinical product.

## Current results

The project verified a complete 4-bit QLoRA pipeline on an RTX 2060 Super with
8 GB VRAM. A one-epoch Qwen3-4B capability run trained a rank-8 adapter with
0.743% trainable parameters, recorded a 5.50 GiB peak VRAM allocation, and
passed adapter reload.

The experiments also produced a useful negative result: a small free-form
“safety escalation” corpus improved some summaries but did not reliably teach
the model to switch from ordinary interviewing to escalation. The frozen
multi-turn acuity-transition probe showed that mode-transition behavior—not
mere recognition of obvious isolated emergency patterns—is the hard problem.
The resulting architecture and data-design work are documented rather than
promoted as a production safety feature.

## Text demo backend

The repository includes a research-only, text-first HTTP backend for a future
demo UI. It keeps Baseten credentials on the server, wakes the on-demand model,
reports readiness, and returns one concise non-diagnostic intake turn at a time.
It accepts synthetic standardized-patient text only; never send PHI.

```bash
.venv/bin/python -m demo_service --env-file .env --host 127.0.0.1 --port 8000
```

The future UI calls `POST /demo/start`, polls `GET /demo/status/{session_id}`
until it is ready, uses `POST /demo/turn/{session_id}` with
`{"patient_text":"..."}`, then calls `POST /demo/end/{session_id}`. The full
request/response contract, server-side environment values, and research
disclaimer are in [Demo backend](docs/demo_backend.md).

See:

- [Capability v4 safety review](docs/capability_v4_safety_review.md)
- [Acuity transition probe](docs/acuity_transition_probe_v1.md)
- [Acuity-mode curriculum](docs/acuity_mode_curriculum.md)
- [aLoRA investigation TODO](TODO.md#6-investigate-activated-lora-alora-voice-latency-architecture)

## Architecture

### Current local prototype

```text
synthetic patient turn
        ↓
Qwen base model (4-bit) + one PEFT adapter
        ↓
one concise intake question or structured summary
```

The current `scripts/chat.py` is a minimal local terminal CLI. It loads one
adapter at a time and retains bounded multi-turn history.

### Investigated voice-serving architecture

```text
persistent base-only KV cache + new patient text
        ↓
base-Qwen incremental prefill
        ↓
forked aLoRA branches
   ├── intake interpretation → speculative intake response
   └── safety interpretation → PASS / ASK / EMERGENCY
        ↓
mechanical arbitration
        ↓
selected response folded back into the canonical base cache
```

The safety branch is intended to make the decision; application code only
validates its structured result, chooses the visible response, and maintains
cache/adapter state. This design remains an unimplemented research item pending
cache-correctness, Qwen compatibility, latency, and held-out behavior tests.

## Evaluation principles

The project evaluates interviewing behavior, not diagnostic accuracy.

| Behavior | Evaluation approach |
| --- | --- |
| One-question discipline | Count exactly one concise question per free-generation turn. |
| Fact fidelity | Reveal source-backed facts only after appropriate simulated patient answers; flag unsupported claims. |
| Context and repetition | Track covered rubric items and flag repeated/irrelevant questions for review. |
| Summaries | Score field coverage and unsupported additions against the known revealed state. |
| Safety transitions | Test progressive disclosure: continue history, ask a focused safety question, or escalate only when an explicit pattern appears. |

Automated checks are deliberately transparent heuristics and manual-review aids;
they are not clinical safety guarantees.

## Project layout

```text
scripts/
  inspect_environment.py          Verify WSL2, CUDA, PyTorch, and BitsAndBytes
  download_dataset.py             Fetch MedSP1000 source material
  inspect_medsp.py                Inspect role-specific source files
  build_*                         Build reviewable training/evaluation data
  train_qlora.py                  Run memory-conscious QLoRA training
  evaluate_*.py                   Behavioral, simulated-intake, and acuity probes
  chat.py                         Minimal local multi-turn terminal chat
  voice_proto/                    Headless voice-loop prototype (Polly, streaming
                                  Transcribe, mock/Baseten model, Rime TTS)
  baseten_demo/                    On-demand Baseten demo deployment backend
                                  (scale-to-zero Qwen3-8B profiles, server-side
                                  wake/readiness/inference, cold-start events)

docs/
  capability_dataset_schema.md    Compact, source-traceable data contract
  evaluation_protocol.md          Held-out behavioral evaluation protocol
  acuity_*.md                     Safety/transition research and benchmarks
  DEPLOYMENT.md                   Preliminary voice-serving cost model
  voice_prototype.md              Headless voice-loop prototype (AWS + Rime)

processed/                        Generated datasets (ignored by Git)
outputs/                          Adapters and evaluation artifacts (ignored by Git)
data/                             Downloaded source data and Hugging Face cache (ignored by Git)
```

## Local setup

The initial environment was developed in Ubuntu under WSL2 with an NVIDIA RTX
2060 Super (8 GB VRAM). PyTorch is intentionally not pinned in
`requirements.txt`; use a CUDA-compatible build already appropriate for the
host environment.

```bash
git clone https://github.com/mattpolly/clinical-intake-lora.git
cd clinical-intake-lora

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/inspect_environment.py
python scripts/download_dataset.py
python scripts/inspect_medsp.py
```

Before training, inspect the downloaded source material and build only the
small corpus required for the experiment. Dataset downloads, processed data,
model caches, adapters, and other large artifacts are intentionally excluded
from Git.

## Reproduce a small capability experiment

```bash
source .venv/bin/activate

# Build and validate source-grounded compact records.
python scripts/build_capability_dataset_v4.py
python scripts/validate_capability_dataset.py \
  processed/medsp_capability_v4_train.jsonl \
  processed/medsp_capability_v4_eval.jsonl

# Export assistant-target SFT records; weighting applies only to training rows.
python scripts/build_capability_sft_data.py \
  --model-id Qwen/Qwen3-4B \
  --train-data processed/medsp_capability_v4_train.jsonl \
  --eval-data processed/medsp_capability_v4_eval.jsonl \
  --output-prefix medsp_capability_v4_sft_1024_x4 \
  --max-length 1024 \
  --escalation-oversample 4

# Train a deliberately small 4-bit QLoRA run.
python scripts/train_qlora.py \
  --model-id Qwen/Qwen3-4B \
  --max-length 1024 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --rank 8 \
  --train-data processed/medsp_capability_v4_sft_1024_x4_train.jsonl \
  --eval-data processed/medsp_capability_v4_sft_1024_x4_eval.jsonl \
  --output outputs/qwen3_4b_capability_v4_safety_r8_e1_1024
```

Do not promote an adapter based on loss alone. Run the held-out behavioral and
acuity-transition evaluations, retain artifacts, and manually inspect safety
and summary traces first.

## Data and privacy

- MedSP1000 material must be downloaded separately and used according to its
  upstream terms.
- This repository contains scripts and documentation, not source data, PHI,
  trained model weights, or deployment credentials.
- Never enter PHI into the local CLI or save real clinical transcripts.

## Roadmap

Near-term work focuses on a dedicated, source-backed safety-gate curriculum,
larger frozen progressive-disclosure benchmarks, and a cache-correctness/aLoRA
latency prototype. The first headless voice-loop prototype (AWS Transcribe +
pluggable Baseten model + Rime TTS) is documented in
[docs/voice_prototype.md](docs/voice_prototype.md). The on-demand Baseten demo
deployment backend (scale-to-zero Qwen3-8B, server-side wake/readiness/
inference, cold-start instrumentation) is documented in
[docs/baseten-demo-deployment.md](docs/baseten-demo-deployment.md). Detailed
acceptance criteria live in [TODO.md](TODO.md).
