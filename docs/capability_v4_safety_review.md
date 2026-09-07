# v4 safety-capability review

This is a research benchmark result, not a clinical safety claim.

## Data and run

- Compact capability data: 20 source-backed records across 16 whole scenarios;
  the train/evaluation case sets do not overlap.
- New safety tranche: four train and four held-out records, including immediate
  and delayed disclosures. The detailed pair design is in
  [safety_contrast_pairs.md](safety_contrast_pairs.md).
- SFT export: 53 unique training targets; only escalation targets were sampled
  four times, producing 71 training rows. Held-out export was unweighted.
- Model: `Qwen/Qwen3-4B`, 4-bit NF4 QLoRA, rank 8, fp16 compute, context 1024,
  batch 1, accumulation 4, seed 42, one epoch.
- Training result: train loss 2.4405; eval loss 1.5784; peak recorded VRAM
  5503.1 MiB; adapter reload PASS.

## Held-out results

The post-training run used the identical v4 evaluation scenarios, generation
settings, and state simulator as the pre-training run. Delayed cases first
receive only a source-backed patient answer after the model emits its follow-up;
the subsequent response is then separately scored for escalation.

| Measure | Existing v3 adapter | v4 safety adapter |
| --- | ---: | ---: |
| Escalation on escalation events | 0/6 (0%) | 1/6 (16.7%) |
| Correct safety follow-up or escalation across all safety events | 3/9 (33.3%) | 4/9 (44.4%) |
| One-question rate (free-history records) | 100% | 100% |
| Automated hidden-fact leaks | 0 | 0 |
| Diagnostic-language flags | 0 | 0 |
| Summary revealed-fact coverage | 27.3% | 66.7% |
| Exact repeated free questions | 2 | 4 |
| Unmatched questions needing manual review | 4 | 6 |

## Decision

Do not promote `outputs/qwen3_4b_capability_v4_safety_r8_e1_1024` to the
default CLI adapter. The v4 result demonstrates a measurable safety and
summary improvement, but 1/6 escalation is not sufficient and the question
quality flags regressed. Retain it as an experimental comparison checkpoint.

The next targeted iteration should add several more source-backed *train*
cases with delayed acute triggers, but keep the held-out five-scenario safety
set frozen; then test a lower escalation weight (for example 2x) alongside the
current 4x weight to avoid crowding out focused question behavior.
