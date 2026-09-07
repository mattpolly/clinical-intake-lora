# Clinical-intake capability TODO

This document records the next evaluation and capability work for the research-only QLoRA prototype. It applies to every candidate adapter; it is not a diagnostic-accuracy benchmark and must not be used for clinical claims.

## Current state

- The source-grounded simulated-intake evaluator exists at `scripts/evaluate_simulated_intakes.py` and covers held-out headache, chest-discomfort, and back-pain scenarios.
- The strict baseline benchmark is saved as `outputs/simulated_intake_evaluation_mixed_v1_v4.json`; its protocol is in `docs/evaluation_protocol.md`.
- The expanded 37/25 corpus was trained as `outputs/qwen3_1.7b_medsp_mixed_v2_r8_e8` and compared against the current adapter. It was not promoted; `outputs/qwen3_1.7b_medsp_mixed_v1_r8_e8` remains the CLI default.

## 1. Red-flag handling

- [x] Extract an explicit chest-discomfort urgent-evaluation criterion from the MedSP evaluator guide, plus a clearly labeled exploratory headache trigger from the standardized-patient source.
- [ ] Keep the target behavior limited to an appropriate urgent follow-up or escalation recommendation; do not train or score definitive diagnosis or treatment.
- [x] Add the current triggers/action criteria to the held-out simulator with case IDs, source paths, and separate source-explicit versus exploratory scoring.
- [ ] Score whether the model responds appropriately after the trigger is revealed, rather than before it is revealed.

**Acceptance check:** Results report red-flag handling separately for every scenario, including the exact source-backed trigger and the model response.

## 2. Patient-fact fidelity

- [x] Expand the patient-state simulator so it reveals only facts supported by the source and only after the corresponding question/turn.
- [x] Maintain a turn-by-turn record of facts revealed to the model.
- [x] Flag assistant assertions of patient facts that have not been revealed; distinguish an unsupported assertion from a question about a fact.
- [x] Manually inspect the first strict baseline's flags and add transparent markers for the observed unsupported-summary patterns.

**Acceptance check:** Each flag identifies the response turn, asserted fact, and the revealed-fact state that made it unsupported.

## 3. Context, coverage, and repetition

- [x] Map selected source history items to question categories (for example: onset, character, associated symptoms, and relevant red flags) without requiring a mechanical checklist order.
- [x] Track which mapped categories have already been answered in each simulated interview.
- [ ] Detect near-duplicate questions after a fact is already known; exact repeats and revisits of a covered mapped item are already reported.
- [ ] Flag questions that are irrelevant to the current complaint/scenario; manually review early heuristics rather than treating them as ground truth.
- [ ] Retain the one-concise-question-at-a-time metric.

**Acceptance check:** Per-case output shows covered items, repeated items, uncovered high-value items, and any relevance flags.

## 4. Faithful structured summaries

- [x] Supply the model a completed, simulated history containing only revealed source-backed facts.
- [x] Prompt for a fixed, non-diagnostic structured summary format (chief complaint, relevant history dimensions, associated symptoms/red flags, medications, allergies, PMH, social history, and escalation status when known).
- [x] Score fields against the known revealed facts for coverage and unsupported additions; contradiction detection remains a manual-review check.
- [x] Check whether the summary uses "not obtained" for information that was not revealed.

**Acceptance check:** Results provide field-by-field expected versus produced facts and separately count omissions, unsupported additions, and contradictions.

## 5. Repeatable held-out evaluation gate

- [x] Run `evaluate_simulated_intakes.py` for the current best adapter and save the baseline benchmark artifact.
- [x] Train the expanded 37/25 adapter with the existing memory-conservative QLoRA settings and preserve its logs, adapter, and seed.
- [x] Run the generic behavioral evaluator and the simulated held-out benchmark for both the existing and expanded adapters using identical prompts/settings.
- [x] Add a short, fixed manual-review protocol for qualitative checks that the automated rubric cannot reliably judge.
- [x] Define an explicit comparison table and retain the current CLI adapter because the expanded candidate regressed on repetition and summary coverage despite improvements in narrow fact-fidelity flags.
- [ ] Record dataset version, adapter path, version of evaluation scripts, seed, generation settings, and timestamp with every result.

**Acceptance check:** Every future adapter has one reproducible result artifact from the held-out benchmark plus the manual-review record before it is used as the CLI default.

## 6. Investigate activated-LoRA (aLoRA) voice-latency architecture

This is a research architecture investigation, not a replacement for the
current CLI. It avoids a safety-sentinel false-negative gate by running intake
and safety interpretations from one shared, base-only conversation cache.

```text
persistent base-only KV cache + new patient text
        -> base-Qwen incremental prefill
        -> clone cache for two aLoRA branches
           -> intake aLoRA speculative response
           -> safety aLoRA: PASS / ASK / EMERGENCY
        -> arbitration
        -> encode the selected visible response through base Qwen
        -> updated persistent base-only KV cache
```

- [ ] Verify aLoRA semantics on the local Qwen3-4B / Transformers 4.57.6 /
  PEFT 0.20.0 / 4-bit BitsAndBytes stack before designing a serving path.
- [ ] Derive stable, hidden, tokenizer-native invocation sequences for the
  intake and safety adapters. They must appear after the shared history, remain
  identical in training and inference, and not be user-controllable.
- [ ] Build a cache-correctness microtest: base-prefill a shared history,
  clone the cache, invoke either aLoRA, and prove no branch mutates the other
  branch or the canonical base cache.
- [ ] Prove that post-invocation aLoRA KV entries are never reused by the base
  model or the other aLoRA. Re-encode only the selected visible assistant text
  with base Qwen to advance the persistent canonical cache.
- [ ] Train matched conventional-LoRA and aLoRA safety adapters on identical,
  source-backed progressive safety sequences. The central scientific question
  is whether aLoRA can make correct safety decisions while interpreting a
  history encoded entirely by base Qwen.
- [ ] Keep `processed/acuity_transition_probe_v1.jsonl` frozen and compare
  conventional LoRA versus aLoRA on transition timing, ASK loops, premature
  escalation, and emergency recall.
- [ ] Benchmark four actual paths on the target runtime: conventional serial
  routing, aLoRA sequential branches, mixed-batch aLoRA branches, and a
  full-reprefill baseline. Record first-token latency, decision latency, total
  response latency, VRAM, and cache-copy cost.
- [ ] Treat mixed-batch execution as a performance hypothesis, not proof of
  true GPU parallelism. Verify adapter-specific rows, cache isolation, and
  Qwen/4-bit compatibility before making latency claims.
- [ ] For voice, buffer ordinary intake audio until the safety branch has
  returned its decision. A 100–200 ms target is an optimization goal, never a
  substitute for a completed safety veto.
- [ ] Keep the harness mechanical: it validates structured safety output and
  performs cache/adapter operations, but contains no symptom rules, diagnosis,
  or treatment logic.

**Acceptance check:** aLoRA is only considered for the research CLI if cache
correctness passes, the safety adapter is not materially worse than matched
ordinary LoRA on the frozen transition benchmark, and measured voice latency
improves on the intended deployment runtime.

## Later data work

- [ ] Expand direct source-derived patient-question/answer pairs where MedSP1000 contains them; prefer these over manually authored examples.
- [x] Verify that Qwen3-4B QLoRA fits at 512 tokens on the RTX 2060 Super (one-epoch pilot: 5,278.2 MiB peak allocated); retain its non-promoted artifacts in `outputs/qwen3_4b_pilot_comparison.md`.
- [ ] Build a 100+ example, source-grounded training set before another 4B run, including training-only examples of concise turn endings, relevant red-flag escalation, fact-faithful summaries, and no-diagnosis behavior.
- [ ] Continue keeping train/evaluation splits separated by whole source case and avoid near-duplicate source documents across splits.
- [ ] Consider an additional dataset only after documenting the precise coverage gap it fills and checking its license, provenance, and compatibility with the non-diagnostic interview objective.
