# Long-term proposal: latency-conscious clinical-intake demo

> **Research only.** This is a technical research and demonstration roadmap
> using synthetic/standardized-patient scenarios. It is not a clinical service,
> makes no diagnosis or treatment claims, and must not process PHI.

## Purpose and current MVP boundary

This repository is evolving a local clinical-intake research prototype into an
on-demand cloud demo. The immediate goal is a small, inspectable proof of
concept: a scale-to-zero, text-first Baseten demo with server-held credentials,
bounded sessions, research-only safeguards, instrumentation, and mock-tested
backend behavior.

The longer-term direction below must not expand the MVP prematurely. In
particular, voice, aLoRA cache branching, production telephony, VPC placement,
and trained cloud adapters are separate experiments that require evidence
before they become delivery commitments.

## Local research baseline

The local prototype uses frozen, 4-bit `Qwen3-4B` with separate clinical-intake
and safety-gate LoRA adapters. Python mechanically maintains bounded
conversation history; it contains no symptom rules or clinical diagnosis logic.

- The intake adapter asks one concise, context-aware question at a time;
  supports structured history-taking, medications, allergies, PMH, and social
  history; and creates faithful summaries without diagnosing or prescribing.
- The safety adapter recognizes potentially life-threatening patterns and emits
  structured `PASS`, `ASK_SAFETY_QUESTION`, or `EMERGENCY` outcomes.
- `EMERGENCY` selects a fixed, reviewed emergency-evaluation message rather
  than model-generated treatment advice.

This two-pass design is conceptually useful but expensive for voice latency:
ordinary turns currently require two full-history prefills.

## Cloud model and deployment direction

Do not treat local Qwen3-4B as the permanent demo model merely because it fits
the local hardware. Initial comparable candidates remain within Qwen so capacity
can be measured without simultaneously changing the model family, tokenizer,
prompts, and training method:

1. Qwen3-8B — first candidate.
2. Qwen3-14B — quality challenger.
3. Existing Qwen3-4B — baseline.

LoRA weights do not transfer between model sizes, but datasets, evaluation
methodology, and behavioral contracts do. Model/GPU profiles must therefore be
configurable rather than embedding a single model choice in the application.

The current on-demand demo uses one scale-to-zero Baseten replica
(`min_replica=0`, `max_replica=1`, 300-second idle policy). Current account
availability, GPU behavior, and pricing are facts to verify before each
benchmark; no remembered A10G/L4 price or availability is authoritative. The
existing deployed proof of concept uses an L4-class configuration. An A10G
benchmark is a candidate only if the account makes it available and measured
latency justifies it.

## Preferred research architecture: base cache plus aLoRA branches

The preferred latency experiment is PEFT Activated LoRA (aLoRA). A branch may
reuse a **base-model-only** KV prefix cache; after an adapter activates, its KV
entries are adapter-specific and must never be reused by another adapter or the
base model.

```text
canonical conversation
        ↓
frozen base-Qwen prefill
        ↓
persistent base-only KV cache
        ↓
new finalized patient utterance through base Qwen
        ↓
fork/clone cache
       / \
      /   \
intake    safety
aLoRA     aLoRA
```

The shared conversation is prefetched once through frozen base Qwen. Each
adapter activates only after a fixed, tokenizer-existing invocation sequence
appended after that shared base-only history:

```text
[shared history encoded by base Qwen] + [INTAKE invocation] → intake aLoRA
[shared history encoded by base Qwen] + [SAFETY invocation] → safety aLoRA
```

Training must use precisely the invocation structure used at inference. Do not
casually add vocabulary. Validate actual aLoRA support with the selected Qwen
version, PEFT/Transformers versions, quantization mode, and cache API before
making architectural claims.

### Parallel and speculative safety

The desired voice-latency behavior forks intake and safety from the same base
cache:

```text
                           ┌→ intake aLoRA → speculative response
base-only prefill/cache ───┤
                           └→ safety aLoRA → PASS / ASK / EMERGENCY
```

Safety has veto authority:

- `PASS`: continue the speculative intake response.
- `ASK_SAFETY_QUESTION`: discard normal intake output and deliver the targeted
  safety clarification.
- `EMERGENCY`: cancel normal output and deliver the fixed reviewed message.

Voice may use a short output/TTS holdback (roughly 100–200 ms, subject to
measurement) so a safety veto can arrive before normal speech is audible. A
barge-in immediately stops TTS, continues STT, and processes the newly
finalized utterance through the same safety/intake path.

Two logical GPU branches are not necessarily physically parallel. Benchmark
sequential execution, batched rows with per-row adapters where supported,
separate CUDA streams only when safe/useful, and separate replicas only if
justified. The measure of success is end-to-end latency, not theoretical
parallelism.

### Canonical cache synchronization

Adapter branch caches are disposable. Once the patient-visible response is
selected, incrementally run its short assistant text through base Qwen:

```text
prior persistent base cache + selected assistant text
        ↓
incremental base-Qwen pass
        ↓
new persistent base-only cache
```

Extend that canonical base cache with each finalized patient utterance before
forking again. Never share an intake-adapter cache with safety or vice versa.
Synchronization may be overlapped with TTS only after cache correctness is
demonstrated.

### Sentinel-head fallback

Also benchmark a sentinel architecture as a fallback:

```text
Qwen + intake LoRA → hidden state → high-recall safety sentinel
                                      ├→ NORMAL: continue intake
                                      └→ SAFETY_REVIEW: run safety LoRA
```

The sentinel is a high-recall tripwire, never the final safety decision-maker.
Its false-negative risk means it is not preferred over parallel safety, but it
may offer a useful latency comparison.

## Safety data and evaluation

Train and evaluate safety behavior independently from ordinary interviewing.
The core unit is a progressive trajectory with a known decision boundary, for
example `PASS → PASS → ASK → ASK → EMERGENCY`, and annotated first warranted
escalation turn `t*`.

Include progressive disclosure, minimal contrast pairs, late danger after
benign context, `ASK → PASS`, `ASK → EMERGENCY`, hard negatives, negation,
historical/hypothetical/third-party statements, corrections, contradictions,
fragmentary language, late answers, assistant-mentioned but patient-denied
symptoms, prompt-like patient text, and multiple clinical-system categories.
Safety conclusions must rest on patient-asserted or confirmed information, not
on dangerous terminology merely appearing in assistant questions.

Evaluate whole conversations rather than only random example accuracy. Key
measures include emergency recall, false negatives, `ASK → EMERGENCY` and
`ASK → PASS` accuracy, ASK-loop rate, premature escalation, hard-negative
specificity, counterfactual consistency, calibration, and transition latency:

```text
transition latency = predicted emergency turn - true boundary t*
```

Split by scenario/source family before generating variants and retain a frozen
transition challenge set.

## Voice remains an I/O layer

Clinical reasoning remains text-based. Raw audio is not initially included in
the Qwen clinical context. Only finalized patient utterances are committed:

```text
provisional STT transcript → do not commit clinical state
finalized patient utterance → append and process
```

The intended eventual stack is:

```text
phone/browser audio
  → AWS telephony or streaming ingress
  → streaming STT
  → AWS orchestration/session service
  → Baseten Qwen inference
  → Rime streaming TTS
  → voice transport
```

Potential AWS pieces include Amazon Connect for PSTN, streaming Transcribe (or
another appropriate STT), and a small ECS/Fargate-style orchestration/session
service. None are part of the current MVP.

Rime is the preferred early TTS candidate because conversational latency and
naturalness matter. Early demos use its hosted API. Only after measuring hosted
TTS latency should we investigate commercial private/VPC/self-hosted or
near-AWS placement, if Rime offers it:

```text
AWS Connect / STT / orchestration / Rime in-or-near AWS
                         ↕
                 primary external hop
                         ↕
                     Baseten
```

That is a later latency optimization, not a claim that Rime is presently
deployed in our VPC.

## Instrumentation and delivery state

Measure every boundary rather than inferring latency. At minimum record speech
start/end, final STT, base-prefill start/end, intake and safety branch starts,
safety result, intake first token/completion, veto/cancel, TTS request/first
audio/completion, and TTS stop on interruption.

Primary UX measures are end-of-speech to final transcript, final transcript to
safety decision, first model token, and first audible assistant audio. Track
interruption-to-TTS-stop and safety-signal-to-normal-output-cancel.

Keep clinical and delivery state separate:

```text
clinical_state: PASS | ASK | EMERGENCY
delivery_state: PENDING | DELIVERED | INTERRUPTED | FAILED
```

The model and safety adapter provide learned behavior. The harness mechanically
maintains state, validates schemas, activates/routes adapters, cancels output,
selects reviewed emergency text, manages caches, and tracks delivery. It must
not grow symptom-rule tables or diagnosis logic. Voice/orchestration owns
telephony, STT/TTS, retries, identity, readiness, and response delivery.

## Ordered research plan

1. Maintain the on-demand Qwen3-8B text demo with complete scale-to-zero
   evidence.
2. Measure cold start, TTFT, throughput, and VRAM on available hardware.
3. Compare 8B, 4B, and 14B against frozen intake/safety scenarios.
4. Recreate/train intake and safety adapters for the selected model size.
5. Validate ordinary multi-adapter PEFT behavior.
6. Build a minimal aLoRA proof: base-prefix reuse, independent branch caches,
   invocation behavior, and quantized-Qwen compatibility.
7. Compare sequential and batched/parallel aLoRA branches.
8. Add safety veto/cancellation and canonical base-cache synchronization.
9. Add voice only after text inference and cache semantics are correct.
10. Add streaming STT/TTS and barge-in; measure speech-end to first audio.
11. Only then optimize provider region/private connectivity/VPC placement.

Before every integration or optimization, verify current official documentation
for PEFT aLoRA, cache copying, mixed-adapter batching, quantization,
Baseten scaling/wake/readiness/GPU availability/pricing, and Rime streaming and
cancellation behavior. When compatibility or performance is uncertain, build a
small isolated benchmark before redesigning the main system.

The optimization objective is: dedicated learned safety behavior, one
long-history base prefill, speculative intake generation, parallel safety veto,
very low time-to-first-audio, and scale-to-zero economics while the demo is
unused.
