# Baseten on-demand demo deployment (Qwen3-8B, scale-to-zero)

<!-- Agentic Devshop feature: baseten-demo-deployment -->

## Intent

On-demand Baseten deployment of Qwen3-8B with scale-to-zero, server-side wake/readiness/inference, cold-start instrumentation, and configurable model/GPU profiles.

## Requirements

# Baseten on-demand demo deployment (Qwen3-8B, scale-to-zero)

## Intent

Move the clinical-intake prototype from the local Qwen3-4B setup to an
on-demand Baseten demo deployment: Qwen3-8B (quantized for inference) on one
24 GB A10G-class GPU, persistent deployment with scale-to-zero
(min_replicas=0, max_replicas=1, idle scale-down ~2–5 minutes), a backend wake/
readiness/inference path that never exposes the Baseten API key to the browser,
and instrumentation whose headline measurement is the **actual cold-start
latency for Qwen3-8B + adapter(s)** — never published generic figures.

## Customer requirements (as given)

- Start with Qwen3-8B, quantized appropriately for inference.
- Single 24 GB A10G-class GPU initially; autoregressive latency matters more
  than cheapest VRAM.
- Persistent Baseten deployment, scale-to-zero: min_replicas=0, max_replicas=1,
  aggressive idle scale-down approximately 2–5 minutes; no continuously warm
  GPU; effectively $0 GPU cost while scaled to zero.
- Demo lifecycle: user visits clinicalinterview.com → clicks Start Demo → our
  backend calls Baseten's wake mechanism → UI intro/setup while GPU becomes
  ready → backend verifies readiness → interview begins → ~15-minute session →
  2–5 min inactivity → Baseten scales replica back to zero.
- The Baseten API key must never reach browser JavaScript; Start Demo calls
  our backend, and the backend authenticates to Baseten.
- Model/LoRA artifacts are pre-deployed; waking only provisions/loads the
  inference replica, never rebuilds/redeploys the model.
- Preserve the clinical architecture as independently as practical from
  hosting details. A subsequent aLoRA/base-KV-cache branching architecture is
  possible, so avoid infrastructure choices that assume only one ordinary
  LoRA.
- Instrument at least: wake requested timestamp; replica/model ready
  timestamp; first inference request; time-to-first-token; inference duration;
  last request/activity; scale-to-zero behavior if observable.
- Model/GPU choice must be configurable so Qwen3-14B can be benchmarked
  against 8B later without redesigning the application.
- **Before coding against Baseten APIs/config syntax, check the current
  Baseten documentation** for the exact supported wake, autoscaling,
  readiness, and deployment-configuration interfaces. Do not rely on
  remembered API details. (Host can reach docs.baseten.co and PyPI; the
  official `baseten` Python package is available.)

## Lead-scoped decisions (unless the customer overrides)

1. **UI is out of this slice.** `/srv/www/clinicalinterview.com` is an empty
   placeholder. This slice delivers the backend wake/readiness/inference API
   that a future Start Demo button calls. A minimal demo page is a follow-up.
2. **No adapter artifacts exist on this host.** Trained adapters (1.7B/4B)
   live on the customer's WSL2 machine and are dimensionally incompatible with
   Qwen3-8B. The deployment is built LoRA-ready — adapter(s) pre-deployed and
   loaded at replica start, multiple adapters representable (aLoRA
   compatibility) — but the initial cold-start measurement covers Qwen3-8B
   base + the adapter-load path; an 8B-matched adapter is a training
   follow-up on the WSL2 machine.
3. **Role-based slice builds and validates; live deployment is a bounded
   execution step.** The implement-local worktree flow produces committed,
   evaluator-passed code/config (grounded in current Baseten docs, tested
   against mocks of the documented interfaces). Actually creating the
   Baseten deployment, waking it, and recording live cold-start numbers uses
   the customer's account/credentials and runs afterward as an explicit,
   customer-visible step — implement-local never deploys by design.
4. All Baseten interaction stays server-side (backend service or CLI tool in
   this repo); the API key is read from the local `.env`/environment and is
   never committed or emitted in artifacts/logs.
5. Instrumentation events append to a JSON-lines artifact under `outputs/`
   with the exact fields listed above; bounded, no secrets.

## Constraints and non-goals

- Research-only disclaimer carries through (no PHI, no diagnosis/treatment,
  no clinical-use claims).
- No telephony in this slice; no Amazon Connect; the voice-loop prototype
  (`voice_proto/`) stays as-is, with its Baseten model backend able to adopt
  the new deployment via `BASETEN_MODEL_URL` once it exists.
- No aLoRA implementation in this slice — only LoRA-count-agnostic interface
  design.
- No push/merge of the delivered branch without explicit customer approval.

## Definition of enough (this slice)

- A committed feature branch (from implement-local) containing:
  - a Baseten deployment definition for Qwen3-8B, quantized, one A10G-class
    GPU, min_replicas=0, max_replicas=1, idle timeout configurable in the
    2–5 minute range, per **current documented Baseten interfaces** (cited
    doc URLs recorded in a notes file);
  - a backend module exposing wake → readiness-poll → inference calls,
    authenticating server-side only, suitable for a future Start Demo
    endpoint, with a documented local API (e.g. HTTP or CLI-invocable);
  - an instrumentation harness recording every required timestamp/metric to a
    JSON-lines artifact;
  - a model/GPU configuration profile (8B default, 14B switchable) that
    changes deployment definition without application redesign;
  - tests against mocks of the documented Baseten APIs (no live calls in
    CI); the evaluator passes the acceptance review.
- A follow-up execution plan (not in-slice) for the live deployment + real
  cold-start measurement using the customer's credentials.

## Follow-ups (out of this slice)

- Create the live Baseten deployment; run and record the actual cold-start
  measurement (wake → ready → first token) for Qwen3-8B.
- Train an 8B-matched intake adapter on the WSL2 machine and attach it;
  re-measure cold-start with adapter loading.
- Minimal clinicalinterview.com demo page with the Start Demo button calling
  the backend.
- Qwen3-14B benchmark profile.
- aLoRA base-KV-cache branching architecture (TODO.md track).

## Delivery intent

mvp

## Definition of enough

- A committed feature branch (from `implement-local`) containing:
  - a Baseten deployment definition for Qwen3-8B on one A10G-class 24 GB GPU,
    `min_replicas=0`, `max_replicas=1`, idle scale-down configurable in the
    2–5 minute range, using **current documented Baseten interfaces only**;
  - a backend module exposing wake → readiness-poll → inference, holding the
    Baseten key server-side only, invocable locally (CLI) and ready to sit
    behind a future Start Demo HTTP endpoint;
  - an instrumentation harness emitting every required event (wake requested,
    replica/model ready, first inference, time-to-first-token, inference
    duration, last activity, scale-to-zero when observable) to a JSON-lines
    artifact under `outputs/`;
  - a declarative model/GPU profile config (8B default, 14B switchable
    without application redesign);
  - a docs-notes file citing the exact Baseten doc URL for every API/config
    interface used;
  - tests against mocks of those documented interfaces (no live calls in CI).
- Observable evidence of done: the mock-tested suite passes; a dry-run of the
  harness produces a complete simulated wake → ready → infer → idle event
  trace; the evaluator's independent review passes.
- Creating the live deployment and recording real cold-start numbers is the
  explicit follow-up execution step with the customer's credentials — not
  this slice.

## Decisions and assumptions

Customer-given: Qwen3-8B quantized appropriately; one A10G-class 24 GB GPU;
latency over cheapest VRAM; persistent deployment, min_replicas=0,
max_replicas=1; idle scale-down ~2–5 min; $0 when scaled to zero; key never
in browser; artifacts pre-deployed so wake never rebuilds; hosting-independent
clinical architecture; aLoRA-compatible (no single-LoRA assumptions);
instrumentation list as given; configurable model/GPU for a later 14B
benchmark; check current Baseten docs before coding — never remembered APIs.

Lead decisions:

1. Wake is an **explicit backend call** (per the customer's lifecycle: backend
calls Baseten's wake mechanism), followed by readiness polling; the exact
   documented API is determined during implementation and cited in the notes
   file.
2. Readiness = the documented readiness/health signal **plus one trivial
   inference probe** (which doubles as the first-inference metric).
3. Endpoint access: our backend is the sole caller, authenticating with the
   server-held Baseten key. No additional auth layer this slice; a future
   demo page goes through the backend.
4. Cost guardrails: min=0/max=1 caps concurrent spend; the 2–5 min idle
   timeout bounds the tail; no spend alerting this slice — uptime events make
   cost computable after the fact.
5. Profiles: one declarative config entry per profile (model id,
   precision/quantization, GPU class, idle timeout seconds). Default =
   Qwen3-8B fp16 (latency priority; fits 24 GB with KV cache), idle
   scale-down default 300 s validated to the closed 120–300 s range. A
   concrete `qwen3-14b` benchmark profile ships in the same config file and
   is switchable without code changes; live 14B benchmarking is a follow-up.
6. No cold-start SLA is assumed or asserted — actuals are measured and
   recorded only.
7. Failure behavior: errors are recorded as instrumentation events with
   explicit error states returned to the caller; at most one bounded retry;
   readiness polling terminates in a bounded deadline (see decision 9);
   scale-to-zero observation is best-effort per documented interfaces.
8. Credentials: `BASETEN_API_KEY` from the local `.env` (already provided);
   assume the account's default workspace with no existing-deployment
   constraints; the key is never committed, logged, or emitted in artifacts.
9. Readiness polling is bounded: readiness must be confirmed within a
   configurable deadline (default 600 s / 10 minutes); on deadline expiry
   the backend stops polling, returns an explicit timeout error state, and
   never polls indefinitely.
10. Instrumentation schema: the JSON-lines artifact uses canonical event
    identifiers `wake_requested`, `replica_ready`, `first_inference`,
    `inference_result` (duration, time-to-first-token), `last_activity`,
    `scaled_to_zero` (best-effort), and `error`; each line is a single
    machine-parseable JSON object (stable `event` type + UTC `timestamp`,
    ≤4096 bytes) so wake-to-ready, first-token latency, session length, and
    idle interval are computable from the artifact alone.

Assumptions to verify with the customer if wrong:

- No trained adapter artifacts exist on this host (they live on the WSL2
  machine and are 1.7B/4B, incompatible with 8B). The deployment is built
  LoRA-ready; the initial live cold-start measurement covers the 8B base,
  with an 8B-matched adapter as a training follow-up.
- `clinicalinterview.com` is an empty placeholder: UI work is out of this
  slice; the backend API is what a future Start Demo button calls.

## Score

- Requirements snapshot: `sha256:09c09644dfc386a55d480390bae1a128ef398310fc22e455a2e6d14bac144879`.
- Independent scores:
  - `glm` (fireworks/accounts/fireworks/models/glm-5p3): scope small; complexity medium; ambiguity small; risk small. Scope is a bounded MVP slice: one Baseten deployment definition, one backend wake/readiness/infer module, a JSON-lines instrumentation harness, a declarative profile config, docs citations, and mock-based tests. Live deployment and real cold-start measurement are explicitly deferred, further reducing delivered surface. Complexity is medium: the work must couple to current documented Baseten APIs (wake, readiness, scale-to-zero), manage an async lifecycle (wake → poll → probe → idle), and keep the harness reusable behind a future HTTP endpoint — but CI runs entirely against mocks, removing live-API coupling from the test path. Ambiguity is low: customer constraints are given (GPU class, min/max replicas, idle window, key handling, profile list, event list), and lead decisions resolve the remaining seams (explicit backend wake, readiness plus probe, single-caller auth, error/retry behavior). The one open seam — the exact documented wake/readiness API — is deliberately deferred to implementation with a mandatory docs-citation artifact, which is an acceptable documented-interface lookup rather than a product decision. Risk is small: no live spend in this slice; cost is structurally capped (min_replicas=0, max_replicas=1, 2–5 min idle); the API key is server-held and explicitly excluded from commits, logs, and artifacts; no privacy or safety-sensitive data is processed; the only follow-up risk is that the live step will need customer credentials, which is already scoped out.
  - `qwen` (fireworks/accounts/fireworks/models/qwen3p7-plus): scope medium; complexity medium; ambiguity small; risk small. Scope is medium: six distinct deliverables (deployment definition, backend wake/readiness/inference module, instrumentation harness emitting 8+ event types to JSON-lines, declarative profile config, docs-notes with source URLs, mock-tested suite) but all tightly focused on one feature. Complexity is medium: external Baseten API integration with a wake→readiness-poll→inference flow, bounded retry/error handling, and mock-testing of those external interfaces add technical coupling, but the lead has deferred exact API discovery to implementation time with a 'check docs before coding' guardrail. Ambiguity is small: the Definition of Enough is concrete and observable, lead decisions are explicit on wake semantics, readiness definition (health signal + trivial probe), auth model, cost guardrails, profile structure, failure behavior, and credential handling; assumptions are listed separately. Risk is small: no live deployment this slice, no customer data, spend capped by min=0/max=1, key never leaves server-side, mock-only CI, and no SLA asserted. The package is well-bounded for an MVP with a clear follow-up slice for live cold-start measurement.
- Reconciled score: conservative maximum — scope medium; complexity medium; ambiguity small; risk small.
- Scope: **medium**; complexity: **medium**; ambiguity: **small**; risk: **small**.
- Work size: **medium** (the largest of scope, complexity, and ambiguity).
- Required roles: `implementation`, `evaluator`.

## Plan

- [ ] Plan the smallest dependency-aware tasks with evidence of done.

## Status

Intake

## Implementation log

- Feature created.
- 2026-09-07: implement-local attempt 1 stopped at the spec-author receipt seam: role (deepseek) committed valid contract refinements (worktree commit 21553f9) but its final output was not parseable as the required JSON receipt, and the tool discarded the offending output (no evidence of its shape). Worktree/branch preserved non-destructively as change/baseten-demo-deployment-po-retry1 for diagnosis; devshop receipt-evidence repair planned before retry.
- 2026-09-07: implement-local attempt 2: spec-author committed valid refinements (7924bb5) but prefixed its fenced JSON receipt with one line of prose ('The commit is complete... Here is my receipt.'), which the strict parser rejects. Evidence captured via the unmerged receipt-evidence repair (attempt-2 preserved as change/baseten-demo-deployment-po-retry2). Systematic model behavior, not a one-off: extending the repair to accept a fenced receipt embedded in prose, then retrying.
- 2026-09-07: spec-author-refinement (this worktree): made the contract observable, bounded, and testable — pinned default precision to fp16, shipped a concrete `qwen3-14b` profile, fixed idle scale-down at default 300 s validated to the closed 120–300 s range, bounded readiness polling with a configurable deadline (default 600 s) and explicit timeout, and defined a machine-parseable instrumentation schema (canonical event identifiers + UTC timestamp, ≤4096 bytes per record). Feature-record lead decisions updated to match.

## Follow-ups

- None.
