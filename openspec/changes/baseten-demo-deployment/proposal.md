## Why

The clinical-intake prototype currently runs only on the local WSL2 GPU
(Qwen3-4B). We need an on-demand, hosted demo: a visitor can start a demo
interview without us keeping a warm GPU. Baseten with scale-to-zero gives
$0 idle cost and a wake-on-demand path, and the customer has already provided
a Baseten account and API key. The headline engineering question is the
**actual cold-start latency for Qwen3-8B (+ adapter path)** on a single
A10G-class GPU — measured, never assumed from published generic figures.

## What Changes

- **New `baseten_demo` package** in this repository (deployment definition,
  wake/readiness/inference backend, instrumentation, profiles) — the clinical
  architecture stays independent of hosting details.
- **Declarative model/GPU profiles**: Qwen3-8B default (quantized for
  inference, latency over cheapest VRAM), one 24 GB A10G-class GPU,
  `min_replicas=0`, `max_replicas=1`, idle scale-down configurable ~2–5 min;
  a Qwen3-14B profile switchable later without application redesign.
- **Backend wake → readiness → inference path** that holds the Baseten API
  key server-side only (never browser JavaScript), performs an explicit wake
  call, verifies readiness, and serves inference — the seam a future
  clinicalinterview.com "Start Demo" button calls.
- **Cold-start instrumentation**: wake requested, replica/model ready, first
  inference, time-to-first-token, inference duration, last activity, and
  scale-to-zero (when observable), appended to a JSON-lines artifact.
- **Docs-grounded Baseten integration**: every API/config interface used is
  verified against current Baseten documentation (reachable from this host)
  and cited with its doc URL in a notes file. No remembered API details.
- **LoRA-ready, not LoRA-assumed**: the deployment pre-deploys model/adapter
  artifacts so wake only provisions a replica, and the interface represents
  multiple adapters (aLoRA-compatible), never exactly one ordinary LoRA.

Out of scope: the clinicalinterview.com UI (empty placeholder), telephony,
aLoRA implementation itself, and creating the live deployment/running the real
measurement (an explicit follow-up execution step with the customer's
credentials).

## Capabilities

- `baseten-deployment-profiles` — declarative profiles and the Baseten
  deployment definition, scale-to-zero configuration, LoRA-ready artifact
  layout, doc citations.
- `backend-wake-readiness` — server-side-authenticated wake, readiness
  verification, and inference path with bounded failure behavior.
- `cold-start-instrumentation` — the required event/metric set, JSON-lines
  artifact, and a fully simulated dry-run trace.

## Impact

- New code and tests under `baseten_demo/`; a deployment config file; a
  docs-notes file; README/docs update. No changes to `voice_proto/` semantics
  (its Baseten model backend can later point at the new deployment via
  `BASETEN_MODEL_URL`). Research-only disclaimer carries through: no PHI, no
  diagnosis/treatment behavior, no clinical-use claims.
