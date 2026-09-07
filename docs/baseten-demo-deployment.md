# Baseten on-demand demo deployment (`baseten_demo`)

> **Research only.** Synthetic/standardized-patient scenarios, no PHI, no
> diagnosis or treatment behavior, and no clinical-use claim. The deployed
> model is operated only for deliberate, bounded research demonstrations.

This is the hosting layer for an on-demand Baseten demo of the clinical-intake
prototype: Qwen3-8B (fp16) on one L4 24 GiB GPU (`L4:4x16`), a persistent
deployment with scale-to-zero (`min_replica=0`, `max_replica=1`, idle
scale-down ~2–5 minutes), a backend wake → readiness → inference path that
never exposes the Baseten API key to a browser, and cold-start instrumentation
whose headline measurement is the **actual** cold-start latency.

Everything is grounded in the current Baseten documentation; every interface
used is cited in [`baseten_demo/BASETEN_DOCS.md`](../baseten_demo/BASETEN_DOCS.md).

## Package layout

```text
baseten_demo/
  __init__.py               version + research note
  profiles.yaml             declarative model/GPU profiles (default + 14B)
  profiles.py               profile loading + contract validation
  deployment.py             deployment-definition builder (config.yaml,
                            autoscaling payload, endpoints) — no live create
  client.py                 server-side Baseten client (wake/status/infer)
  harness.py                wake -> readiness -> inference orchestration
  mock.py                   mock client for dry-run + CI
  instrumentation.py        JSON-lines event writer (canonical events)
  __main__.py               CLI entry (python -m baseten_demo ...)
  BASETEN_DOCS.md           doc-URL citations + recorded conflicts
```

## Profiles

`profiles.yaml` is versioned and ships two profiles:

- **`qwen3-8b-fp16`** (default): `Qwen/Qwen3-8B`, fp16, `L4:4x16` (24 GiB),
  `min_replicas=0`, `max_replicas=1`, idle scale-down `300 s`.
- **`qwen3-14b`** (benchmark): `Qwen/Qwen3-14B`, fp16, `A100:12x144`
  (80 GiB) — 14B fp16 does not fit 24 GiB VRAM. Same loader path, so switching
  is a config change only.

Profile validation rejects `min_replicas > max_replicas` and idle timeouts
outside the closed `[120, 300]`-second range (approximately 2–5 minutes).

```bash
python -m baseten_demo --deployment --profile qwen3-8b-fp16
python -m baseten_demo --deployment --profile qwen3-14b
```

## Credential boundary

`BASETEN_API_KEY` is read server-side only, from the environment or a local
`.env` (Git-ignored, via `--env-file`). It is sent as an
`Authorization: Bearer` header and never logged, returned, or written to any
artifact. The secret-scan test suite asserts the key and any derivative never
appear in event files, error text, or client `repr`s.

## Instrumentation

Events append one JSON object per line to `outputs/baseten_demo/events_*.jsonl`
(≤ 4096 bytes each, UTC `timestamp`, stable `event` type):

`wake_requested`, `replica_ready`, `first_inference`, `inference_result`
(duration + time-to-first-token), `last_activity`, `scaled_to_zero`
(best-effort), `error`, plus the HTTP service lifecycle events
`session_started`, `session_ready`, `turn_requested`, and `session_ended`.

The wake → ready interval, first-token latency, session length, and idle
interval are all computable from the artifact alone.

## Dry run (no credentials, no network)

```bash
python -m baseten_demo --dry-run \
  --demo-turn --observe-scale-to-zero \
  --poll-interval-seconds 0.05
```

The mock client simulates `SCALED_TO_ZERO → WAKING_UP → ACTIVE` plus the
idle scale-down, and emits a complete simulated event trace.

## Tests

```bash
python -m pytest tests/ -q
```

All Baseten interfaces are exercised against mocks of the documented HTTP
interfaces; there are **no live Baseten calls in CI**.

## Live deployment and measurement record

The current research deployment is model `qvmvdojq`, deployment `qvxmvrr`.
It is configured `min_replica=0`, `max_replica=1`, with the 300-second idle
scale-down policy. Its status was verified as `SCALED_TO_ZERO` before the
single authorized cold-start attempt on 2026-09-07.

That wake is recorded in
`outputs/baseten_demo/events_20260907T210642170Z.jsonl`. The foreground
measurement process was interrupted before it could record readiness; when
the same wake was later continued without issuing another wake, the readiness
probe received HTTP 500: the model reported itself unhealthy and not ready for
predictions. The available deployment-log window had no matching lines. No
second wake, inference retry, or polling cycle was performed.

Consequently there is **no valid cold-start, TTFB, inference-duration, or
idle-to-zero measurement** to report from this attempt. A diagnostic/retry is
an explicit future, cost-bearing operation and requires customer approval.

## Subsequent execution plan

Any further live measurement runs only as an **explicit, customer-visible
step** with the customer's Baseten credentials. Nothing in `baseten_demo`
creates, wakes, or bills a real deployment when run from this repository
without explicit server-side credentials.

1. **Diagnose and re-authorize a cold-start measurement.** Confirm why the
   deployed model was unhealthy, then explicitly approve one bounded retry.
   Export server-side
   `BASETEN_API_KEY`, `BASETEN_MODEL_ID`, `BASETEN_DEPLOYMENT_ID` and run
   `python -m baseten_demo` (live mode). The
   `wake_requested → replica_ready` interval plus the first-inference
   time-to-first-token are the measured cold start.
2. **Attach an 8B-matched intake adapter** (a follow-up on the customer's
   WSL2 machine — the existing 1.7B/4B adapters are dimensionally
   incompatible) and re-measure with the adapter-load path.
3. **Qwen3-14B benchmarking** via the `qwen3-14b` profile.
4. **clinicalinterview.com demo page** calling this backend (UI is a separate
   slice; the backend seam is what a future "Start Demo" button calls).
5. **aLoRA** base-KV-cache branching stays on the
   [`TODO.md`](TODO.md) aLoRA track; the adapter-slot list here is
   deliberately aLoRA-compatible.
