## Context

The repository already contains a working headless voice-loop prototype
(`voice_proto/`: Polly fixture → streaming Transcribe → pluggable
`ModelBackend` (mock | Baseten) → Rime TTS) and a committed feature record for
this work (`features/baseten-demo-deployment.md`). The host is headless, has
no GPU, and can reach docs.baseten.co and PyPI (official `baseten` package
v0.10.0 is installable). The Baseten API key exists in the local `.env`
(Git-ignored). No trained adapter artifacts exist on this host; existing
adapters (1.7B/4B) live on the customer's WSL2 machine and are
dimensionally incompatible with 8B, so the deployment is LoRA-ready while the
initial live measurement covers the 8B base model.

## Technical decisions

1. **Docs first.** Before any Baseten API or config is coded, fetch and read
   the current Baseten documentation (wake/scale-to-zero, autoscaling
   policies, readiness/health, deployment configuration, model serving
   endpoints). Record every interface used with its doc URL in
   `baseten_demo/BASETEN_DOCS.md`. If the documented interface conflicts with
   anything in the feature record, the documented interface wins and the
   conflict is reported back.
2. **Package layout.** New top-level package `baseten_demo/` (deployment
   definition, profiles, wake/readiness/inference client, instrumentation,
   CLI entry), plus `baseten_demo/profiles.yaml` for declarative profiles.
   The clinical conversation logic stays where it is; `baseten_demo` is a
   hosting layer.
3. **Profiles.** One entry per profile: model id, precision/quantization, GPU
   class, min/max replicas, idle timeout (integer seconds), adapter slots
   (list, default empty; aLoRA-compatible — no single-LoRA assumption).
   Default profile: `qwen3-8b-fp16` on an A10G-class 24 GB GPU, fp16
   precision, `min_replicas=0`, `max_replicas=1`, idle scale-down default
   300 s validated to the closed 120–300 s range (2–5 min). A concrete
   `qwen3-14b` benchmark profile ships alongside the default and is
   switchable without code changes (live 14B benchmarking is a follow-up).
   fp16 default because autoregressive latency matters more than cheapest
   VRAM and 8B fp16 fits 24 GB with KV cache; precision/quantization is
   per-profile configurable.
4. **Wake path.** Explicit backend wake call (customer's lifecycle: Start
   Demo → backend calls Baseten's wake mechanism), then readiness polling:
   the documented readiness/health signal **plus one trivial inference probe**
   (which doubles as the first-inference metric). Readiness polling is
   bounded: finite, spaced polls terminate in success or an explicit timeout
   error once a configurable deadline (default 600 s) elapses — never an
   unbounded poll loop. At most one bounded retry on transient failure;
   errors become explicit error states, never silent retries.
5. **Credential boundary.** The Baseten key is read from environment/`.env`
   server-side only. It is never embedded in any artifact, log line, test
   fixture, or client-visible response. No browser-facing code in this slice
   exists at all.
6. **Instrumentation.** Append-only JSON-lines artifact under
   `outputs/baseten_demo/events_*.jsonl`; one event per lifecycle step with
   a canonical identifier and UTC timestamp: `wake_requested`,
   `replica_ready`, `first_inference`, `inference_result` (duration,
   time-to-first-token), `last_activity`, `scaled_to_zero` (best-effort if
   observable), and `error` (explicit failure state). Each line is a single
   JSON object with a stable `event` type field and a UTC `timestamp` field,
   never exceeding 4096 bytes; no secrets; no full transcripts.
7. **Testing.** Unit/integration tests mock the documented Baseten interfaces
   (HTTP/transcript fixtures); no live Baseten calls in CI. A dry-run mode
   simulates wake → ready → infer → idle end-to-end and must emit a complete
   event trace.
8. **Live execution is a separate step.** This slice delivers everything
   needed to create and measure the deployment; actually creating it on the
   customer's account and recording real cold-start numbers happens afterward
   as an explicit customer-visible execution step (implement-local never
   deploys by design).

## Risks / non-goals

- Baseten interface details are intentionally not assumed here; the docs
  check is a task gate, not a formality.
- No SLA is asserted for cold start; the point is to measure it.
- No aLoRA implementation; only the adapter-slot abstraction.
- The existing `voice_proto/` behavior is unchanged.
