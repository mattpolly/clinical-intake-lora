## 1. Documentation research (gate for all Baseten coding)

- [ ] 1.1 Fetch current Baseten docs for: wake/scale-from-zero mechanism,
  autoscaling policy configuration (min/max replicas, idle timeout),
  readiness/health interface, deployment configuration interface, model
  serving/inference endpoint, adapter/LoRA artifact deployment
- [ ] 1.2 Record every interface used with its doc URL in
  `baseten_demo/BASETEN_DOCS.md`; flag any conflict with repository
  assumptions

## 2. Profiles and deployment definition

- [ ] 2.1 Create `baseten_demo/profiles.yaml` with the default
  `qwen3-8b-fp16` profile (fp16 precision, A10G-class 24 GB, min_replicas=0,
  max_replicas=1, idle timeout default 300 s, adapter slots as a list) and a
  concrete `qwen3-14b` profile
- [ ] 2.2 Implement profile loading/validation: reject min_replicas >
  max_replicas and idle timeout outside the closed 120–300 s range
  (config-only profile switching)
- [ ] 2.3 Implement the Baseten deployment-definition builder per the
  documented configuration interface (no live deployment creation in this
  slice)

## 3. Wake / readiness / inference backend

- [ ] 3.1 Implement the server-side Baseten client (key from env/.env only;
  never logged, never in artifacts)
- [ ] 3.2 Implement explicit wake call per documented interface
- [ ] 3.3 Implement readiness polling: documented health signal + one
  trivial inference probe, terminating in the bounded readiness deadline
- [ ] 3.4 Implement inference call (OpenAI-compatible or documented serving
  interface) returning text + timing (time-to-first-token, duration)
- [ ] 3.5 Implement bounded failure behavior: at most one retry, bounded
  readiness deadline (default 600 s) with explicit timeout, error
  instrumentation events
- [ ] 3.6 CLI entry (`python -m baseten_demo ...`) covering wake → readiness
  → inference with a `--dry-run` mock mode

## 4. Instrumentation

- [ ] 4.1 Implement JSON-lines event writer under `outputs/baseten_demo/`
  with all required events (wake_requested, replica_ready, first_inference,
  inference_result, last_activity, scaled_to_zero best-effort, error); each
  line one JSON object with `event` type and UTC `timestamp` field
- [ ] 4.2 Ensure events are bounded (≤4096 bytes) and secret-free
  (key-scan test)

## 5. Tests and evidence

- [ ] 5.1 Unit tests: profile loading/validation (idle timeout 120–300 s,
  out-of-range rejection, min>max rejection), deployment definition builder
  (fp16 default, min_replicas=0, max_replicas=1, 14B profile switch with no
  code change, no build step), event writer (canonical identifiers, schema,
  size bound)
- [ ] 5.2 Integration tests against mocks of the documented Baseten
  interfaces (no live calls in CI)
- [ ] 5.3 Dry-run evidence: full simulated wake → ready → infer → idle trace
  in a JSON-lines artifact (asserts canonical event types, order, and size
  bound)
- [ ] 5.4 Secret scan test over artifacts and fixtures

## 6. Docs

- [ ] 6.1 Update `docs/` and README with the new package, the profile system,
  and the follow-up live-deployment/cold-start-measurement execution plan
  (explicit customer-visible step; this slice deploys nothing live)
