# Research-only demo backend

`demo_service` is the text-first HTTP boundary for synthetic,
standardized-patient demonstrations. It does not accept PHI, make diagnoses,
recommend treatment, or support clinical use. Every response includes the same
research-only disclaimer.

## Configuration and launch

The service reads credentials and deployment identity only from server-side
environment variables. The current deployed demo identity is documented here
for local configuration, not embedded as a credential:

```text
BASETEN_MODEL_ID=qvmvdojq
BASETEN_DEPLOYMENT_ID=qvxmvrr
BASETEN_ENVIRONMENT=production
BASETEN_API_KEY=<server-side secret>
```

Run it from the repository root:

```bash
.venv/bin/python -m demo_service --env-file .env --host 127.0.0.1 --port 8000
```

The server reuses `baseten_demo.BasetenClient` for the documented
OpenAI-compatible production endpoint:

```text
https://model-{model_id}.api.baseten.co/environments/{environment}/sync/v1/chat/completions
```

The exact Baseten interface citations are maintained in
[`baseten_demo/BASETEN_DOCS.md`](../baseten_demo/BASETEN_DOCS.md). The API key
is sent only by the server as a Bearer credential and is never included in an
API response or instrumentation event.

## HTTP contract

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /demo/start` | none | `session_id`, `state: "waking"`, elapsed time, disclaimer |
| `GET /demo/status/{session_id}` | none | current `waking`, `ready`, `error`, `ended`, or `expired` state and elapsed time |
| `POST /demo/turn/{session_id}` | `{"patient_text":"synthetic scenario text"}` | one concise non-diagnostic model turn plus TTFB/duration |
| `POST /demo/end/{session_id}` | none | closed session state and disclaimer |

Representative payloads (every response also includes `research_only: true`
and `disclaimer`):

```json
POST /demo/start                 -> {"session_id":"…","state":"waking","elapsed_seconds":0.0}
GET /demo/status/{session_id}    -> {"session_id":"…","state":"ready","turns":0}
POST /demo/turn/{session_id}     <- {"patient_text":"synthetic scenario text"}
                                -> {"state":"ready","turn":{"text":"…","time_to_first_token_ms":12.0,"duration_ms":25.0}}
POST /demo/end/{session_id}      -> {"session_id":"…","state":"ended"}
```

`/demo/turn` returns HTTP 503 with its current state until readiness has passed
the documented deployment health signal and a trivial inference probe. It never
silently waits for a cold start. A session retains at most 40 turns and an
estimated 6,000 history tokens; patient text is bounded to 4,000 characters.

`POST /demo/start` initiates the shared server-side wake once. The background
readiness worker reuses `baseten_demo.Harness`, so the HTTP path and measurement
harness share the same health-signal-plus-probe contract. Ending a session emits
last-activity evidence and lets Baseten's configured 300-second idle policy
scale the single L4 replica to zero.

## Instrumentation

Events append to `outputs/baseten_demo/demo_service_events.jsonl`. They contain
bounded lifecycle metadata only: session identifiers, state, timings, character
counts, and token estimates. They never contain API keys, patient text, or full
conversation history. The event writer enforces a 4096-byte line limit.

The service emits `session_started`, `wake_requested`, `first_inference`,
`replica_ready`, `session_ready`, `turn_requested`, `inference_result`,
`last_activity`, `session_ended`, and explicit `error` events as applicable.

## Live measurement status

The one authorized cold-start attempt on 2026-09-07 began from
`SCALED_TO_ZERO` and recorded its wake request, but produced no valid headline
latency numbers: its first continuation probe returned HTTP 500 because the
model reported itself unhealthy/not ready. No retry, extra wake, or extra
inference was made. See [the deployment record](baseten-demo-deployment.md)
for the evidence and the required approval boundary for any follow-up.
