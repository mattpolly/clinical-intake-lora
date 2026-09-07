# Specification

## Purpose

Defines the instrumentation that measures the deployment lifecycle, with the
actual cold-start latency for Qwen3-8B (+ adapter path) as the headline
measurement. No cold-start SLA is assumed; actuals are recorded.

## ADDED Requirements

### Requirement: Required lifecycle events

The instrumentation SHALL record, with UTC timestamps, at least the following
canonical events, one JSON object per line, appended to a JSON-lines artifact
under `outputs/baseten_demo/`:

- `wake_requested` — UTC timestamp when the backend issues the wake call
- `replica_ready` — UTC timestamp when readiness is proven (health signal +
  trivial probe)
- `first_inference` — UTC timestamp of the first (trivial probe) inference
  request
- `inference_result` — duration and time-to-first-token for each inference
- `last_activity` — UTC timestamp of the most recent inference request
- `scaled_to_zero` — UTC timestamp when scale-down is observed (best-effort)
- `error` — explicit error state when wake/readiness/inference fails

Each event SHALL carry an event identifier and machine-parseable fields so
the wake-to-ready interval, first-token latency, session length, and idle
interval are computable from the artifact alone.

#### Scenario: Complete simulated trace
- **WHEN** a dry-run simulates wake → ready → infer → idle against mocked
  interfaces
- **THEN** the JSON-lines artifact contains every canonical event
  (`wake_requested`, `replica_ready`, `first_inference`,
  `inference_result`, `last_activity`, `scaled_to_zero`) for the simulated
  session, each with the fields its identifier requires

#### Scenario: Canonical event identifiers
- **WHEN** the dry-run produces the JSON-lines artifact
- **THEN** no unknown event identifiers appear in the artifact, and the
  wake-to-ready interval is computable as the difference between the
  `wake_requested` and `replica_ready` timestamps

#### Scenario: Real cold-start session
- **WHEN** a live session wakes the deployment from zero replicas
- **THEN** the wake requested timestamp, replica/model ready timestamp, first
  inference request, time-to-first-token, and inference duration are all
  recorded, and the wake-to-ready interval is computable from the artifact

### Requirement: Machine-readable event schema

Each event SHALL be emitted as one JSON object per line with a stable `event`
type field drawn from the fixed canonical vocabulary (at minimum
`wake_requested`, `replica_ready`, `first_inference`, `inference_result`,
`last_activity`, `scaled_to_zero`, `error`) and a UTC `timestamp` field, so a
consumer can parse and verify a trace without custom logic.

#### Scenario: Trace is machine-verifiable
- **WHEN** a consumer reads the JSON-lines artifact
- **THEN** each line parses as a JSON object carrying an `event` type and a
  UTC `timestamp`, and the read ordering reflects the actual wake → ready →
  infer sequence

### Requirement: Bounded, secret-free events

Event records SHALL be bounded in size — each record no larger than 4096 bytes
of serialized JSON — and MUST NOT contain secret values (the Baseten key or
derivatives) or full conversation transcripts. At most bounded result
summaries or references are recorded.

#### Scenario: Secret scan of artifacts
- **WHEN** the instrumentation artifact is scanned for the configured API key
- **THEN** the key and any derivative do not appear

#### Scenario: Events stay small
- **WHEN** any lifecycle event is emitted
- **THEN** the serialized record is a single JSON object no larger than 4096
  bytes and contains no full conversation transcript

### Requirement: Last-activity and idle tracking

The instrumentation SHALL update a last-activity record on every inference
request, so session length and idle time toward scale-to-zero are computable
after the fact.

#### Scenario: Idle interval is computable
- **WHEN** a session's events are read after the fact
- **THEN** the last-activity timestamp and the (observed or best-effort)
  scale-to-zero event allow the idle interval to be computed
