# Specification

## Purpose

Defines the instrumentation that measures the deployment lifecycle, with the
actual cold-start latency for Qwen3-8B (+ adapter path) as the headline
measurement. No cold-start SLA is assumed; actuals are recorded.

## ADDED Requirements

### Requirement: Required lifecycle events

The instrumentation SHALL record, with UTC timestamps, at least: wake
requested; replica/model ready; first inference request; time-to-first-token;
inference duration; last request/activity; and scale-to-zero behavior when
observable. Events SHALL be appended to a JSON-lines artifact under
`outputs/`.

#### Scenario: Complete simulated trace
- **WHEN** a dry-run simulates wake → ready → infer → idle against mocked
  interfaces
- **THEN** the JSON-lines artifact contains events covering every required
  measurement for the simulated session

#### Scenario: Real cold-start session
- **WHEN** a live session wakes the deployment from zero replicas
- **THEN** the wake requested timestamp, replica/model ready timestamp, first
  inference request, time-to-first-token, and inference duration are all
  recorded, and the wake-to-ready interval is computable from the artifact

### Requirement: Bounded, secret-free events

Event records SHALL be bounded in size and MUST NOT contain secret values
(the Baseten key or derivatives) or full conversation transcripts. At most
bounded result summaries or references are recorded.

#### Scenario: Secret scan of artifacts
- **WHEN** the instrumentation artifact is scanned for the configured API key
- **THEN** the key and any derivative do not appear

### Requirement: Last-activity and idle tracking

The instrumentation SHALL update a last-activity record on every inference
request, so session length and idle time toward scale-to-zero are computable
after the fact.

#### Scenario: Idle interval is computable
- **WHEN** a session's events are read after the fact
- **THEN** the last-activity timestamp and the (observed or best-effort)
  scale-to-zero event allow the idle interval to be computed
