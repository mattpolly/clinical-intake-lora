# Specification

## Purpose

Defines the server-side wake → readiness → inference path that a future
clinicalinterview.com "Start Demo" button calls. The Baseten API key never
reaches browser JavaScript; the backend is the sole authenticating caller.

## ADDED Requirements

### Requirement: Server-side credential boundary

The Baseten API key SHALL be read only from server-side environment/.env by
the backend. It MUST NOT appear in any client-visible response, artifact, log
line, test fixture, or committed file.

#### Scenario: Key never leaves the backend
- **WHEN** any wake, readiness, or inference call is made
- **THEN** authentication uses the server-held key and no response, event, or
  artifact contains the key or a derivative of it

### Requirement: Explicit wake and readiness verification

The backend SHALL perform an explicit wake request to the Baseten deployment
using the documented wake/scale interface, then poll readiness. Readiness
SHALL be considered proven only when the documented readiness/health signal
is satisfied AND one trivial inference probe succeeds.

#### Scenario: Start Demo triggers an explicit wake
- **WHEN** a demo session is requested while the deployment is scaled to zero
- **THEN** the backend issues an explicit wake request and records the wake
  requested timestamp before any readiness polling

#### Scenario: Readiness requires a real probe
- **WHEN** the documented health signal reports ready
- **THEN** readiness is not declared until a trivial inference request also
  succeeds, and the probe is recorded as the first inference request

### Requirement: Bounded failure behavior

Transient failures during wake or readiness SHALL receive at most one bounded
retry. Readiness polling SHALL terminate within a bounded, configurable
deadline and MUST NOT poll indefinitely. Persistent failures SHALL surface as
explicit error states to the caller with the failure recorded as an
instrumentation event. The backend MUST NOT retry indefinitely or silently
absorb errors.

#### Scenario: Wake request fails transiently
- **WHEN** a wake request fails with a transient error
- **THEN** the backend retries at most once and then returns an explicit
  error state if it fails again

#### Scenario: Deployment unavailable
- **WHEN** Baseten is unreachable or the deployment errors persistently
- **THEN** the caller receives an explicit error state and an error
  instrumentation event is recorded

#### Scenario: Replica never becomes ready
- **WHEN** readiness is not confirmed within the configured deadline
- **THEN** the backend stops polling and returns an explicit timeout error
  state, and an error instrumentation event is recorded

### Requirement: Local invocability

The wake → readiness → inference path SHALL be invocable locally (CLI) with
mockable Baseten interfaces, so the full path is testable without live
Baseten calls.

#### Scenario: Dry-run of the full path
- **WHEN** the backend path is run locally against mocked Baseten interfaces
- **THEN** wake, readiness, and inference all execute and complete without
  any live Baseten call
