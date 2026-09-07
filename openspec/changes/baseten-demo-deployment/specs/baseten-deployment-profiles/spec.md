# Specification

## Purpose

Defines the on-demand Baseten demo deployment for the clinical-intake
prototype: declarative model/GPU profiles, a scale-to-zero deployment
definition grounded in current documented Baseten interfaces, and a
LoRA-ready artifact layout that does not assume a single ordinary LoRA.
Research-only: no PHI, no diagnosis or treatment behavior, no clinical-use
claims.

## ADDED Requirements

### Requirement: Declarative deployment profiles

The deployment SHALL be defined by declarative profiles in a versioned config
file. Each profile SHALL specify model id, precision/quantization, GPU class,
min/max replicas, idle scale-down timeout (integer seconds), and adapter
slots (a list, possibly empty). The versioned config SHALL ship the default
`qwen3-8b-fp16` profile AND a concrete `qwen3-14b` benchmark profile.
Switching profiles (e.g. from Qwen3-8B to Qwen3-14B) SHALL require only a
config change, not application redesign.

#### Scenario: Default profile matches the customer decisions
- **WHEN** the deployment definition is built from the default
  `qwen3-8b-fp16` profile
- **THEN** it targets Qwen3-8B at fp16 precision (quantization configurable
  per profile) on one A10G-class 24 GB GPU, with min_replicas=0,
  max_replicas=1, and an idle scale-down timeout defaulting to 300 seconds
  that is validated to stay within the closed 120–300 second range
  (approximately 2–5 minutes)

#### Scenario: Zero idle GPU cost is configured
- **WHEN** the deployment definition is built from the default profile
- **THEN** it declares min_replicas=0 and an idle scale-down timeout within
  the 120–300 second range, configuring Baseten to scale the replica count to
  zero when no demo session is active and the timeout elapses (the actual
  scale-down event is observed in the deferred live step)

#### Scenario: Switching to a 14B benchmark profile
- **WHEN** the shipped `qwen3-14b` profile is selected in config
- **THEN** the deployment definition is produced for that profile without any
  application code change

#### Scenario: Impossible or out-of-range autoscaling bounds are rejected
- **WHEN** a profile specifies min_replicas greater than max_replicas or an
  idle scale-down timeout outside the closed 120–300 second range
- **THEN** profile loading rejects the configuration with an explicit
  validation error and no deployment definition is produced

### Requirement: Documented Baseten interfaces only

Every Baseten API or configuration interface the code uses SHALL be verified
against current Baseten documentation before coding, and each used interface
SHALL be cited with its documentation URL in a docs-notes file committed with
the code. No interface SHALL be used from memory alone.

#### Scenario: Docs check precedes implementation
- **WHEN** a Baseten wake, autoscaling, readiness, or deployment-configuration
  interface is implemented
- **THEN** the committed docs-notes file cites the exact current documentation
  URL for that interface

#### Scenario: Documentation conflicts with assumptions
- **WHEN** current documentation contradicts an interface assumption in this
  repository's records
- **THEN** the documented interface is used and the conflict is reported in
  the docs-notes file rather than silently keeping the assumption

### Requirement: LoRA-ready artifact layout

The deployment SHALL be built so the model and any adapter artifacts are
pre-deployed, and waking SHALL only provision/load an inference replica —
never rebuild or redeploy the model. The configuration SHALL represent
adapter slots as a list, not a single adapter, so a later aLoRA
(multiple-adapter, base-KV-cache) architecture requires no infrastructure
redesign.

#### Scenario: No build step in the deployment definition
- **WHEN** the deployment definition and profile are inspected
- **THEN** they reference pre-deployed model/adapter artifacts and contain no
  build, package, or redeployment step, so a later wake only provisions/loads
  a replica

#### Scenario: Multiple adapter slots are representable
- **WHEN** a profile declares more than one adapter slot
- **THEN** the deployment definition and configuration represent all of them
  without structural change

### Requirement: Hosting-independent clinical architecture

The clinical conversation logic SHALL remain independent of Baseten hosting
details. The Baseten integration SHALL live in its own module, and the
existing prototype code (`voice_proto/`) SHALL keep its current behavior.

#### Scenario: Clinical logic does not import hosting internals
- **WHEN** the clinical/conversation modules are inspected
- **THEN** they do not depend on Baseten-specific client internals
