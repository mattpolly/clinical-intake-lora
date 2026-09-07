# Acuity-mode curriculum: design before retraining

## What the model must learn

Not a diagnosis, treatment plan, or a memorized mapping from complaint to
escalation. After each newly revealed patient fact, it must choose one of three
observable interview modes:

1. `CONTINUE_HISTORY` — continue one concise, complaint-relevant question.
2. `FOCUSED_SAFETY_CHECK` — ask one targeted safety question when a possible
   danger pattern is incomplete.
3. `EMERGENCY` — stop routine intake and give a non-diagnostic immediate
   emergency-evaluation recommendation when an explicit source-backed pattern
   is present.

## Why this is different from v4

The v4 adapter saw natural-language questions and escalation responses but no
explicit decision target. Its small weighted safety set made escalation wording
more frequent without teaching when to change modes. The frozen transition
probe demonstrates that this worsened mode selection.

## Minimal signal vocabulary

Use eight source-cited, observable safety domains rather than disease labels:

`airway_breathing`, `circulation_cardiac`, `acute_neurologic`,
`systemic_deterioration`, `major_bleeding`, `pregnancy_postpartum`,
`toxicologic_exposure`, and `trauma_or_severe_pain`.

Each training example should identify at most two domains. A domain is an audit
label for dataset construction and evaluation, not a diagnosis shown to the
patient.

## Record shape

Keep the existing compact conversation schema. Add no broad clinical feature
matrix. For a new mode curriculum record, the assistant target begins with a
short machine-readable header, followed by patient-facing language:

```text
<mode>FOCUSED_SAFETY_CHECK</mode>
Are you having trouble breathing right now?
```

or:

```text
<mode>EMERGENCY</mode>
The symptoms you described need immediate emergency evaluation. Please seek emergency care now.
```

The CLI will parse and hide the header, displaying only the patient-facing
portion. This makes the decision supervised and testable without exposing a
diagnosis or hidden chain-of-thought.

## Required data pattern

For each safety domain, create several *progressive* sequences:

1. A non-emergency opening -> `CONTINUE_HISTORY`.
2. A partial pattern -> `FOCUSED_SAFETY_CHECK`.
3. A documented decisive cue -> `EMERGENCY`.

Use source-backed near-neighbor controls for every emergency endpoint. Split
whole source scenarios before export; keep `acuity_transition_probe_v1` frozen.
Do not use simple oversampling as a replacement for domain diversity.

## Acceptance gate before a new adapter

- At least 4 distinct source scenarios in each represented safety domain.
- At least 2 progressive sequences per domain, with varied wording and no
  source-case overlap with the frozen transition benchmark.
- Balanced mode targets before weighting; if weighting is necessary, document
  it and never weight held-out rows.
- Untouched-base and post-adapter results reported on the frozen transition
  probe, plus manual review of every emergency endpoint.

The next implementation task is source selection and a reviewable curriculum
manifest—not another free-form LoRA run.
