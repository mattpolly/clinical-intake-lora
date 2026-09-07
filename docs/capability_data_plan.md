# Capability-focused MedSP source pool

This pool is separate from the fixed held-out benchmark and is training-only
until individual cases are deliberately assigned to evaluation. It was selected
only where MedSP supplies both `sp_actor` material (patient facts) and an
`evaluator` rubric/checklist (expected interviewer behavior).

## Red flags and escalation

- `mededportal_10009/scenario1` — stroke-code checklist
- `mededportal_10406/scenario1`, `mededportal_9237/scenario1` — critical-action checklists
- `mededportal_9826/scenario1` — severe preeclampsia/eclampsia checklist
- `mededportal_9890/scenario1`, `mededportal_9914/scenario1` — acute neurologic/sepsis faculty checklists
- `mep-15-10813-s001/scenario1-3` — hyponatremia, alcohol-withdrawal, and brain-herniation critical actions
- `mep-15-10831-s001/scenario1-2` — MI checklists

Target examples: one source-backed urgent disclosure followed by a concise
escalation recommendation or immediate-safety follow-up, with no diagnosis or
treatment prescription.

## Fact fidelity and contextual history

- `mededportal_10312/scenario1-2` — standardized-patient checklists
- `mededportal_10350/scenario1` — psychiatric interview checklist
- `mededportal_9067/scenario1-2` — nutrition-history checklists
- `mededportal_9110/scenario1-2` — substance-use assessment
- `mededportal_9218/scenario1-2` — LGBTI SP checklists
- `mep-12-10505-s001/scenario2` — history-taking checklist
- `mep-12-10519-s001/scenario1` — medication-history assessment
- `mep-13-10622-s001/scenario1` — IPV screening/counseling rubric
- `mep-14-10773-s001/scenario1-2` — diabetes/angina SP implementation checklists

Target examples: only ask for a fact not yet revealed; model summaries must
include revealed facts and mark relevant unrevealed fields as `not obtained`.

## Structured summaries and communication

- `mededportal_9246/scenario1` — SOAP note rubric
- `mededportal_9507/scenario1` — SP and communication checklists
- `mededportal_9759/scenario1` — guide to history checklist and grading rubric
- `mededportal_10400/scenario1` — simulated-encounter evaluation checklist

Target examples: completed, source-backed conversation histories paired with a
non-diagnostic structured summary and field-level expected facts.

## Construction rules

1. Cite the scenario and exact SP/evaluator file for every record.
2. Keep a patient fact hidden until a prior patient answer reveals it.
3. Build multi-turn examples, not just independent question/answer fragments.
4. Use the compact annotation contract in
   [`capability_dataset_schema.md`](capability_dataset_schema.md). Capability
   labels are derived from its rubric and turn-action fields rather than
   separately tagged.
5. Reserve complete scenarios for evaluation before authoring any records from
   them; never split turns of one scenario across train/evaluation.
6. Manually review every escalation and summary target before training.
