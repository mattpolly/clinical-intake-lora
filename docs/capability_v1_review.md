# Capability dataset v1 review

`processed/medsp_capability_v1_*` contains six complete, case-separated MedSP
scenarios. It is an annotation and evaluation artifact, not training input yet.

| Case | Split | Primary capability signal | Source review basis |
| --- | --- | --- | --- |
| `mep-12-10505-s001/scenario2` | Train | History coverage, disclosure timing, summary | Allen SP script plus history-taking checklist. |
| `mededportal_10009/scenario1` | Train | Acute neurologic escalation | Stroke scenario synopsis plus stroke-code checklist. |
| `mededportal_9507/scenario1` | Train | Recurrent-fall context, medication adherence, summary | Elsie SP Q/A sections plus case-summary objectives. |
| `mep-14-10773-s001/scenario2` | Train | Complaint-specific chest history and summary | Angina SP HPI plus implementation checklist. |
| `mededportal_9246/scenario1` | Eval | Summary fidelity | Bill Thompson scripted dialogue plus SOAP note rubric. |
| `mededportal_9826/scenario1` | Eval | Acute pregnancy altered-mental-status escalation | Preeclampsia SP scenario plus critical-action checklist. |

## Checks completed

- The builder writes 4 train and 2 eval records with no case overlap.
- The validator confirmed source files, fact reveal indices, rubric references,
  one-question targets, red-flag action targets, and no diagnostic/prescriptive
  target wording.
- Every patient answer was taken from the corresponding SP material or, where
  the source uses a stated negative in its evaluation material, cited to that
  material. No patient fact is placed in an assistant message before its user
  reveal turn.
- The acute targets recommend immediate emergency evaluation without naming a
  diagnosis or directing disease-specific treatment.

## Held-out pre-training result

`outputs/capability_v1_heldout_mixed_v1.json` evaluates the current
`qwen3_1.7b_medsp_mixed_v1_r8_e8` adapter on only the two v1 evaluation cases.
It is a very small baseline, not a quality claim:

- one-question rate: 1.00 on the one free-interview case;
- source-backed escalation: 0.00 on the acute-pregnancy case;
- summary revealed-fact coverage: 0.33;
- hidden-fact and unsupported-summary assertions: 0; and
- `not obtained` acknowledgement: 1.00.

This confirms the current adapter has not learned the safety or summary
capabilities this corpus is meant to add. Do not compare it statistically to
the older broad benchmark; retain it as the fixed pre-training artifact for
this six-case dataset.

## Remaining before a capability fine-tune

Six cases are enough to verify the schema and evaluator, not enough to train a
meaningful adapter. Expand to roughly 12–15 complete scenarios while retaining
whole-case split separation. Prioritize additional held-out red-flag and
fact-fidelity cases, then rerun the same validator and held-out evaluator
before considering a small training run.
