# MedSP1000 inspection report — direct-pair pilot

## Scope

The original five-case review was extended with MedSP1000's remote file index
(9,959 Markdown files; 344 evaluator files with checklist-like names) and two
targeted cases from `mededportal_10373`.  This is enough to establish that the
dataset is heterogeneous: it is not a pre-formatted dialogue corpus, but some
evaluator checklists contain explicit patient-response and learner-question
pairs.

## Role mapping

| Role | Typical contents | Training value |
| --- | --- | --- |
| `sp_actor` | Opening statement, facts disclosed on request, affect, and embedded concerns. | Patient state and permissible responses. |
| `evaluator` | Learning objectives, scoring rubrics, history checklist items, and sometimes direct response/question pairs. | Question targets and behavioral evaluation criteria. |
| `examinee` | Learner-facing setup, tasks, and prompts. | Context only; not patient facts. |
| `environment_controller` | Simulation state, hidden test/exam findings, and event sequencing. | Exclude from ordinary history-taking targets unless the encounter reaches that state. |

## Verified direct pairs

Two evaluator checklists use an explicit pattern: a patient response followed
immediately by `Student: <question>`.  The direct-pair pilot preserves that
wording and records source lines for each record.

| Case | File | Direct one-question pairs | Split |
| --- | --- | ---: | --- |
| Cough | `mededportal_10373/scenario1/evaluator/B. Cough Case Standardized Patient Master Encounter Checklist.md` | 7 | Train |
| Back pain | `mededportal_10373/scenario2/evaluator/G. Back-Pain Case Standardized Patient Master Encounter Checklist.md` | 10 | Evaluation |

For example, the cough checklist lines 5–7 pair “It started about 3 days ago.”
with “How long have you had a cough?”  The back-pain checklist lines 5–7 pair
location information with “Can you show me/tell me where it hurts?”

## Findings from the initial five scenarios

| Case | Patient facts | Interview targets / rubric | Direct dialogue status |
| --- | --- | --- | --- |
| Headache | `sp_actor` script contains opening statement, HPI, history, and conditionally revealed findings. | Evaluator resource lists objectives and tasks. | No verified adjacent question/answer checklist in reviewed file. |
| Acute abdominal pain | SP material contains detailed history and physical-exam instructions. | H&P checklist specifies pain dimensions and red-flag questions. | Rubric targets, not verified direct pairs. |
| Cough | SP material contains conditional responses and embedded concern. | Master encounter checklist supplies direct pairs and scoring values. | Direct pairs available. |
| Chest pain | Instructor guide provides triage facts and escalation expectations. | Triage and information-gathering guidance. | Narrative/guide, not direct dialogue. |
| Dyspnea | SP script and storyboard provide scenario facts. | Reviewed evaluation file is course-level, not an intake rubric. | No verified direct pairs. |

## Dataset decision

MedSP1000 remains the primary dataset; no external supplemental dataset is
being added.  The legacy `medsp_intake_*` files are manually authored,
source-grounded conversations and must remain labeled as such.  The new
`medsp_direct_pairs_v1_*` files are a separate, smaller high-provenance pilot
for testing whether verbatim checklist questions improve one-question behavior.

Before a larger retraining run, inspect and download more cases with compatible
direct-pair checklists, then expand the direct-pair dataset across complaints
and keep train/evaluation splits separated by complete source case.
