# Capability-training dataset schema

This is the proposed annotation contract for the next capability-focused
training and evaluation records. It is deliberately separate from the existing
`processed/*.jsonl` datasets: no records have been converted to this format or
used for training yet.

The unit is a complete MedSP scenario. A scenario belongs to exactly one split;
individual turns from it must never be divided between training and evaluation.
Every non-obvious target must be traceable to an SP or evaluator source file.

## Design constraint

The dataset has to support four things at once:

1. train concise, state-aware interviewing;
2. score assertions against facts that have actually been revealed;
3. teach source-backed urgent follow-up or escalation; and
4. score a non-diagnostic summary against the revealed history.

It must do so with a small number of carefully reviewed cases. A wide
annotation scheme would multiply manual work and create many sparsely used
parameters, so this design was reduced in three passes.

## Revision 1 — expressive, but too wide

The first draft separated all possible concepts into fields such as
`capability_tags`, `patient_facts`, `fact_visibility`, `question_targets`,
`rubric_items`, `question_priority`, `red_flag_trigger`, `safety_policy`,
`summary_spec`, `summary_fields`, `forbidden_assertions`, `turn_state`,
`review_status`, and per-field source locators.

It could represent nearly anything, but it has three problems:

- the same state is written three times (a patient answer, fact visibility,
  and turn state);
- a small corpus would leave many categories nearly empty; and
- annotators would need to make subjective distinctions that do not improve
  training, such as whether a question is "high" versus "medium" priority.

**Decision:** reject it. Most of those are validation rules or derived
properties, not data that should be labeled per record.

## Revision 2 — compact, but still redundant

The second draft retained `messages`, `facts`, `rubric_items`, `turn_targets`,
`safety`, and `summary`. It was much smaller, but `facts.reveal_turn`,
`turn_targets.covered_facts`, and `safety.trigger_turn` all encoded the same
conversation timeline. A separate `summary` object also repeated which facts
belonged in a summary.

**Decision:** merge timeline fields into the fact and target objects. Derive
summary expectations from the facts that are visible at the summary turn.

## Revision 3 — settled schema

Each record has seven top-level fields. Apart from the actual dialogue, there
are only three annotation collections: `facts`, `rubric`, and `turns`.

```json
{
  "id": "mep-12-10505-s001-scenario2-intake-01",
  "case": {
    "id": "mep-12-10505-s001/scenario2",
    "split": "train"
  },
  "source": {
    "sp": "mep-12-10505-s001/scenario2/sp_actor/E. Script for Simulated Patient Allen.md",
    "evaluator": "mep-12-10505-s001/scenario2/evaluator/G. History-Taking Checklist for Simulated Patient Allen.md"
  },
  "messages": [
    {
      "role": "system",
      "content": "Conduct a structured clinical intake interview. Ask one concise question at a time. Gather relevant history before summarizing. Do not diagnose."
    },
    {"role": "user", "content": "My regular doctor referred me here because I've been having blood in my stool."},
    {"role": "assistant", "content": "When did you first notice the bleeding?"},
    {"role": "user", "content": "About a year ago."},
    {"role": "assistant", "content": "Does anything make the bleeding better or worse?"},
    {"role": "user", "content": "Nothing I can identify. I tried eating more fiber and drinking water, but it did not make a difference."},
    {"role": "assistant", "content": "What does the blood and your stool usually look like?"}
  ],
  "facts": [
    {
      "id": "bleeding_onset",
      "field": "hpi",
      "reveal_in": 3,
      "terms": ["about a year ago"],
      "evidence": "sp:onset"
    },
    {
      "id": "no_identified_aggravating_or_relieving_factor",
      "field": "hpi",
      "reveal_in": 5,
      "terms": ["nothing", "fiber", "water", "did not make a difference"],
      "evidence": "sp:HPI—Aggravated by what / Relieved by what"
    }
  ],
  "rubric": [
    {"id": "onset", "tier": "core", "evidence": "evaluator:item 1 (onset)"},
    {"id": "aggravating_relieving", "tier": "core", "evidence": "evaluator:items 2–3 (alleviating/aggravating)"},
    {"id": "quality", "tier": "core", "evidence": "evaluator:item 4 (quality)"}
  ],
  "turns": [
    {"assistant_in": 2, "action": "question", "items": ["onset"]},
    {"assistant_in": 4, "action": "question", "items": ["aggravating_relieving"]},
    {"assistant_in": 6, "action": "question", "items": ["quality"]}
  ]
}
```

Message indices are zero-based. `reveal_in` and `assistant_in` therefore point
directly to an entry in `messages`; the builder must reject a mismatch between
the index and its role.

### The seven fields

| Field | Purpose | Required contents |
| --- | --- | --- |
| `id` | Immutable record identifier. | String, unique in its dataset. |
| `case` | Split safety and scenario identity. | MedSP scenario ID and `train` or `eval`. |
| `source` | Case-level provenance. | SP file and evaluator file paths. |
| `messages` | The trainable dialogue and ground-truth disclosure order. | System, patient, and assistant turns. |
| `facts` | The patient-state ledger and expected summary facts. | ID, field, patient-message index, matching terms, short source locator. |
| `rubric` | What the source says is worth obtaining. | ID, `core`/`red_flag`/`background`, short evaluator locator. |
| `turns` | What a labeled assistant turn is doing. | Assistant-message index, action, and relevant rubric IDs. |

The intentional controlled vocabularies are small:

- `facts[].field`: `chief_complaint`, `hpi`, `associated`,
  `meds_allergies`, `history`, or `social_family`.
- `rubric[].tier`: `core`, `red_flag`, or `background`.
- `turns[].action`: `question`, `safety_followup`, `escalation`, or `summary`.

`items` normally references `rubric[].id`. For a `safety_followup` or
`escalation`, it references one or more `red_flag` rubric IDs. A `summary`
turn has an empty `items` list: its expected content is all facts whose
`reveal_in` occurs earlier in the dialogue, organized by `field`.

## What is deliberately *not* a schema field

- **Capability tags:** derived from `turns.action`, `rubric.tier`, and whether
  a summary turn exists.
- **A separate patient-state object:** derived from `facts.reveal_in` and the
  current message index.
- **Summary targets or field checklists:** derived from revealed `facts` and
  their `field` values; unrepresented fields are scored as `not obtained`.
- **Question count and no-diagnosis flags:** deterministic validators over the
  assistant text and the shared system instruction.
- **Question priority scores, diagnoses, treatment plans, or free-form
  safety labels:** not needed for the intended behavior and too subjective for
  this corpus.

This keeps annotation focused on source evidence and makes the corpus useful
for both fine-tuning and evaluation without inventing patient information.

## Required builder and evaluator checks

These are code-level checks, not annotation parameters:

1. Assert unique record IDs and no overlap of `case.id` across splits.
2. Assert every source path exists and every `evidence` locator is non-empty.
3. Assert `reveal_in` points to a user turn and `assistant_in` points to an
   assistant turn.
4. At every assistant turn, expose only facts with an earlier `reveal_in`.
   Flag assistant assertions that match facts not yet revealed.
5. Require question-action targets to contain one concise question and reject
   obvious diagnosis/treatment language in every assistant target.
6. Reject a turn that covers an already covered rubric item unless it is
   explicitly a clarification; for this first corpus, do not author
   clarification turns.
7. For each summary, score only facts revealed before the summary, require the
   fixed six-field layout, and flag unsupported additions.
8. For source-explicit red flags, require a subsequent `safety_followup` or
   `escalation` target, then manually review the wording.

## How it serves the four capabilities

| Capability | Training signal | Evaluation signal |
| --- | --- | --- |
| Context and no repetition | Ordered `messages` plus question `turns`. | Compare asked rubric IDs with prior covered IDs. |
| Fact fidelity | `facts.reveal_in` blocks hidden facts. | Flag facts asserted before their patient turn. |
| Red flags | `red_flag` rubric items and a safety action. | Check an appropriate action after the documented trigger. |
| Structured summary | A `summary` assistant turn after the conversation. | Compare fields and terms only to earlier revealed facts. |

## Annotation workflow

1. Reserve the complete MedSP scenario as train or evaluation before authoring.
2. Extract a short list of evaluator rubric items; use `core` by default and
   add `red_flag` only where the source explicitly supports it.
3. Build a short multi-turn dialogue from SP facts, assigning each disclosed
   fact a `reveal_in` index.
4. Attach a `turns` entry only to assistant targets that should be trained or
   evaluated. Keep questions single-purpose and concise.
5. Add a final non-diagnostic summary only when the selected source material
   supports a sufficiently complete history.
6. Run the checks above and manually review every safety and summary turn.

The initial construction target should be a small, balanced set of complete
cases (roughly 3–5 red-flag, 4–6 history/fidelity, and 3–5 summary-capable
cases), rather than trying to annotate every selected scenario at once.
