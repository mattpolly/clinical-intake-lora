# Capability dataset v2 review

Capability v2 extends v1 to nine complete, case-separated MedSP scenarios:
six train and three held out. It remains separate from all training inputs.

New source-reviewed additions are Michelle DeMoro (`mep-14-10773-s001/scenario1`)
for symptom/medication history, Ann Bonder (`mededportal_9218/scenario1`) for
explicitly gated sensitive-fact disclosure, and Patricia Smith
(`mededportal_10312/scenario1`) for medication-use and fainting history.

```bash
python scripts/build_capability_dataset_v2.py
python scripts/validate_capability_dataset.py \
  processed/medsp_capability_v2_train.jsonl \
  processed/medsp_capability_v2_eval.jsonl
```

The validator passed all nine records with source paths, reveal order, target
language, rubric references, and complete-case split separation intact.

## Held-out pre-training baseline

`outputs/capability_v2_heldout_mixed_v1.json` evaluates only Bill Thompson,
the acute-pregnancy trigger, and Patricia Smith using the current adapter. It
reports one-question rate 1.00 across the two free interviews, two exact
repeats, six unmatched questions for manual review, zero hidden-fact leaks,
zero escalation on the acute trigger, 0.33 summary fact coverage, and no
unsupported summary facts. This is a small behavioral baseline, not a clinical
quality or safety claim.

v2 is superseded for the next training experiment by the 12-case v3 corpus.
Its v2 artifact is retained as a source-separated intermediate baseline.
