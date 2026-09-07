# Acuity probe v1

This is a deliberately small baseline test of whether a model recognizes an
*explicitly stated* emergency warning pattern. It is not a clinical triage
instrument, a diagnosis benchmark, or training data.

It contains 22 short paraphrased presentations. Emergency labels are grounded
in public guidance from the American Heart Association, CDC, ACOG, MedlinePlus,
and Poison Control; non-emergency controls omit the cited emergency features.
The source URL and the supporting cue cluster are stored alongside each record.

The constrained labels are `EMERGENCY` and `NOT_EMERGENCY`. This avoids asking
the model for a diagnosis or treatment and exposes a necessary prerequisite for
the future intake behavior: can it distinguish an explicit acute pattern from a
non-acute control before any LoRA training?

Use `python scripts/build_acuity_probe.py`, then
`python scripts/evaluate_acuity_probe.py --model-id Qwen/Qwen3-4B`.

## Initial baseline result

On 2026-09-05, untouched `Qwen/Qwen3-4B` scored 22/22: emergency recall
14/14, control specificity 8/8, and no unparseable responses. The experimental
v4 safety adapter scored the same 22/22 under this constrained prompt. Results:

- `outputs/acuity_probe_v1_qwen3_4b_base.json`
- `outputs/acuity_probe_v1_qwen3_4b_v4_safety.json`

This is encouraging but intentionally insufficient to claim broad clinical
reasoning: the examples are small, obvious, and source-shaped. The important
finding is architectural: Qwen can classify the stated patterns when prompted
to do so, while the free-form intake benchmark often fails to switch from
questioning to escalation. The next probe should be blinded, larger, include
multi-turn progressive disclosure and near-threshold contrasts, and keep its
source set frozen before any training.
