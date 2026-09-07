# Acuity mode-transition probe v1

This frozen probe evaluates a more relevant prerequisite than a one-shot label:
whether Qwen changes its intended next action as patient facts accrue.

Each of 12 source-cited cases has two stages: an initial non-emergency or
focused-safety state, followed by a public-guidance-backed explicit emergency
pattern. The three labels are intentionally behavioral rather than diagnostic:
`CONTINUE_HISTORY`, `FOCUSED_SAFETY_CHECK`, and `EMERGENCY`.

The initial benchmark is a baseline only. It is manually curated, small, and
not a clinical decision rule. It must remain unchanged before using it to
compare any later adapter.

## Baseline results — 2026-09-05

| Model | State accuracy | Emergency recall (12 endpoints) | Non-emergency mode accuracy | Fully correct transitions (12 cases) |
| --- | ---: | ---: | ---: | ---: |
| Untouched Qwen3-4B | 70.8% | 58.3% | 83.3% | 5 |
| v4 safety adapter | 50.0% | 66.7% | 33.3% | 1 |

The base model's common failure was retaining `FOCUSED_SAFETY_CHECK` after a
decisive cue arrived. The adapter improved endpoint emergency recall slightly,
but over-selected `FOCUSED_SAFETY_CHECK` at initial non-emergency states and
therefore degraded the mode-transition task. This is a direct reason not to
extend the old free-form escalation corpus.

Raw results are in:

- `outputs/acuity_transition_probe_v1_qwen3_4b_base.json`
- `outputs/acuity_transition_probe_v1_qwen3_4b_v4_safety.json`
