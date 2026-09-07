# Safety-improvement plan

This is the active follow-through plan for the capability adapter. The current
4B capability-v3 adapter is retained for comparison and is not the CLI default.

1. **Safety tranche — complete (v4):** added 8 source-backed escalation conversations, with
   4–6 training cases and 3–4 complete held-out cases. Include both immediate
   and delayed disclosures.
2. **Contrast cases — complete:** paired routine source-backed histories with closely
   related histories that cross a documented urgent threshold, without
   inventing patient facts.
3. **Balanced export — complete:** retained case separation and assistant-only loss, but
   oversample escalation targets about 4× in the training export only.
4. **Shared policy — complete in v4:** updated the common system prompt to say that a documented
   serious red flag pauses routine intake for immediate emergency evaluation;
   continue to prohibit diagnosis and treatment instructions.
5. **Verification and promotion — complete, not promoted:** ran an expanded, source-explicit held-out
   safety benchmark before and after one small QLoRA run. Promote only if
   escalation improves without material regression in one-question behavior,
   fact fidelity, or summaries.

All new records must pass `validate_capability_dataset.py`; all split choices
remain whole-scenario choices. Automated flags remain manual-review prompts,
not clinical safety claims.

## v4 outcome

`medsp_capability_v4` contains 20 records across 16 non-overlapping whole
scenarios (12 train records/11 cases; 8 held-out records/5 cases). Its 1024-token
export has 71 weighted train targets: 40 questions, 6 summaries, 1 safety
follow-up, and 24 escalation targets; the 28 held-out targets were not weighted.

One 4B QLoRA epoch completed in 73.6 seconds: train loss 2.44, eval loss 1.58,
peak recorded VRAM 5.50 GiB, and adapter reload PASS. On the corrected held-out
safety simulator, escalation improved from 0/6 to 1/6 events and summary
fact coverage from 27% to 67%, with no automated hidden-fact or diagnostic
flags. Repetition/manual-review question flags worsened, so the adapter is
experimental and **not** the CLI default.
