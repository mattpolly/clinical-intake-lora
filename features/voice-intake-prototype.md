# Voice intake prototype (AWS + Rime)

<!-- Agentic Devshop feature: voice-intake-prototype -->

## Intent

Headless STT -> intake model -> TTS loop using AWS Transcribe, a pluggable Baseten Qwen backend, and Rime Mist, with latency evidence.

## Requirements

# Voice intake prototype (AWS + Rime)

## Intent

Prototype the `docs/DEPLOYMENT.md` voice call path on this host as a headless,
component-level loop: speech-to-text via AWS Transcribe (streaming), intake-model
inference via a remote Qwen endpoint (Baseten; customer is creating the account),
and text-to-speech via Rime Mist, with per-stage latency measurement. Research
only: synthetic/standardized-patient scenarios, no PHI, no diagnosis or treatment
behavior.

## Customer requirements

- Build the prototype using AWS and Rime, following the call path in
  `docs/DEPLOYMENT.md` (minus telephony for now).
- Customer is setting up a Baseten account and will supply credentials
  separately; the model backend must be pluggable so Baseten drops in later.
- AWS credentials come from the shared TextLife account
  (`../textlife-specs/.env`); use the account but never read, write, or depend
  on TextLife resources (DynamoDB table `textlife`, existing Lambdas, Telnyx
  SMS webhook).
- No Rime API key exists yet; build the TTS client and report a clear
  missing-credential marker until the customer provides one.

## Delivery intent

MVP: one command runs the loop end-to-end — synthetic patient speech fixture →
real AWS Transcribe streaming result → intake-model turn (deterministic mock
until Baseten credentials arrive) → Rime TTS (or explicit missing-credential
marker) — and saves an artifact with per-stage latencies.

## Decisions (blocking-question answers)

1. **Synthetic fixture:** generated at runtime from text via Amazon Polly
   (per-request API, no created resource), saved as raw 16 kHz 16-bit mono PCM
   (`.wav` header for storage, stripped for streaming) under `outputs/`
   (Git-ignored), never checked into the repo. Polly standard-engine voice,
   `en-US`, PCM output requested at 16000 Hz. Content is one short scripted
   synthetic-patient complaint statement (research scenario, no real-person
   data). Hard cap: one sentence, ≤10 seconds. Transcribe streaming media
   declaration is fixed accordingly: `pcm`, 16000 Hz, mono, `en-US`.
2. **Pipeline shape: one-shot** for this MVP — one utterance → one model turn
   → one TTS output. A continuous multi-turn conversational loop (session
   state, interruption handling) is an explicit follow-up slice, not this one.
3. **AWS credential availability:** assume the shared credentials may work; if
   missing or IAM-denied for Polly/Transcribe, the loop degrades to an explicit
   `missing-credential` or `access-denied` stage marker — exactly the same
   treatment as Baseten and Rime. The loop never silently aborts; it always
   produces a complete artifact.
4. **Intake-model turn contract:** a `ModelBackend` interface with one method,
   `turn(conversation) -> {"text", "latency_ms"}`, where `conversation` is a
   list of `{"role": "patient"|"intake", "content": str}` (the model stage
   receives only this list; no extra metadata). Output is non-streaming plain
   text (one concise question or summary). The mock backend returns a
   deterministic scripted follow-up question; the Baseten backend sends the
   same conversation as an OpenAI-compatible chat-completions request to a
   configurable endpoint. The loop and artifact schema are identical for both
   backends.
5. **Latency boundaries:** STT starts when the first audio byte is written to
   the Transcribe stream and ends when the final transcript event arrives
   (time-to-first-partial is recorded as a separate field); model starts when
   the HTTP request is sent and ends when the full body is received
   (time-to-first-byte recorded separately); TTS likewise. Fixture generation
   (Polly) is timed separately and never counted in loop latency.
6. **Credential injection:** standard environment variables (`AWS_*`,
   `BASETEN_API_KEY` + `BASETEN_MODEL_URL`, `RIME_API_KEY`), optionally loaded
   from a `--env-file` path (e.g. `../textlife-specs/.env`); secret values
   never appear in code, artifacts, or Git. Failure mode: partial artifact with
   explicit per-stage markers; process exits non-zero only on unexpected
   errors, not on missing optional credentials.
7. **Cost caps:** streaming Transcribe is bounded by the fixture — max 10
   seconds of audio per run (enforced in code). Polly synthesis is one short
   request. Total per-run AWS cost is sub-cent.
8. **"No standing AWS resource"** means no persistent infrastructure: no S3
   buckets, DynamoDB tables, Lambda functions, Connect instances, or phone
   numbers. Polly and streaming Transcribe are per-request API calls that
   require no created resources; batch Transcribe (which needs S3) is
   explicitly avoided for this reason.
9. **TTS artifact shape:** when Rime credentials are present, the artifact
   records a saved audio-file path under `outputs/` plus metadata (bytes,
   format, latency); audio bytes are never embedded in the JSON artifact. The
   loop only persists audio — it never plays it (headless host, no audio
   device).
10. **A model turn is a single free-text reply** (one concise question or a
   structured summary text), not structured field extraction and not a
   multi-turn exchange; multi-turn is the follow-up slice.
11. **Verifying the AWS boundary:** the code calls only
   `polly:SynthesizeSpeech` and streaming `transcribe:StartStreamTranscription`;
   absence of any other AWS API call is verifiable by inspecting the package
   imports and a single grep over the source. No IAM policy is changed by this
   work; if the customer wants credential scoping later, that is a follow-up
    with them.

## Constraints and non-goals

- No Amazon Connect phone provisioning or any standing-cost AWS resource in
  this slice; ask the customer before creating resources that cost money.
- Synthetic patient speech fixtures generated locally (Polly, pay-per-request)
  so no microphone is needed on this headless host.
- No PHI, no real patient audio, no clinical-use claims; keep the research-only
  disclaimer in all new docs.
- Secrets only via environment / `--env-file`; never committed to Git.
- No aLoRA work in this slice; the remote endpoint is a conventional
  single-adapter-style inference API behind a backend interface.

## Definition of enough

- `python -m voice_proto.run_loop` (or equivalent) produces a saved JSON
  artifact under `outputs/` containing: a real AWS Transcribe streaming
  transcript of the synthetic fixture (or explicit credential/denied marker),
  a model turn from the configured backend, TTS output (or explicit
  missing-credential marker), and per-stage latency timings.
- No TextLife AWS resource is read or written; no standing AWS resource is
  created.
- No secret values are committed to Git.
- A short doc records how to plug in Baseten and Rime credentials later.

## Follow-ups (out of this slice)

- Baseten endpoint wiring once credentials arrive; re-run loop with real model.
- Rime key wiring; verify the Rime API shape against live docs.
- Telephony slice (Connect or Telnyx voice) — requires explicit customer
  approval for phone-number spend.
- aLoRA cache-fork investigation stays on the TODO.md track.

## Intent

Prototype the `docs/DEPLOYMENT.md` voice call path on this host as a headless,
component-level loop: speech-to-text via AWS Transcribe (streaming), intake-model
inference via a remote Qwen endpoint (Baseten; customer is creating the account),
and text-to-speech via Rime Mist, with per-stage latency measurement. Research
only: synthetic/standardized-patient scenarios, no PHI, no diagnosis or treatment
behavior.

## Customer requirements

- Build the prototype using AWS and Rime, following the call path in
  `docs/DEPLOYMENT.md` (minus telephony for now).
- Customer is setting up a Baseten account and will supply credentials
  separately; the model backend must be pluggable so Baseten drops in later.
- AWS credentials come from the shared TextLife account
  (`../textlife-specs/.env`); use the account but never read, write, or depend
  on TextLife resources (DynamoDB table `textlife`, existing Lambdas, Telnyx
  SMS webhook).
- No Rime API key exists yet; build the TTS client and report a clear
  missing-credential marker until the customer provides one.

## Delivery intent

MVP: one command runs the loop end-to-end — synthetic patient speech fixture →
real AWS Transcribe streaming result → intake-model turn (deterministic mock
until Baseten credentials arrive) → Rime TTS (or explicit missing-credential
marker) — and saves an artifact with per-stage latencies.

## Decisions (blocking-question answers)

1. **Synthetic fixture:** generated at runtime from text via Amazon Polly
   (per-request API, no created resource), saved as raw 16 kHz 16-bit mono PCM
   (`.wav` header for storage, stripped for streaming) under `outputs/`
   (Git-ignored), never checked into the repo. Polly standard-engine voice,
   `en-US`, PCM output requested at 16000 Hz. Content is one short scripted
   synthetic-patient complaint statement (research scenario, no real-person
   data). Hard cap: one sentence, ≤10 seconds. Transcribe streaming media
   declaration is fixed accordingly: `pcm`, 16000 Hz, mono, `en-US`.
2. **Pipeline shape: one-shot** for this MVP — one utterance → one model turn
   → one TTS output. A continuous multi-turn conversational loop (session
   state, interruption handling) is an explicit follow-up slice, not this one.
3. **AWS credential availability:** assume the shared credentials may work; if
   missing or IAM-denied for Polly/Transcribe, the loop degrades to an explicit
   `missing-credential` or `access-denied` stage marker — exactly the same
   treatment as Baseten and Rime. The loop never silently aborts; it always
   produces a complete artifact.
4. **Intake-model turn contract:** a `ModelBackend` interface with one method,
   `turn(conversation) -> {"text", "latency_ms"}`, where `conversation` is a
   list of `{"role": "patient"|"intake", "content": str}` (the model stage
   receives only this list; no extra metadata). Output is non-streaming plain
   text (one concise question or summary). The mock backend returns a
   deterministic scripted follow-up question; the Baseten backend sends the
   same conversation as an OpenAI-compatible chat-completions request to a
   configurable endpoint. The loop and artifact schema are identical for both
   backends.
5. **Latency boundaries:** STT starts when the first audio byte is written to
   the Transcribe stream and ends when the final transcript event arrives
   (time-to-first-partial is recorded as a separate field); model starts when
   the HTTP request is sent and ends when the full body is received
   (time-to-first-byte recorded separately); TTS likewise. Fixture generation
   (Polly) is timed separately and never counted in loop latency.
6. **Credential injection:** standard environment variables (`AWS_*`,
   `BASETEN_API_KEY` + `BASETEN_MODEL_URL`, `RIME_API_KEY`), optionally loaded
   from a `--env-file` path (e.g. `../textlife-specs/.env`); secret values
   never appear in code, artifacts, or Git. Failure mode: partial artifact with
   explicit per-stage markers; process exits non-zero only on unexpected
   errors, not on missing optional credentials.
7. **Cost caps:** streaming Transcribe is bounded by the fixture — max 10
   seconds of audio per run (enforced in code). Polly synthesis is one short
   request. Total per-run AWS cost is sub-cent.
8. **"No standing AWS resource"** means no persistent infrastructure: no S3
   buckets, DynamoDB tables, Lambda functions, Connect instances, or phone
   numbers. Polly and streaming Transcribe are per-request API calls that
   require no created resources; batch Transcribe (which needs S3) is
   explicitly avoided for this reason.

## Constraints and non-goals

- No Amazon Connect phone provisioning or any standing-cost AWS resource in
  this slice; ask the customer before creating resources that cost money.
- Synthetic patient speech fixtures generated locally (Polly, pay-per-request)
  so no microphone is needed on this headless host.
- No PHI, no real patient audio, no clinical-use claims; keep the research-only
  disclaimer in all new docs.
- Secrets only via environment / `--env-file`; never committed to Git.
- No aLoRA work in this slice; the remote endpoint is a conventional
  single-adapter-style inference API behind a backend interface.

## Definition of enough

- `python -m voice_proto.run_loop` (or equivalent) produces a saved JSON
  artifact under `outputs/` containing: a real AWS Transcribe streaming
  transcript of the synthetic fixture (or explicit credential/denied marker),
  a model turn from the configured backend, TTS output (or explicit
  missing-credential marker), and per-stage latency timings.
- No TextLife AWS resource is read or written; no standing AWS resource is
  created.
- No secret values are committed to Git.
- A short doc records how to plug in Baseten and Rime credentials later.

## Follow-ups (out of this slice)

- Baseten endpoint wiring once credentials arrive; re-run loop with real model.
- Rime key wiring; verify the Rime API shape against live docs.
- Telephony slice (Connect or Telnyx voice) — requires explicit customer
  approval for phone-number spend.
- aLoRA cache-fork investigation stays on the TODO.md track.

## Intent

Prototype the `docs/DEPLOYMENT.md` voice call path on this host as a headless,
component-level loop: speech-to-text via AWS Transcribe (streaming), intake-model
inference via a remote Qwen endpoint (Baseten; customer is creating the account),
and text-to-speech via Rime Mist, with per-stage latency measurement. Research
only: synthetic/standardized-patient scenarios, no PHI, no diagnosis or treatment
behavior.

## Customer requirements

- Build the prototype using AWS and Rime, following the call path in
  `docs/DEPLOYMENT.md` (minus telephony for now).
- Customer is setting up a Baseten account and will supply credentials
  separately; the model backend must be pluggable so Baseten drops in later.
- AWS credentials come from the shared TextLife account
  (`../textlife-specs/.env`); use the account but never read, write, or depend
  on TextLife resources (DynamoDB table `textlife`, existing Lambdas, Telnyx
  SMS webhook).
- No Rime API key exists yet; build the TTS client and report a clear
  missing-credential marker until the customer provides one.

## Delivery intent

MVP: one command runs the loop end-to-end — synthetic patient speech fixture →
real AWS Transcribe streaming result → intake-model turn (deterministic mock
until Baseten credentials arrive) → Rime TTS (or explicit missing-credential
marker) — and saves an artifact with per-stage latencies.

## Decisions (blocking-question answers)

1. **Synthetic fixture:** generated at runtime from text via Amazon Polly
   (per-request API, no created resource), saved as 16 kHz 16-bit mono PCM WAV
   under `outputs/` (Git-ignored), never checked into the repo. Content is one
   short scripted synthetic-patient complaint statement (research scenario, no
   real-person data). Hard cap: 10 seconds / one sentence of speech per
   fixture.
2. **AWS credential availability:** assume the shared credentials may work; if
   missing or IAM-denied for Polly/Transcribe, the loop degrades to an explicit
   `missing-credential` or `access-denied` stage marker — exactly the same
   treatment as Baseten and Rime. The loop never silently aborts; it always
   produces a complete artifact.
3. **Intake-model turn contract:** a `ModelBackend` interface with one method,
   `turn(conversation) -> {"text", "latency_ms"}`, where `conversation` is a
   list of `{"role": "patient"|"intake", "content": str}`. The mock backend
   returns a deterministic scripted follow-up question (one concise question,
   honoring the project's one-question discipline). The Baseten backend sends
   the same conversation to a configurable endpoint (OpenAI-compatible chat
   shape). The loop and artifact schema are identical for both backends.
4. **Credential injection:** standard environment variables (`AWS_*`,
   `BASETEN_API_KEY` + `BASETEN_MODEL_URL`, `RIME_API_KEY`), optionally loaded
   from a `--env-file` path (e.g. `../textlife-specs/.env`); secret values
   never appear in code, artifacts, or Git. Failure mode: partial artifact with
   explicit per-stage markers; process exits non-zero only on unexpected
   errors, not on missing optional credentials.
5. **Cost caps:** streaming Transcribe is bounded by the fixture — max 10
   seconds of audio per run (enforced in code). Polly synthesis is one short
   request. Total per-run AWS cost is sub-cent.
6. **"No standing AWS resource"** means no persistent infrastructure: no S3
   buckets, DynamoDB tables, Lambda functions, Connect instances, or phone
   numbers. Polly and streaming Transcribe are per-request API calls that
   require no created resources; batch Transcribe (which needs S3) is
   explicitly avoided for this reason.

## Constraints and non-goals

- No Amazon Connect phone provisioning or any standing-cost AWS resource in
  this slice; ask the customer before creating resources that cost money.
- Synthetic patient speech fixtures generated locally (Polly, pay-per-request)
  so no microphone is needed on this headless host.
- No PHI, no real patient audio, no clinical-use claims; keep the research-only
  disclaimer in all new docs.
- Secrets only via environment / `--env-file`; never committed to Git.
- No aLoRA work in this slice; the remote endpoint is a conventional
  single-adapter-style inference API behind a backend interface.

## Definition of enough

- `python -m voice_proto.run_loop` (or equivalent) produces a saved JSON
  artifact under `outputs/` containing: a real AWS Transcribe streaming
  transcript of the synthetic fixture (or explicit credential/denied marker),
  a model turn from the configured backend, TTS output (or explicit
  missing-credential marker), and per-stage latency timings.
- No TextLife AWS resource is read or written; no standing AWS resource is
  created.
- No secret values are committed to Git.
- A short doc records how to plug in Baseten and Rime credentials later.

## Follow-ups (out of this slice)

- Baseten endpoint wiring once credentials arrive; re-run loop with real model.
- Rime key wiring; verify the Rime API shape against live docs.
- Telephony slice (Connect or Telnyx voice) — requires explicit customer
  approval for phone-number spend.
- aLoRA cache-fork investigation stays on the TODO.md track.

## Intent

Prototype the `docs/DEPLOYMENT.md` voice call path on this host as a headless,
component-level loop: speech-to-text via AWS Transcribe (streaming), intake-model
inference via a remote Qwen endpoint (Baseten; customer is creating the account),
and text-to-speech via Rime Mist, with per-stage latency measurement. Research
only: synthetic/standardized-patient scenarios, no PHI, no diagnosis or treatment
behavior.

## Customer requirements

- Build the prototype using AWS and Rime, following the call path in
  `docs/DEPLOYMENT.md` (minus telephony for now).
- Customer is setting up a Baseten account and will supply credentials
  separately; the model backend must be pluggable so Baseten drops in later.
- AWS credentials come from the shared TextLife account
  (`../textlife-specs/.env`); use the account but never read, write, or depend
  on TextLife resources (DynamoDB table `textlife`, existing Lambdas, Telnyx
  SMS webhook).
- No Rime API key exists yet; build the TTS client and report a clear
  missing-credential marker until the customer provides one.

## Delivery intent

MVP: one command runs the loop end-to-end — synthetic patient speech fixture →
real AWS Transcribe streaming result → intake-model turn (deterministic mock
until Baseten credentials arrive) → Rime TTS (or explicit missing-credential
marker) — and saves an artifact with per-stage latencies.

## Constraints and non-goals

- No Amazon Connect phone provisioning or any standing-cost AWS resource in
  this slice; ask the customer before creating resources that cost money.
- Prefer streaming Transcribe over batch/S3 jobs so no S3 bucket or other
  AWS resource is created.
- Synthetic patient speech fixtures generated locally (e.g., via Amazon Polly,
  pay-per-request) so no microphone is needed on this headless host.
- No PHI, no real patient audio, no clinical-use claims; keep the research-only
  disclaimer in all new docs.
- Secrets only via environment / `--env-file`; never committed to Git.
- No aLoRA work in this slice; the remote endpoint is a conventional
  single-adapter-style inference API behind a backend interface.

## Definition of enough

- `python -m voice_proto.run_loop` (or equivalent) produces a saved JSON
  artifact under `outputs/` containing: a real AWS Transcribe streaming
  transcript of the synthetic fixture, a model turn from the configured
  backend, TTS output (or explicit missing-credential marker), and per-stage
  latency timings.
- No TextLife AWS resource is read or written; no standing AWS resource is
  created.
- No secret values are committed to Git.
- A short doc records how to plug in Baseten and Rime credentials later.

## Follow-ups (out of this slice)

- Baseten endpoint wiring once credentials arrive; re-run loop with real model.
- Rime key wiring; verify the Rime API shape against live docs.
- Telephony slice (Connect or Telnyx voice) — requires explicit customer
  approval for phone-number spend.
- aLoRA cache-fork investigation stays on the TODO.md track.

## Delivery intent

mvp

## Definition of enough

- [ ] The lead agent will replace this with observable MVP evidence before implementation.

## Decisions and assumptions

- None recorded yet.

## Score

- Requirements snapshot: `sha256:010c57eb2eb1eeb7c364648d606e2d73fdc93c834ed21e736dd7ac57c6c1889b`.
- Independent scores:
  - `glm` (fireworks/accounts/fireworks/models/glm-5p3): scope small; complexity medium; ambiguity medium; risk small. Scope is small: one CLI entrypoint, one synthetic fixture, one JSON artifact, one short doc. Complexity is medium: three external integrations (AWS Transcribe streaming, pluggable Baseten backend, Rime TTS) must be coordinated in a single loop with per-stage latency capture, plus a pluggable-backend seam and explicit missing-credential markers per stage. Ambiguity is medium: the package records no decisions or assumptions; the intake-model turn content/prompt is unspecified (even the deterministic mock's expected output is undefined), the synthetic fixture's content and audio format are unspecified, the artifact schema beyond 'transcript, model turn, TTS output, timings' is unspecified, and TTS 'output' (saved audio file vs. metadata) is unclear. Risk is small but not zero: all data is synthetic (no patient PII), no standing AWS resources are created, and secrets must not be committed; residual risk is limited to accidental paid-API usage once real credentials are plugged in and correct handling of the 'do not touch TextLife AWS resources' constraint, which is stated and testable.
  - `qwen` (fireworks/accounts/fireworks/models/qwen3p7-plus): scope medium; complexity medium; ambiguity medium; risk small. Scope is medium: the MVP bundles four distinct workstreams — synthetic audio fixture generation, AWS Transcribe streaming integration, a pluggable model backend (Baseten with deterministic mock fallback), and Rime TTS — plus artifact serialization and credential documentation. Complexity is medium: Transcribe streaming (WebSocket-based, chunked audio) is the hardest technical surface, and the pluggable-backend pattern adds abstraction, but there is no UI, no persistence layer, and no real-time multi-user concern. Ambiguity is medium: the package records 'Decisions and assumptions: None,' leaving several implementation choices unspecified — synthetic fixture format/encoding/sample-rate, the mock model's return shape, the exact credential-isolation mechanism (separate AWS profile vs. account), and the 'explicit missing-credential marker' schema. These are resolvable by the implementer with reasonable defaults for a prototype, so they do not block planning, but they are not decisions the scorer should make. Risk is small: the fixture is synthetic (no real patient audio), no standing AWS resources are created, secrets must not be committed, and the prototype makes no clinical claims. The 'no TextLife AWS resource' constraint is a clear guardrail that limits blast radius.
- Reconciled score: conservative maximum — scope medium; complexity medium; ambiguity medium; risk small.
- Customer decisions needed: What should the intake-model turn actually contain? Even the deterministic mock needs a target: is it a fixed canned string, a structured intake summary derived from the fixture transcript, or something else? This is the core observable output of the loop and is currently undefined. | What is the synthetic patient speech fixture — content, language, and audio format (duration, WAV vs. MP3, sample rate)? AWS Transcribe streaming behavior and latency numbers depend materially on this. | For the TTS stage, does 'TTS output' in the JSON artifact mean a saved audio file path, audio bytes/metadata, or just a success marker plus latency? This affects what 'enough' looks like at acceptance time.
- Scope: **medium**; complexity: **medium**; ambiguity: **medium**; risk: **small**.
- Work size: **medium** (the largest of scope, complexity, and ambiguity).
- Required roles: `implementation`, `evaluator`.

## Plan

Residual score blockers resolved against the sealed requirements (no
material disagreement; risk small): fixture format/content = Decision 1;
model-turn contract = Decisions 4 and 10; TTS artifact shape = Decision 9;
credential injection = Decision 6; AWS boundary verification = Decision 11.

- [x] T1 Scaffold: `.venv` with `boto3`, `amazon-transcribe`, `requests` —
      imports verified.
- [x] T2 `voice_proto` package: env-file loader, Polly fixture, streaming-
      Transcribe client, `ModelBackend` (mock + Baseten), Rime TTS client,
      `run_loop` orchestrator with per-stage latencies and marker states.
- [x] T3 Offline smoke run: complete artifact with explicit markers, no creds.
- [x] T4 Live AWS run with `--env-file ../textlife-specs/.env`: real Polly →
      Transcribe transcript in the artifact
      (`outputs/voice_proto/run_20260907T171757874Z.json`).
- [x] T5 Boundary + secrets check: only `polly:SynthesizeSpeech` and streaming
      Transcribe are called; no secret values in code/artifacts/Git.
- [x] T6 `docs/voice_prototype.md`: credential plug-in instructions for
      Baseten and Rime, research-only disclaimer.

## Status

Delivered (MVP slice). Awaiting Baseten and Rime credentials for follow-up
wiring; telephony requires explicit customer approval.

## Implementation log

- Feature created.
- 2026-09-07: T1-T6 delivered. Live evidence outputs/voice_proto/run_20260907T171757874Z.json: Polly fixture ok (674ms, 3.54s audio); streaming Transcribe ok (transcript accurate, 3710ms pacing-dominated, ttfp recorded); mock model turn ok; Rime missing-credential marker as expected; loop 4168ms. Boundary check: only polly:SynthesizeSpeech + streaming transcribe called; zero secret-value leaks in code/artifacts/docs; .env stays outside repo. Fixed one defect: silent sender-task failure (wrong SDK method name add_audio_event vs send_audio_event) surfaced via gathered task exceptions.

## Follow-ups

- None.
