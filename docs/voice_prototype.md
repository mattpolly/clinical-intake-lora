# Voice intake prototype (AWS + Rime) — headless loop

> **Research only.** Synthetic/standardized-patient scenarios, no PHI, no
> diagnosis or treatment behavior, and no clinical-use claim. This prototype
> measures plumbing and latency, not medical capability.

This is the first implementation slice of the voice-serving path sketched in
[DEPLOYMENT.md](DEPLOYMENT.md), minus telephony. It is a **one-shot, headless
loop** — one utterance → one intake-model turn → one TTS output — designed so
the Baseten and Rime credentials can be dropped in later without code changes.

## Pipeline

```text
scripted synthetic-patient sentence
        ↓  Amazon Polly (standard voice, 16 kHz PCM)      [fixture, timed separately]
patient audio (≤10 s WAV)
        ↓  AWS Transcribe streaming (pcm/16000/en-US)     [STT stage]
transcript
        ↓  intake model turn                              [mock | Baseten Qwen]
one concise question or summary
        ↓  Rime Mist TTS                                  [inert until API key]
reply audio (persisted, never played; headless host)
```

Run it:

```bash
source .venv/bin/activate

# Offline smoke test (no credentials needed; stages get explicit markers):
python -m voice_proto.run_loop --skip-aws

# Live AWS fixture + streaming transcription:
python -m voice_proto.run_loop --env-file ../textlife-specs/.env
```

Every run writes a complete JSON artifact to `outputs/voice_proto/run_*.json`
with per-stage status (`ok`, `missing-credential`, `access-denied`, `skipped`,
`error`), latencies, and the texts/audio paths involved. The loop never
silently aborts on a missing optional credential.

## Latency definitions (fixed for comparability)

- **STT:** first audio byte written to the stream → final transcript event
  (time-to-first-partial recorded separately). Pre-recorded fixtures are sent
  at real-time pace, so STT latency includes the fixture duration; treat
  `fixture.duration_s` as the pacing floor.
- **Model:** HTTP request sent → full body received (time-to-first-byte
  recorded separately). Non-streaming.
- **TTS:** HTTP request sent → full audio received (time-to-first-byte
  recorded separately).
- **Fixture (Polly):** timed separately, never counted in loop latency.

## Plugging in credentials

Credentials come from environment variables (optionally via `--env-file`,
which never overwrites real env vars). No secret values are stored in code,
artifacts, or Git.

### Baseten (intake model)

```bash
export BASETEN_API_KEY=...          # from your Baseten account
export BASETEN_MODEL_URL=https://model-XXXX.api.baseten.co/environments/production
export BASETEN_MODEL_ID=...         # optional
python -m voice_proto.run_loop --backend baseten --env-file ../textlife-specs/.env
```

The backend sends an OpenAI-compatible chat-completions request
(`system` + `patient→user` / `intake→assistant` messages, temperature 0.2) to
`$BASETEN_MODEL_URL/v1/chat/completions` and expects
`choices[0].message.content`. With no Baseten credentials the loop
automatically uses the deterministic mock backend, which returns a single
scripted follow-up question.

### Rime Mist (reply TTS)

```bash
export RIME_API_KEY=...             # from rime.ai
# optional overrides:
export RIME_API_URL=...             # default: https://api.rime.ai/v1/rvoice/tts
export RIME_MODEL=mist              # default
export RIME_SPEAKER=eleanor         # default
```

Until a key is present the TTS stage records `missing-credential`. **The
default endpoint and payload shape must be verified against live Rime
documentation once the key exists** — the client accepts an audio-bytes
response (saved under `outputs/`) and reports a bounded preview of any
unexpected JSON shape rather than embedding payloads in the artifact.

## Cost and AWS boundary

- Only two AWS APIs are used, both per-request with no created infrastructure:
  `polly:SynthesizeSpeech` and streaming
  `transcribe:StartStreamTranscription`. No S3, DynamoDB, Lambda, Connect, or
  phone numbers are touched, and no TextLife resource is read or written.
- Each run costs well under one cent (one Polly request + one ≤10 s streaming
  transcription, both capped in code).

## Evidence from the first live run

Live run `run_20260907T171757874Z` (shared account, `us-east-1`): Polly
fixture 674 ms (3.54 s audio); streaming Transcribe returned
"I've been having headaches for the past 3 days, mostly in the morning." for
the fixture "…for the past three days…" in 3,710 ms (pacing-dominated);
mock model turn OK; Rime correctly marked `missing-credential`; loop total
4,168 ms.

## Follow-ups (out of this slice)

1. Wire the Baseten endpoint when credentials arrive; re-run with a real
   Qwen + adapter behind the same backend interface.
2. Verify the Rime API shape with a live key; record real TTS latency.
3. Continuous multi-turn conversational loop with session state and
   interruption handling.
4. Telephony (Amazon Connect or Telnyx voice) — requires explicit approval
   for phone-number spend and BAA confirmation.
5. aLoRA cache-fork investigation remains on its own [TODO.md](TODO.md)
   track.
