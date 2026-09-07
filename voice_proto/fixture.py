"""Synthetic patient-speech fixture via Amazon Polly (per-request, no created
AWS resources). One short scripted complaint sentence, ≤10 s, en-US,
16 kHz 16-bit mono PCM saved as WAV under outputs/ (Git-ignored).
"""

from __future__ import annotations

import io
import time
import wave
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

MAX_FIXTURE_SECONDS = 10
SAMPLE_RATE = 16_000

# Synthetic research scenario text (standardized-patient style; no real person).
DEFAULT_FIXTURE_TEXT = (
    "I've been having headaches for the past three days, mostly in the morning."
)


def synthesize_fixture(
    text: str = DEFAULT_FIXTURE_TEXT,
    region: str = "us-east-1",
    out_dir: Path = Path("outputs/voice_proto"),
) -> dict:
    """Synthesize one sentence with Polly; return PCM bytes + evidence dict."""
    result: dict = {
        "text": text,
        "status": "error",
        "error": None,
        "wav_path": None,
        "duration_s": None,
        "latency_ms": None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    creds_present = (
        os_env("AWS_ACCESS_KEY_ID") and os_env("AWS_SECRET_ACCESS_KEY")
    )
    if not creds_present:
        result["status"] = "missing-credential"
        return result

    start = time.perf_counter()
    try:
        polly = boto3.client("polly", region_name=region)
        resp = polly.synthesize_speech(
            Engine="standard",
            LanguageCode="en-US",
            OutputFormat="pcm",
            SampleRate=str(SAMPLE_RATE),
            Text=text,
            VoiceId="Joanna",
        )
        pcm = resp["AudioStream"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        msg = str(exc.response.get("Error", {}).get("Message", exc))
        if code in ("UnrecognizedClientException", "NoCredentialsError"):
            result["status"] = "missing-credential"
        elif code in ("AccessDeniedException", "AccessDenied", "NotAuthorizedException"):
            result["status"] = "access-denied"
        else:
            result["error"] = f"{code}: {msg}"
        return result
    except (BotoCoreError, OSError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)

    duration = len(pcm) / (SAMPLE_RATE * 2)  # 16-bit mono
    if duration > MAX_FIXTURE_SECONDS:
        pcm = pcm[: MAX_FIXTURE_SECONDS * SAMPLE_RATE * 2]
        duration = MAX_FIXTURE_SECONDS
    result["duration_s"] = round(duration, 2)

    wav_path = out_dir / f"fixture_{stamp()}.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    result.update(status="ok", wav_path=str(wav_path))
    return result


def read_wav_pcm(path: str | Path) -> bytes:
    """Return raw 16 kHz 16-bit mono PCM from a WAV file (header stripped)."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1
        return wf.readframes(wf.getnframes())


def os_env(key: str) -> str | None:
    import os

    return os.environ.get(key)


def stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
