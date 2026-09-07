"""Rime Mist TTS client.

Inert until RIME_API_KEY is provided (explicit missing-credential marker).
The endpoint defaults to Rime's public v1 rvoice URL and is env-overridable
(RIME_API_URL, RIME_MODEL, RIME_SPEAKER); the exact API shape must be
verified against live Rime docs once a key exists (see docs/voice_prototype.md).
Artifact contract (Decision 9): saved audio path + metadata, never embedded
bytes; the loop only persists audio (headless host).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

DEFAULT_API_URL = "https://api.rime.ai/v1/rvoice/tts"
DEFAULT_MODEL = "mist"
DEFAULT_SPEAKER = "eleanor"
TIMEOUT_S = 30


def synthesize(
    text: str,
    out_dir: Path = Path("outputs/voice_proto"),
) -> dict:
    result: dict = {
        "text": text,
        "status": "error",
        "error": None,
        "audio_path": None,
        "bytes": None,
        "content_type": None,
        "ttfb_ms": None,
        "latency_ms": None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("RIME_API_KEY")
    if not api_key:
        result["status"] = "missing-credential"
        return result

    url = os.environ.get("RIME_API_URL", DEFAULT_API_URL)
    body = {
        "model": os.environ.get("RIME_MODEL", DEFAULT_MODEL),
        "text": text,
        "speaker": os.environ.get("RIME_SPEAKER", DEFAULT_SPEAKER),
    }
    start = time.perf_counter()
    try:
        resp = requests.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT_S,
            stream=True,
        )
        ttfb = round((time.perf_counter() - start) * 1000, 1)
        result["ttfb_ms"] = ttfb
        result["content_type"] = resp.headers.get("Content-Type", "")
        chunks = []
        for chunk in resp.iter_content(chunk_size=8192):
            chunks.append(chunk)
        audio = b"".join(chunks)
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        if resp.status_code != 200:
            preview = audio[:200].decode("utf-8", "replace")
            result["error"] = f"HTTP {resp.status_code}: {preview}"
            return result
    except requests.RequestException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if audio and result["content_type"].startswith("audio/"):
        audio_path = out_dir / _audio_name(result["content_type"])
        audio_path.write_bytes(audio)
        result.update(status="ok", audio_path=str(audio_path), bytes=len(audio))
        return result

    # JSON-shaped response: record a bounded preview, never a whole payload.
    preview = audio[:300].decode("utf-8", "replace")
    result["status"] = "unexpected-format"
    result["error"] = f"non-audio body ({result['content_type']}): {preview}"
    return result


def _audio_name(content_type: str) -> str:
    subtype = content_type.split("/")[1].split(";")[0] if "/" in content_type else "bin"
    from .fixture import stamp

    return f"reply_{stamp()}.{subtype}"
