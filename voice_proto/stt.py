"""Streaming AWS Transcribe STT for raw 16 kHz 16-bit mono PCM.

Latency contract (feature Decision 5): the measured interval starts when the
first audio byte is written to the stream and ends when the final transcript
event arrives; time-to-first-partial is recorded separately. Pre-recorded
chunks are rate-limited to real time as required by the streaming service.
"""

from __future__ import annotations

import asyncio
import time

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from .fixture import SAMPLE_RATE

CHUNK_BYTES = 16_000  # 0.5 s of 16 kHz 16-bit mono
STREAM_TIMEOUT_S = 90


class _Collector(TranscriptResultStreamHandler):
    def __init__(self, stream):
        super().__init__(stream)
        self.final_transcript = ""
        self.partial_count = 0
        self.first_event_at: float | None = None

    async def handle_transcript_event(self, event: TranscriptEvent) -> None:
        self.first_event_at = self.first_event_at or time.perf_counter()
        for result in event.transcript.results:
            if result.is_partial:
                self.partial_count += 1
            elif result.alternatives:
                self.final_transcript += result.alternatives[0].transcript + " "


async def _run(pcm: bytes, region: str) -> dict:
    result: dict = {
        "status": "error",
        "error": None,
        "transcript": None,
        "partial_count": 0,
        "ttfp_ms": None,
        "latency_ms": None,
    }
    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm",
    )

    collector = _Collector(stream.output_stream)
    started = time.perf_counter()

    async def send_audio() -> None:
        bytes_per_second = SAMPLE_RATE * 2
        for offset in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[offset : offset + CHUNK_BYTES]
            await stream.input_stream.send_audio_event(chunk)
            # real-time pace, slightly ahead to cover send overhead
            await asyncio.sleep(max(0.0, len(chunk) / bytes_per_second - 0.02))
        await stream.input_stream.end_stream()

    sender = asyncio.create_task(send_audio())
    handler = asyncio.create_task(collector.handle_events())
    try:
        outcomes = await asyncio.wait_for(
            asyncio.gather(sender, handler, return_exceptions=True),
            timeout=STREAM_TIMEOUT_S,
        )
        for outcome in outcomes:  # never swallow sender/handler failures
            if isinstance(outcome, BaseException):
                raise outcome
    except asyncio.TimeoutError:
        result["error"] = f"stream timed out after {STREAM_TIMEOUT_S}s"
    finally:
        for task in (sender, handler):
            task.cancel()

    if collector.first_event_at is not None:
        result["ttfp_ms"] = round((collector.first_event_at - started) * 1000, 1)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    transcript = collector.final_transcript.strip()
    if transcript:
        result.update(status="ok", transcript=transcript,
                      partial_count=collector.partial_count)
    elif result["error"] is None:
        result["error"] = "no final transcript returned"
    return result


def transcribe_pcm(pcm: bytes, region: str = "us-east-1") -> dict:
    """Synchronous wrapper around the streaming-Transcribe coroutine."""
    try:
        return asyncio.run(_run(pcm, region))
    except Exception as exc:  # classify auth-shaped failures as markers
        msg = f"{type(exc).__name__}: {exc}"
        lowered = msg.lower()
        if "credential" in lowered or "not signed" in lowered:
            return {"status": "missing-credential", "error": msg, "transcript": None,
                    "partial_count": 0, "ttfp_ms": None, "latency_ms": None}
        if "access" in lowered and "denied" in lowered:
            return {"status": "access-denied", "error": msg, "transcript": None,
                    "partial_count": 0, "ttfp_ms": None, "latency_ms": None}
        return {"status": "error", "error": msg, "transcript": None,
                "partial_count": 0, "ttfp_ms": None, "latency_ms": None}
