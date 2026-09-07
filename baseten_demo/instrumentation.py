"""Cold-start instrumentation: append-only JSON-lines event writer.

One JSON object per line under ``outputs/baseten_demo/events_*.jsonl``. Each
record carries a canonical ``event`` identifier and a UTC ``timestamp`` and is
no larger than 4096 serialized bytes. No secrets and no full conversation
transcripts are ever written.

Canonical event vocabulary (feature contract decision 10):

    wake_requested   UTC timestamp when the backend issues the wake call
    replica_ready    UTC timestamp when readiness is proven (health signal + probe)
    first_inference  UTC timestamp of the first (trivial probe) inference request
    inference_result duration and time-to-first-token for each inference
    last_activity    UTC timestamp of the most recent inference request
    scaled_to_zero   UTC timestamp when scale-down is observed (best-effort)
    error            explicit error state when wake/readiness/inference fails
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CANONICAL_EVENTS = frozenset(
    {
        "wake_requested",
        "replica_ready",
        "first_inference",
        "inference_result",
        "last_activity",
        "scaled_to_zero",
        "error",
        "session_started",
        "session_ready",
        "turn_requested",
        "session_ended",
    }
)

MAX_RECORD_BYTES = 4096


class InstrumentationError(ValueError):
    """Raised when an event cannot be emitted within the contract bounds."""


def utc_now_iso() -> str:
    """Current UTC wall-clock time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_field(value: Any, max_len: int = 2000) -> Any:
    """Bound free-text fields so a record never carries unbounded content."""
    if isinstance(value, str):
        return value[:max_len]
    return value


def build_record(event: str, fields: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """Build a canonical event record (event + UTC timestamp + fields)."""
    if event not in CANONICAL_EVENTS:
        raise InstrumentationError(
            f"unknown event identifier {event!r}; canonical vocabulary: "
            f"{sorted(CANONICAL_EVENTS)}"
        )
    record: dict[str, Any] = {
        "event": event,
        "timestamp": utc_now_iso(),
    }
    merged = dict(fields or {})
    merged.update(extra)
    for key, value in merged.items():
        record[key] = _sanitize_field(value)
    return record


def serialize_record(record: dict[str, Any]) -> bytes:
    """Serialize a record and enforce the 4096-byte single-line bound."""
    line = json.dumps(record, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if b"\n" in line:
        raise InstrumentationError("event record must not contain a newline")
    if len(line) > MAX_RECORD_BYTES:
        raise InstrumentationError(
            f"event record is {len(line)} bytes, exceeding the "
            f"{MAX_RECORD_BYTES}-byte bound"
        )
    return line


class EventWriter:
    """Append-only JSON-lines writer for lifecycle events."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record = build_record(event, fields)
        line = serialize_record(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(line + b"\n")
        return record

    def read_records(self) -> list[dict[str, Any]]:
        """Read back every event currently written (for tests/evidence)."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                records.append(json.loads(raw))
        return records
