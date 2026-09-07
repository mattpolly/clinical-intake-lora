"""Instrumentation event-writer tests."""

import json

import pytest

from baseten_demo.instrumentation import (
    CANONICAL_EVENTS,
    MAX_RECORD_BYTES,
    EventWriter,
    InstrumentationError,
    build_record,
    serialize_record,
)


def test_canonical_event_vocabulary_is_fixed():
    assert CANONICAL_EVENTS == frozenset(
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


def test_build_record_rejects_unknown_event():
    with pytest.raises(InstrumentationError, match="unknown event identifier"):
        build_record("mystery_event")


def test_record_carries_event_type_and_utc_timestamp():
    record = build_record("wake_requested", profile="qwen3-8b-fp16")
    assert record["event"] == "wake_requested"
    assert record["timestamp"].endswith("Z")
    assert "profile" in record


def test_writer_appends_one_json_object_per_line(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.emit("wake_requested", profile="p")
    writer.emit("replica_ready", profile="p")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert obj["event"] in CANONICAL_EVENTS
        assert obj["timestamp"].endswith("Z")


def test_read_records_roundtrip(tmp_path):
    writer = EventWriter(tmp_path / "e.jsonl")
    writer.emit("wake_requested")
    writer.emit("first_inference", time_to_first_token_ms=12.3)
    records = writer.read_records()
    assert [r["event"] for r in records] == ["wake_requested", "first_inference"]


def test_record_size_bound_enforced():
    # A non-string field (unbounded list) must still hit the 4096-byte bound.
    record = build_record("error", detail=["x" * 100] * 100)
    with pytest.raises(InstrumentationError, match="exceeding"):
        serialize_record(record)


def test_normal_records_are_well_within_bound():
    line = serialize_record(build_record("inference_result", duration_ms=42.7))
    assert len(line) <= MAX_RECORD_BYTES


def test_free_text_fields_are_bounded():
    record = build_record("error", message="y" * 99999)
    assert len(record["message"]) <= 3000  # sanitized before serialize
