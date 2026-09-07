"""Harness integration + dry-run tests (full simulated event trace)."""

from pathlib import Path

import pytest

from baseten_demo.harness import Harness, HarnessConfig, WakeReadyInferResult
from baseten_demo.instrumentation import CANONICAL_EVENTS, EventWriter
from baseten_demo.mock import MockBasetenClient


def _harness(client=None, tmp_path=None, **config_overrides):
    writer = EventWriter(tmp_path / "events.jsonl" if tmp_path else Path("events.jsonl"))
    config = HarnessConfig(
        readiness_deadline_seconds=10,
        poll_interval_seconds=0.01,
        **config_overrides,
    )
    return Harness(client or MockBasetenClient(), writer, config=config,
                   profile_name="qwen3-8b-fp16"), writer


def test_dry_run_emits_complete_simulated_trace(tmp_path):
    harness, writer = _harness(tmp_path=tmp_path)
    result = harness.wake_ready_probe()
    assert result.ok
    # Simulate a demo turn then the idle interval elapsing.
    harness.infer_turn()
    harness.client.mark_idle_and_scale_to_zero()  # type: ignore[attr-defined]
    harness.observe_scale_to_zero(deadline_seconds=10)

    records = writer.read_records()
    events = [r["event"] for r in records]
    for canonical in (
        "wake_requested",
        "first_inference",
        "inference_result",
        "last_activity",
        "replica_ready",
        "scaled_to_zero",
    ):
        assert canonical in events, f"missing {canonical} in {events}"
    # Ordering reflects wake -> ready -> infer -> idle.
    assert events.index("wake_requested") < events.index("replica_ready")
    assert events.index("first_inference") < events.index("replica_ready")
    assert events.index("replica_ready") < events.index("scaled_to_zero")


def test_wake_to_ready_interval_is_computable(tmp_path):
    harness, writer = _harness(tmp_path=tmp_path)
    result = harness.wake_ready_probe()
    records = writer.read_records()
    wake_ts = result.wake_requested_at
    ready_ts = result.replica_ready["timestamp"]
    assert wake_ts.endswith("Z") and ready_ts.endswith("Z")
    # Both are UTC ISO strings; a consumer can diff them.
    from datetime import datetime

    wake = datetime.fromisoformat(wake_ts.replace("Z", "+00:00"))
    ready = datetime.fromisoformat(ready_ts.replace("Z", "+00:00"))
    assert ready >= wake


def test_all_records_parse_and_use_canonical_vocabulary(tmp_path):
    harness, writer = _harness(tmp_path=tmp_path)
    harness.wake_ready_probe()
    for record in writer.read_records():
        assert record["event"] in CANONICAL_EVENTS
        assert record["timestamp"].endswith("Z")


def test_readiness_timeout_emits_error_event(tmp_path):
    class NeverReady(MockBasetenClient):
        def is_ready(self):
            return False

    harness, writer = _harness(client=NeverReady(), tmp_path=tmp_path)
    result = harness.wake_ready_probe()
    assert result.ok is False
    assert result.state == "readiness_timeout"
    events = [r["event"] for r in writer.read_records()]
    assert "error" in events
    err = next(r for r in writer.read_records() if r["event"] == "error")
    assert err["error_type"] == "ReadinessTimeout"


def test_wake_transient_failure_retries_once_then_errors(tmp_path):
    class FlakyWake(MockBasetenClient):
        def __init__(self):
            super().__init__()
            self.wake_attempts = 0

        def wake(self):
            self.wake_attempts += 1
            from baseten_demo.client import TransientError

            raise TransientError("transient")

    client = FlakyWake()
    harness, writer = _harness(client=client, tmp_path=tmp_path)
    with pytest.raises(Exception):
        harness.wake()
    # Exactly one retry: two total attempts.
    assert client.wake_attempts == 2
    events = [r["event"] for r in writer.read_records()]
    assert events[0] == "wake_requested"
    assert "error" in events


def test_probe_error_state_returns_explicitly(tmp_path):
    class BadProbe(MockBasetenClient):
        def infer(self, *args, **kwargs):
            from baseten_demo.client import TransientError

            raise TransientError("probe failed")

    harness, writer = _harness(client=BadProbe(), tmp_path=tmp_path)
    result = harness.wake_ready_probe()
    assert result.ok is False
    assert result.state == "probe_error"
    assert any(r["event"] == "error" for r in writer.read_records())


def test_result_is_a_typed_struct():
    assert isinstance(
        WakeReadyInferResult(
            ok=True, state="ready", detail="", wake_requested_at="t"
        ),
        WakeReadyInferResult,
    )
