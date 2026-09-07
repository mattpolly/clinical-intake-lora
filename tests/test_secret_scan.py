"""Secret-scan tests: the Baseten API key never reaches artifacts/fixtures."""

import json
from pathlib import Path

import pytest

from baseten_demo.client import BasetenClient, TransientError
from baseten_demo.harness import Harness, HarnessConfig
from baseten_demo.instrumentation import EventWriter
from baseten_demo.mock import MockBasetenClient

TEST_KEY = "sk-baseten-SUPER-secret-9f8e7d6c5b4a"


class SecretFakeTransport:
    """Returns a failure that echoes nothing back about the key."""

    def __init__(self):
        self.response = _FakeResp(503, "unavailable")

    def post(self, url, *, headers, json=None, stream=False):
        return self.response

    def get(self, url, *, headers):
        return self.response


class _FakeResp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

    def json(self):
        raise ValueError("not json")


def test_key_never_appears_in_error_events(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl")
    client = BasetenClient(
        api_key=TEST_KEY,
        model_id="m123",
        deployment_id="d456",
        transport=SecretFakeTransport(),
    )
    harness = Harness(client, writer, config=HarnessConfig(
        readiness_deadline_seconds=1, poll_interval_seconds=0.01
    ))
    with pytest.raises(Exception):
        harness.wake()

    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert TEST_KEY not in raw
    assert "Bearer" not in raw  # authorization header value never serialized
    # Every line still parses and has a canonical event + UTC timestamp.
    for line in raw.splitlines():
        if line.strip():
            obj = json.loads(line)
            assert obj["event"] in {"wake_requested", "error"}
            assert obj["timestamp"].endswith("Z")


def test_client_repr_and_errors_do_not_contain_key():
    client = BasetenClient(
        api_key=TEST_KEY,
        model_id="m123",
        deployment_id="d456",
        transport=SecretFakeTransport(),
    )
    assert TEST_KEY not in repr(client)
    with pytest.raises(TransientError) as info:
        client.wake()
    assert TEST_KEY not in str(info.value)


def test_full_dry_run_artifact_is_secret_free(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl")
    harness = Harness(
        MockBasetenClient(),
        writer,
        config=HarnessConfig(readiness_deadline_seconds=10, poll_interval_seconds=0.01),
    )
    harness.wake_ready_probe()
    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert TEST_KEY not in raw
    assert "sk-baseten" not in raw
