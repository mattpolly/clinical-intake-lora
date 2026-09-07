"""Mock-only API tests for the research-only text demo service."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from baseten_demo.instrumentation import EventWriter
from baseten_demo.mock import MockBasetenClient
from demo_service import RESEARCH_DISCLAIMER
from demo_service.app import create_app
from demo_service.service import DemoService, DemoSettings, MAX_HISTORY_TOKENS


def _settings(tmp_path: Path) -> DemoSettings:
    return DemoSettings(
        api_key="test-key",
        model_id="test-model",
        deployment_id="test-deployment",
        output_dir=tmp_path,
        readiness_deadline_seconds=1,
        poll_interval_seconds=0.001,
    )


def _service(tmp_path: Path, client_factory=None) -> DemoService:
    return DemoService(
        _settings(tmp_path),
        client_factory=client_factory or (lambda: MockBasetenClient()),
        writer=EventWriter(tmp_path / "events.jsonl"),
    )


def _wait_ready(client: TestClient, session_id: str) -> dict:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        response = client.get(f"/demo/status/{session_id}")
        body = response.json()
        if body["state"] == "ready":
            return body
        time.sleep(0.005)
    raise AssertionError("mock deployment never became ready")


def test_full_mock_api_flow_records_lifecycle_without_patient_text(tmp_path):
    service = _service(tmp_path)
    client = TestClient(create_app(service))

    started = client.post("/demo/start")
    assert started.status_code == 200
    session_id = started.json()["session_id"]
    assert started.json()["research_only"] is True
    assert started.json()["disclaimer"] == RESEARCH_DISCLAIMER

    ready = _wait_ready(client, session_id)
    assert ready["state"] == "ready"
    turn = client.post(f"/demo/turn/{session_id}", json={"patient_text": "Synthetic headache for two days."})
    assert turn.status_code == 200
    assert turn.json()["turn"]["text"]
    assert turn.json()["research_only"] is True

    ended = client.post(f"/demo/end/{session_id}")
    assert ended.status_code == 200
    assert ended.json()["state"] == "ended"

    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "Synthetic headache" not in raw
    records = service.writer.read_records()
    assert {"session_started", "wake_requested", "session_ready", "turn_requested", "inference_result", "session_ended"}.issubset(
        {record["event"] for record in records}
    )
    assert all(len(line.encode("utf-8")) <= 4096 for line in raw.splitlines())


def test_turn_before_readiness_returns_stateful_503(tmp_path):
    service = _service(tmp_path)
    service._deployment_state = "waking"  # Simulate a shared lifecycle already in progress.
    client = TestClient(create_app(service))
    session_id = client.post("/demo/start").json()["session_id"]

    response = client.post(f"/demo/turn/{session_id}", json={"patient_text": "Synthetic scenario."})
    assert response.status_code == 503
    assert response.json()["detail"]["state"] == "waking"
    assert response.json()["detail"]["research_only"] is True


def test_session_history_and_input_are_bounded(tmp_path):
    service = _service(tmp_path)
    client = TestClient(create_app(service))
    session_id = client.post("/demo/start").json()["session_id"]
    _wait_ready(client, session_id)

    too_large = client.post(f"/demo/turn/{session_id}", json={"patient_text": "x" * 4_001})
    assert too_large.status_code == 422

    session = service._sessions[session_id]
    session.history = [{"role": "user", "content": "x" * (MAX_HISTORY_TOKENS * 4)}]
    capped = client.post(f"/demo/turn/{session_id}", json={"patient_text": "Synthetic text."})
    assert capped.status_code == 409
    assert capped.json()["detail"]["state"] == "ended"


def test_unknown_session_is_a_research_only_404(tmp_path):
    client = TestClient(create_app(_service(tmp_path)))
    response = client.get("/demo/status/not-a-session")
    assert response.status_code == 404
    assert response.json()["detail"]["research_only"] is True


def test_concurrent_session_starts_share_one_wake(tmp_path):
    class CountingClient(MockBasetenClient):
        def __init__(self):
            super().__init__()
            self.wake_calls = 0

        def wake(self):
            self.wake_calls += 1
            super().wake()

    backend = CountingClient()
    service = _service(tmp_path, client_factory=lambda: backend)
    client = TestClient(create_app(service))
    first = client.post("/demo/start").json()["session_id"]
    second = client.post("/demo/start").json()["session_id"]
    _wait_ready(client, first)
    assert client.get(f"/demo/status/{second}").json()["state"] == "ready"
    assert backend.wake_calls == 1


def test_inactive_session_expires_without_contacting_the_model(tmp_path):
    settings = _settings(tmp_path)
    settings = DemoSettings(**{**settings.__dict__, "session_timeout_seconds": 0.0})
    service = DemoService(settings, client_factory=lambda: MockBasetenClient(), writer=EventWriter(tmp_path / "events.jsonl"))
    service._deployment_state = "waking"
    client = TestClient(create_app(service))
    session_id = client.post("/demo/start").json()["session_id"]
    status = client.get(f"/demo/status/{session_id}").json()
    assert status["state"] == "expired"
