"""Baseten client tests against a mock of the documented HTTP interfaces."""

import json

import pytest

from baseten_demo.client import (
    AuthError,
    BasetenClient,
    RequestError,
    TransientError,
)


class FakeResponse:
    def __init__(self, status_code, payload=None, text="", lines=None,
                 content_type="application/json"):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._lines = lines or []
        self._content_type = content_type

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload

    def iter_lines(self):
        for line in self._lines:
            yield line


class FakeTransport:
    """Records requests and returns scripted responses keyed by path suffix."""

    def __init__(self):
        self.posts = []
        self.gets = []
        self.wake_response = FakeResponse(202, text="")
        self.status_response = FakeResponse(
            200, payload={"status": "ACTIVE", "active_replica_count": 1}
        )
        self.infer_response = FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    def post(self, url, *, headers, json=None, stream=False):
        self.posts.append((url, headers, json, stream))
        if url.endswith("/wake"):
            return self.wake_response
        return self.infer_response

    def get(self, url, *, headers):
        self.gets.append((url, headers))
        return self.status_response


def _client(transport=None, **overrides):
    kwargs = dict(
        api_key="test-key-123",
        model_id="m123",
        deployment_id="d456",
        environment="production",
        served_model_name="Qwen/Qwen3-8B",
    )
    kwargs.update(overrides)
    return BasetenClient(transport=transport or FakeTransport(), **kwargs)


def _streaming_lines(content="hello"):
    return [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
        b"data: [DONE]\n",
    ]


def test_wake_posts_to_documented_path_and_accepts_202():
    transport = FakeTransport()
    client = _client(transport)
    assert client.wake() is None
    url, headers, body, stream = transport.posts[0]
    assert url == "https://model-m123.api.baseten.co/deployment/d456/wake"
    assert headers["Authorization"] == "Bearer test-key-123"


def test_wake_rejects_401_with_auth_error():
    transport = FakeTransport()
    transport.wake_response = FakeResponse(401, text="Unauthorized")
    client = _client(transport)
    with pytest.raises(AuthError):
        client.wake()


def test_wake_429_is_transient():
    transport = FakeTransport()
    transport.wake_response = FakeResponse(429, text="try later")
    client = _client(transport)
    with pytest.raises(TransientError):
        client.wake()


def test_status_reads_documented_readiness_signal():
    transport = FakeTransport()
    client = _client(transport)
    assert client.is_ready() is True
    url, headers = transport.gets[0]
    assert url == "https://api.baseten.co/v1/models/m123/deployments/d456"
    assert headers["Authorization"] == "Bearer test-key-123"


def test_status_active_but_zero_replicas_is_not_ready():
    transport = FakeTransport()
    transport.status_response = FakeResponse(
        200, payload={"status": "ACTIVE", "active_replica_count": 0}
    )
    assert _client(transport).is_ready() is False


def test_infer_streaming_captures_ttfb_and_duration():
    transport = FakeTransport()
    transport.infer_response = FakeResponse(200, lines=_streaming_lines("hello"))
    client = _client(transport)
    result = client.infer([{"role": "user", "content": "hi"}], max_tokens=5)
    assert result["status"] == "ok"
    assert result["text"] == "hello"
    assert result["time_to_first_token_ms"] is not None
    assert result["duration_ms"] >= result["time_to_first_token_ms"]


def test_infer_uses_documented_chat_completions_url():
    transport = FakeTransport()
    client = _client(transport)
    client.infer([{"role": "user", "content": "hi"}], stream=False)
    url, headers, body, stream = transport.posts[0]
    assert url == (
        "https://model-m123.api.baseten.co/environments/production/"
        "sync/v1/chat/completions"
    )


def test_infer_non_streaming_parses_message():
    transport = FakeTransport()
    client = _client(transport)
    result = client.infer([{"role": "user", "content": "hi"}], stream=False)
    assert result["text"] == "ok"
    assert result["status"] == "ok"


def test_client_repr_does_not_leak_key():
    client = _client()
    assert "test-key-123" not in repr(client)
    assert "redacted" in repr(client)


def test_client_requires_key_and_ids():
    with pytest.raises(AuthError):
        BasetenClient(api_key="", model_id="m", deployment_id="d")
    with pytest.raises(RequestError):
        BasetenClient(api_key="k", model_id="", deployment_id="d")
    with pytest.raises(RequestError):
        BasetenClient(api_key="k", model_id="m", deployment_id="")


def test_infer_status_error_raises():
    transport = FakeTransport()
    transport.infer_response = FakeResponse(503, text="unavailable")
    client = _client(transport)
    with pytest.raises(TransientError):
        client.infer([{"role": "user", "content": "hi"}], stream=False)
