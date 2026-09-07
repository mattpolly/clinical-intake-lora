"""Baseten deployment-definition builder.

Builds the documented Baseten deployment artifacts for a profile **without
creating anything live**:

- a Truss ``config.yaml`` definition (documented model configuration format),
- the autoscaling settings payload (``PATCH .../autoscaling_settings``), and
- the inference/wake endpoint URLs derived from the documented inference host.

Every interface here is grounded in the current Baseten documentation cited in
``BASETEN_DOCS.md``. Field names use the documented Baseten spelling
(``min_replica`` / ``max_replica`` / ``scale_down_delay``) — not the plural
spelling that appears in the feature record.
"""

from __future__ import annotations

from typing import Any

from .profiles import Profile

# Documented Baseten endpoints (see BASETEN_DOCS.md).
MANAGEMENT_API_BASE = "https://api.baseten.co"


def inference_base_url(model_id: str) -> str:
    """Documented inference host for a deployed model.

    See https://docs.baseten.co/inference/calling-your-model — the predict
    endpoints live on ``https://model-{model_id}.api.baseten.co``.
    """
    return f"https://model-{model_id}.api.baseten.co"


def wake_url(model_id: str, deployment_id: str) -> str:
    """``POST .../deployment/{deployment_id}/wake`` (returns 202 Accepted)."""
    return f"{inference_base_url(model_id)}/deployment/{deployment_id}/wake"


def status_url(model_id: str, deployment_id: str) -> str:
    """``GET /v1/models/{model_id}/deployments/{deployment_id}`` (readiness)."""
    return f"{MANAGEMENT_API_BASE}/v1/models/{model_id}/deployments/{deployment_id}"


def autoscaling_url(model_id: str, deployment_id: str) -> str:
    """``PATCH /v1/models/{model_id}/deployments/{deployment_id}/autoscaling_settings``."""
    return (
        f"{MANAGEMENT_API_BASE}/v1/models/{model_id}/deployments/"
        f"{deployment_id}/autoscaling_settings"
    )


def chat_completions_url(model_id: str, environment: str = "production") -> str:
    """Documented OpenAI-compatible serving endpoint.

    The OpenAI-compatible base is
    ``https://model-{model_id}.api.baseten.co/environments/{environment}/sync/v1``;
    chat completions append ``/chat/completions`` (see the Qwen3 example and
    ``/inference/calling-your-model``).
    """
    return (
        f"{inference_base_url(model_id)}/environments/{environment}"
        f"/sync/v1/chat/completions"
    )


def build_autoscaling_settings(profile: Profile) -> dict[str, Any]:
    """Documented autoscaling payload (field names per the current SDK/docs).

    ``min_replica=0`` enables scale-to-zero; ``max_replica=1`` caps concurrent
    spend; ``scale_down_delay`` is the idle scale-down in seconds.
    """
    return {
        "min_replica": profile.min_replicas,
        "max_replica": profile.max_replicas,
        "autoscaling_window": profile.autoscaling_window_seconds,
        "scale_down_delay": profile.idle_timeout_seconds,
        "concurrency_target": profile.concurrency_target,
        "target_utilization_percentage": 70,
    }


def build_config_yaml(profile: Profile, base_image: str | None = None) -> dict[str, Any]:
    """Build the Truss ``config.yaml`` definition for the profile.

    The definition references pre-deployed model/adapter artifacts (weights
    mount + served model) and contains **no build, package, or redeployment
    step**: a later wake only provisions/loads an inference replica.

    The vLLM start command carries precision/quantization flags when the
    profile specifies a quantization mode; fp16 is the default (no flag).
    Adapter slots are represented as a list (aLoRA-compatible); mounting them
    into a start command is a training/branching follow-up because no
    8B-matched adapter exists on this host.
    """
    image = base_image or "vllm/vllm-openai:v0.27.1"

    config: dict[str, Any] = {
        "model_name": profile.name.replace("-", "_"),
        "model_metadata": {
            "tags": ["openai-compatible"],
            # Bounded metadata only; never transcripts or secrets.
            "example_model_input": {
                "messages": [
                    {"role": "system", "content": "You are a research-only prototype."},
                    {"role": "user", "content": "Answer with one word."},
                ],
                "max_tokens": 1,
                "temperature": 0.2,
            },
        },
        "base_image": {"image": image},
        "docker_server": {
            "start_command": _start_command(profile),
            "readiness_endpoint": "/health",
            "liveness_endpoint": "/health",
            "predict_endpoint": "/v1/chat/completions",
            "server_port": 8000,
        },
        "resources": {
            "accelerator": profile.gpu.gpu_class,
            "instance_type": profile.instance_type,
            "use_gpu": True,
        },
        "runtime": {
            "predict_concurrency": 256,
        },
    }

    if profile.weights is not None:
        config["weights"] = [
            {
                "source": profile.weights.source,
                "mount_location": profile.weights.mount_location,
            }
        ]

    if profile.adapters:
        # LoRA-ready: represent every adapter slot as a list entry. Multiple
        # slots are representable with no structural change (aLoRA compatible).
        config["adapters"] = list(profile.adapters)

    return config


def _start_command(profile: Profile) -> str:
    quant_flags = ""
    if profile.quantization:
        quant_flags = f" --quantization {profile.quantization}"
    mount = profile.weights.mount_location if profile.weights else "/models/"
    return (
        f"vllm serve {mount} --served-model-name {profile.served_model_name} "
        # Cap context length: vLLM's model default (40960 for Qwen3-8B) needs
        # more KV cache than fp16 weights leave free on a 24 GiB GPU
        # (deploy qkj5l7d failed on exactly this). 8192 covers intake dialogs.
        f"--max-model-len {profile.max_model_len} "
        f"--host 0.0.0.0 --port 8000{quant_flags}"
    )


def build_deployment_definition(
    profile: Profile, model_id: str, deployment_id: str, environment: str = "production"
) -> dict[str, Any]:
    """Full, explicit deployment definition for a profile.

    ``model_id`` / ``deployment_id`` are the Baseten identifiers assigned at
    the deferred live step (they are not secrets). The result is a plain
    dictionary suitable for inspection, dry-run, and the follow-up execution.
    """
    return {
        "profile": profile.name,
        "model_id": model_id,
        "deployment_id": deployment_id,
        "endpoints": {
            "wake": wake_url(model_id, deployment_id),
            "status": status_url(model_id, deployment_id),
            "autoscaling": autoscaling_url(model_id, deployment_id),
            "chat_completions": chat_completions_url(model_id, environment),
        },
        # No build/package/redeploy step: artifacts are pre-deployed.
        "autoscaling": build_autoscaling_settings(profile),
        "config": build_config_yaml(profile),
    }
