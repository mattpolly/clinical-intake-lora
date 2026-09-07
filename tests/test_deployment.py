"""Deployment-definition builder tests."""

from baseten_demo.deployment import (
    build_autoscaling_settings,
    build_config_yaml,
    build_deployment_definition,
)
from baseten_demo.profiles import load_profiles


def _default():
    return load_profiles().get()


def test_autoscaling_settings_use_documented_field_names():
    profile = _default()
    settings = build_autoscaling_settings(profile)
    # Documented field names (singular), per BASETEN_DOCS.md.
    assert settings["min_replica"] == 0
    assert settings["max_replica"] == 1
    assert settings["scale_down_delay"] == 300
    assert "min_replicas" not in settings
    assert "max_replicas" not in settings


def test_config_yaml_has_no_build_or_redeploy_step():
    config = build_config_yaml(_default())
    assert "build" not in config
    assert "package" not in config
    # Pre-deployed artifacts: weights mount + served model, no build phase.
    assert config["weights"][0]["source"].startswith("hf://")
    assert "start_command" in config["docker_server"]
    assert "/v1/chat/completions" == config["docker_server"]["predict_endpoint"]


def test_default_resources_are_24gb_single_gpu():
    config = build_config_yaml(_default())
    # L4 is the org-supported A10G-class 24 GiB single GPU (A10G unavailable).
    assert config["resources"]["accelerator"] == "L4"
    assert config["resources"]["instance_type"] == "L4:4x16"
    assert config["resources"]["use_gpu"] is True


def test_start_command_caps_context_to_fit_24gb_kv_cache():
    command = build_config_yaml(_default())["docker_server"]["start_command"]
    # vLLM's 40960 default needs more KV cache than fp16 weights leave free.
    assert "--max-model-len 8192" in command


def test_14b_profile_switches_definition_without_code_change():
    p14 = load_profiles().get("qwen3-14b")
    settings = build_autoscaling_settings(p14)
    config = build_config_yaml(p14)
    assert settings["min_replica"] == 0
    assert settings["max_replica"] == 1
    assert config["resources"]["accelerator"] == "A100"
    assert config["resources"]["instance_type"] == "A100:12x144"
    assert config["weights"][0]["mount_location"] == "/models/qwen14b"


def test_deployment_definition_includes_endpoints():
    definition = build_deployment_definition(
        _default(), model_id="m123", deployment_id="d456"
    )
    endpoints = definition["endpoints"]
    assert endpoints["wake"].endswith("/deployment/d456/wake")
    assert endpoints["status"].endswith("/v1/models/m123/deployments/d456")
    assert endpoints["autoscaling"].endswith(
        "/v1/models/m123/deployments/d456/autoscaling_settings"
    )
    assert "/environments/production/sync/v1/chat/completions" in endpoints[
        "chat_completions"
    ]


def test_quantized_profile_emits_quantization_flag():
    p = load_profiles().get()
    import dataclasses

    q = dataclasses.replace(p, quantization="fp8")
    config = build_config_yaml(q)
    assert "--quantization fp8" in config["docker_server"]["start_command"]


def test_multiple_adapters_represented_without_structural_change():
    import dataclasses

    p = dataclasses.replace(
        load_profiles().get(),
        adapters=[
            {"name": "intake", "source": "hf://org/intake@main"},
            {"name": "safety", "source": "hf://org/safety@main"},
        ],
    )
    config = build_config_yaml(p)
    assert isinstance(config["adapters"], list)
    assert len(config["adapters"]) == 2
