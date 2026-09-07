"""Profile loading/validation tests."""

import pytest

from baseten_demo.profiles import (
    IDLE_TIMEOUT_MAX_SECONDS,
    IDLE_TIMEOUT_MIN_SECONDS,
    ProfileValidationError,
    load_profiles,
    validate_profile,
)


def test_default_profile_matches_customer_decisions():
    profiles = load_profiles()
    default = profiles.get()
    assert default.name == "qwen3-8b-fp16"
    assert default.model_id == "Qwen/Qwen3-8B"
    assert default.precision == "fp16"
    assert default.quantization is None
    assert default.gpu.gpu_class == "A10G"
    assert default.gpu.vram_gb == 24
    assert default.instance_type == "A10Gx8x32"
    assert default.min_replicas == 0
    assert default.max_replicas == 1
    assert default.idle_timeout_seconds == 300
    assert default.adapters == []


def test_profiles_file_ships_a_concrete_14b_profile():
    profiles = load_profiles()
    p14 = profiles.get("qwen3-14b")
    assert p14.model_id == "Qwen/Qwen3-14B"
    assert p14.name == "qwen3-14b"
    # Switchable without any application code change: same loader path.
    assert p14.max_replicas == 1
    assert p14.min_replicas == 0
    assert p14.gpu.vram_gb == 80  # 14B fp16 does not fit 24 GiB


def test_switching_profile_requires_only_config_selection():
    profiles = load_profiles()
    for name in ("qwen3-8b-fp16", "qwen3-14b"):
        profile = profiles.get(name)
        assert profile.name == name


def _valid_profile(**overrides):
    base = {
        "model_id": "Qwen/Qwen3-8B",
        "served_model_name": "Qwen/Qwen3-8B",
        "precision": "fp16",
        "quantization": None,
        "gpu": {"class": "A10G", "instance_type": "A10Gx8x32", "vram_gb": 24},
        "min_replicas": 0,
        "max_replicas": 1,
        "idle_timeout_seconds": 300,
        "autoscaling_window_seconds": 60,
        "concurrency_target": 1,
        "adapters": [],
        "weights": {
            "source": "hf://Qwen/Qwen3-8B@abc123",
            "mount_location": "/models/qwen",
        },
    }
    base.update(overrides)
    return base


def test_min_replicas_greater_than_max_is_rejected():
    with pytest.raises(ProfileValidationError, match="exceeds"):
        validate_profile("bad", _valid_profile(min_replicas=2, max_replicas=1))


def test_idle_timeout_below_range_is_rejected():
    with pytest.raises(ProfileValidationError, match="outside the closed"):
        validate_profile(
            "bad", _valid_profile(idle_timeout_seconds=IDLE_TIMEOUT_MIN_SECONDS - 1)
        )


def test_idle_timeout_above_range_is_rejected():
    with pytest.raises(ProfileValidationError, match="outside the closed"):
        validate_profile(
            "bad", _valid_profile(idle_timeout_seconds=IDLE_TIMEOUT_MAX_SECONDS + 1)
        )


def test_idle_timeout_bounds_inclusive():
    assert validate_profile(
        "lo", _valid_profile(idle_timeout_seconds=IDLE_TIMEOUT_MIN_SECONDS)
    ).idle_timeout_seconds == IDLE_TIMEOUT_MIN_SECONDS
    assert validate_profile(
        "hi", _valid_profile(idle_timeout_seconds=IDLE_TIMEOUT_MAX_SECONDS)
    ).idle_timeout_seconds == IDLE_TIMEOUT_MAX_SECONDS


def test_missing_model_id_is_rejected():
    with pytest.raises(ProfileValidationError, match="model_id"):
        data = _valid_profile()
        data.pop("model_id")
        validate_profile("bad", data)


def test_adapters_must_be_a_list_not_a_single_mapping():
    with pytest.raises(ProfileValidationError, match="list"):
        validate_profile("bad", _valid_profile(adapters={"name": "one-lora"}))


def test_multiple_adapter_slots_are_representable():
    data = _valid_profile(
        adapters=[
            {"name": "intake", "source": "hf://org/intake@main"},
            {"name": "safety", "source": "hf://org/safety@main"},
        ]
    )
    profile = validate_profile("multi", data)
    assert len(profile.adapters) == 2


def test_bad_weights_scheme_is_rejected():
    with pytest.raises(ProfileValidationError, match="weights.source"):
        validate_profile(
            "bad",
            _valid_profile(
                weights={"source": "https://example.com/x", "mount_location": "/m"}
            ),
        )
