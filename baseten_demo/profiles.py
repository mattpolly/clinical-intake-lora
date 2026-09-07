"""Declarative model/GPU profile loading and validation.

Profiles live in ``baseten_demo/profiles.yaml`` (versioned). Each profile
carries the model identifier, precision/quantization, GPU class, autoscaling
bounds, idle scale-down timeout, and LoRA adapter slots (a list, so a later
aLoRA architecture needs no structural change).

Validation rejects impossible or out-of-range configuration explicitly:
``min_replicas > max_replicas`` and an idle timeout outside the closed
[120, 300] second range (approximately 2-5 minutes) are both load errors,
per the feature contract.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILES_PATH = Path(__file__).with_name("profiles.yaml")

# Closed 120-300 second idle scale-down window (approximately 2-5 minutes).
IDLE_TIMEOUT_MIN_SECONDS = 120
IDLE_TIMEOUT_MAX_SECONDS = 300

# Documented Baseten autoscaling bounds (see BASETEN_DOCS.md). min_replica >= 0
# (0 = scale to zero) and max_replica >= 1.
MIN_REPLICAS_MIN = 0
MAX_REPLICAS_MIN = 1

# Weight source schemes the deployment definition accepts (BDN).
_KNOWN_WEIGHT_SCHEMES = ("hf://", "bt://", "s3://", "gs://", "r2://", "cw://")


class ProfileValidationError(ValueError):
    """Raised when a profile fails contract validation."""


@dataclass(frozen=True)
class GpuSpec:
    gpu_class: str
    instance_type: str
    vram_gb: int


@dataclass(frozen=True)
class WeightsSpec:
    source: str
    mount_location: str


@dataclass(frozen=True)
class Profile:
    name: str
    model_id: str
    served_model_name: str
    precision: str
    quantization: str | None
    gpu: GpuSpec
    min_replicas: int
    max_replicas: int
    idle_timeout_seconds: int
    autoscaling_window_seconds: int
    concurrency_target: int
    max_model_len: int = 8192
    adapters: list[dict[str, Any]] = field(default_factory=list)
    weights: WeightsSpec | None = None

    @property
    def instance_type(self) -> str:
        return self.gpu.instance_type

    @property
    def vram_gb(self) -> int:
        return self.gpu.vram_gb


@dataclass(frozen=True)
class ProfileSet:
    profiles: dict[str, Profile]
    default: str
    version: int

    def get(self, name: str | None = None) -> Profile:
        key = name or self.default
        if key not in self.profiles:
            raise ProfileValidationError(
                f"unknown profile {key!r}; available: {sorted(self.profiles)}"
            )
        return self.profiles[key]


def _require(data: dict[str, Any], key: str, profile_name: str) -> Any:
    if key not in data or data[key] is None or data[key] == "":
        raise ProfileValidationError(f"profile {profile_name!r}: missing {key!r}")
    return data[key]


def _require_int(data: dict[str, Any], key: str, profile_name: str) -> int:
    value = _require(data, key, profile_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileValidationError(
            f"profile {profile_name!r}: {key!r} must be an integer, got {value!r}"
        )
    return value


def _require_str(data: dict[str, Any], key: str, profile_name: str) -> str:
    value = _require(data, key, profile_name)
    if not isinstance(value, str):
        raise ProfileValidationError(
            f"profile {profile_name!r}: {key!r} must be a string, got {value!r}"
        )
    return value


def _validate_gpu(gpu: Any, profile_name: str) -> GpuSpec:
    if not isinstance(gpu, dict):
        raise ProfileValidationError(f"profile {profile_name!r}: 'gpu' must be a mapping")
    gpu_class = _require_str(gpu, "class", profile_name)
    instance_type = _require_str(gpu, "instance_type", profile_name)
    vram = _require_int(gpu, "vram_gb", profile_name)
    if vram <= 0:
        raise ProfileValidationError(
            f"profile {profile_name!r}: gpu.vram_gb must be positive, got {vram}"
        )
    return GpuSpec(gpu_class, instance_type, vram)


def _validate_weights(weights: Any, profile_name: str) -> WeightsSpec | None:
    if weights is None:
        return None
    if not isinstance(weights, dict):
        raise ProfileValidationError(
            f"profile {profile_name!r}: 'weights' must be a mapping"
        )
    source = _require_str(weights, "source", profile_name)
    if not source.startswith(_KNOWN_WEIGHT_SCHEMES):
        raise ProfileValidationError(
            f"profile {profile_name!r}: weights.source {source!r} must use one of "
            f"{_KNOWN_WEIGHT_SCHEMES}"
        )
    mount = _require_str(weights, "mount_location", profile_name)
    if not mount.startswith("/"):
        raise ProfileValidationError(
            f"profile {profile_name!r}: weights.mount_location must be absolute"
        )
    return WeightsSpec(source, mount)


def _validate_adapters(adapters: Any, profile_name: str) -> list[dict[str, Any]]:
    if adapters is None:
        return []
    if not isinstance(adapters, list):
        raise ProfileValidationError(
            f"profile {profile_name!r}: 'adapters' must be a list "
            "(a single LoRA assumption is not allowed)"
        )
    validated: list[dict[str, Any]] = []
    for idx, slot in enumerate(adapters):
        if not isinstance(slot, dict):
            raise ProfileValidationError(
                f"profile {profile_name!r}: adapters[{idx}] must be a mapping"
            )
        if not slot.get("name"):
            raise ProfileValidationError(
                f"profile {profile_name!r}: adapters[{idx}] missing 'name'"
            )
        if not slot.get("source"):
            raise ProfileValidationError(
                f"profile {profile_name!r}: adapters[{idx}] missing 'source'"
            )
        validated.append(copy.deepcopy(slot))
    return validated


def validate_profile(name: str, data: dict[str, Any]) -> Profile:
    """Validate a single profile dict into a frozen :class:`Profile`.

    Raises :class:`ProfileValidationError` on any contract violation.
    """
    if not isinstance(data, dict):
        raise ProfileValidationError(f"profile {name!r}: must be a mapping")

    model_id = _require_str(data, "model_id", name)
    served_model_name = _require_str(data, "served_model_name", name)
    precision = _require_str(data, "precision", name)
    quantization = data.get("quantization")

    gpu = _validate_gpu(data.get("gpu"), name)
    min_replicas = _require_int(data, "min_replicas", name)
    max_replicas = _require_int(data, "max_replicas", name)
    idle_timeout = _require_int(data, "idle_timeout_seconds", name)
    window = _require_int(data, "autoscaling_window_seconds", name)
    concurrency = _require_int(data, "concurrency_target", name)
    max_model_len = data.get("max_model_len", 8192)
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise ProfileValidationError(
            f"profile {name!r}: max_model_len must be an integer, got {max_model_len!r}"
        )
    if not (1024 <= max_model_len <= 32768):
        raise ProfileValidationError(
            f"profile {name!r}: max_model_len {max_model_len} is outside the "
            "closed [1024, 32768] range"
        )

    if min_replicas < MIN_REPLICAS_MIN:
        raise ProfileValidationError(
            f"profile {name!r}: min_replicas must be >= {MIN_REPLICAS_MIN}, got {min_replicas}"
        )
    if max_replicas < MAX_REPLICAS_MIN:
        raise ProfileValidationError(
            f"profile {name!r}: max_replicas must be >= {MAX_REPLICAS_MIN}, got {max_replicas}"
        )
    if min_replicas > max_replicas:
        raise ProfileValidationError(
            f"profile {name!r}: min_replicas ({min_replicas}) exceeds "
            f"max_replicas ({max_replicas})"
        )
    if not (IDLE_TIMEOUT_MIN_SECONDS <= idle_timeout <= IDLE_TIMEOUT_MAX_SECONDS):
        raise ProfileValidationError(
            f"profile {name!r}: idle_timeout_seconds {idle_timeout} is outside the "
            f"closed [{IDLE_TIMEOUT_MIN_SECONDS}, {IDLE_TIMEOUT_MAX_SECONDS}] second "
            f"range (approximately 2-5 minutes)"
        )
    if concurrency < 1:
        raise ProfileValidationError(
            f"profile {name!r}: concurrency_target must be >= 1, got {concurrency}"
        )
    if window < 1:
        raise ProfileValidationError(
            f"profile {name!r}: autoscaling_window_seconds must be >= 1, got {window}"
        )

    return Profile(
        name=name,
        model_id=model_id,
        served_model_name=served_model_name,
        precision=precision,
        quantization=quantization,
        gpu=gpu,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        idle_timeout_seconds=idle_timeout,
        autoscaling_window_seconds=window,
        concurrency_target=concurrency,
        max_model_len=max_model_len,
        adapters=_validate_adapters(data.get("adapters"), name),
        weights=_validate_weights(data.get("weights"), name),
    )


def load_profiles(path: Path | str = DEFAULT_PROFILES_PATH) -> ProfileSet:
    """Load and validate the versioned profiles file into a :class:`ProfileSet`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProfileValidationError("profiles file must be a YAML mapping")
    if "profiles" not in raw or not isinstance(raw["profiles"], dict):
        raise ProfileValidationError("profiles file missing 'profiles' mapping")
    if not raw["profiles"]:
        raise ProfileValidationError("profiles file declares no profiles")
    default = raw.get("default")
    if default is None or default not in raw["profiles"]:
        raise ProfileValidationError(
            f"profiles file 'default' must name a declared profile, got {default!r}"
        )
    version = raw.get("version", 1)
    profiles = {
        name: validate_profile(name, data)
        for name, data in raw["profiles"].items()
    }
    return ProfileSet(profiles=profiles, default=default, version=version)
