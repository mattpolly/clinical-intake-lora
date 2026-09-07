#!/usr/bin/env python3
"""Report the Python, package, CUDA, GPU, and bitsandbytes state for this project."""

from __future__ import annotations

import importlib.metadata
import platform
import sys


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"python_executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    for package in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "datasets",
        "accelerate",
        "huggingface_hub",
    ):
        print(f"{package}: {package_version(package)}")

    import torch

    print(f"torch_cuda: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return

    props = torch.cuda.get_device_properties(0)
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"compute_capability: {props.major}.{props.minor}")
    print(f"vram_gib: {props.total_memory / 1024**3:.2f}")

    try:
        import bitsandbytes as bnb

        layer = bnb.nn.Linear8bitLt(4, 2, bias=False, has_fp16_weights=False).cuda()
        value = layer(torch.randn(2, 4, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
        assert value.shape == (2, 2)
        print("bitsandbytes_cuda_smoke_test: PASS")
    except ImportError:
        print("bitsandbytes_cuda_smoke_test: SKIPPED (bitsandbytes not installed)")


if __name__ == "__main__":
    main()
