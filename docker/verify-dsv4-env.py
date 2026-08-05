#!/usr/bin/env python3
"""Fail-fast verification for the DeepSeek-V4-Flash container environment."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "torch": "2.9.1+cu128",
    "torchvision": "0.24.1+cu128",
    "torchaudio": "2.9.1+cu128",
    "sgl-kernel": "0.3.21",
    "flashinfer-python": "0.6.9",
    "flashinfer-cubin": "0.6.9",
    "transformers": "4.57.1",
    "tilelang": "0.1.10",
    "apache-tvm-ffi": "0.1.11",
    "cuda-python": "13.2.0",
    "nvidia-cutlass-dsl": "4.6.1",
}


def verify_versions() -> None:
    for distribution, expected in EXPECTED.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"{distribution} version mismatch: expected {expected}, got {actual}"
            )

    try:
        importlib.metadata.version("transformers-kt")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "transformers-kt must not be installed: it conflicts with transformers 4.57.1"
        )


def verify_dependency_metadata() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    messages = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip()
    ]
    unexpected = []
    for message in messages:
        normalized = message.lower().replace("_", "-")
        intentional = normalized.startswith("sglang-kt ") and any(
            name in normalized
            for name in ("flashinfer-python", "flashinfer-cubin", "transformers-kt")
        )
        intentional = intentional or (
            normalized.startswith("quack-kernels ")
            and "nvidia-cutlass-dsl" in normalized
        )
        if not intentional:
            unexpected.append(message)

    if unexpected:
        raise RuntimeError(
            "Unexpected dependency metadata failures:\n  " + "\n  ".join(unexpected)
        )


def verify_imports() -> None:
    import flashinfer  # noqa: F401
    import kt_kernel  # noqa: F401
    import sglang  # noqa: F401
    import tilelang  # noqa: F401
    import torch
    import transformers
    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config  # noqa: F401

    # DSV4's default MXFP4 path must remain portable to AVX2-only hosts.  The
    # entrypoint selects this variant at runtime when AVX512/AMX is absent.
    package_dir = Path(kt_kernel.__file__).resolve().parent
    avx2_variants = list(package_dir.glob("_kt_kernel_ext_avx2.*.so"))
    if not avx2_variants:
        raise RuntimeError(
            "DSV4 image is missing the AVX2 kt-kernel variant; "
            "build with CPUINFER_BUILD_ALL_VARIANTS=1"
        )

    print(
        "DSV4 environment OK:",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
        f"transformers={transformers.__version__}",
        f"avx2_variant={avx2_variants[0].name}",
    )


if __name__ == "__main__":
    verify_versions()
    verify_dependency_metadata()
    verify_imports()
