#!/usr/bin/env python3
"""Fail-fast verification for the DeepSeek-V4-Flash container environment."""

from __future__ import annotations

import importlib.metadata
import os
import re
import shutil
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
    # Force the portable variant before anything can import kt_kernel.  Merely
    # checking that the file exists missed an AVX-512-contaminated AVX2 build.
    os.environ["KT_KERNEL_CPU_VARIANT"] = "avx2"
    os.environ["KT_MXFP4_BACKEND"] = "avx2"

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

    if kt_kernel.__cpu_variant__ != "avx2":
        raise RuntimeError(
            "DSV4 image validation forced the AVX2 kt-kernel variant, "
            f"but loaded {kt_kernel.__cpu_variant__!r}"
        )

    verify_avx2_instruction_contract(avx2_variants[0])

    # Exercise the exact native constructor that previously raised SIGILL on
    # AVX2-only hosts.  Keep this after the binary scan so an invalid artifact
    # fails with a useful instruction address on AVX-512 build machines too.
    worker_config = kt_kernel.kt_kernel_ext.WorkerPoolConfig()
    worker_config.subpool_count = 1
    worker_config.subpool_numa_map = [0]
    worker_config.subpool_thread_count = [1]
    cpu_infer = kt_kernel.kt_kernel_ext.CPUInfer(worker_config)
    cpu_infer.sync()
    del cpu_infer

    print(
        "DSV4 environment OK:",
        f"torch={torch.__version__}",
        f"cuda={torch.version.cuda}",
        f"transformers={transformers.__version__}",
        f"avx2_variant={avx2_variants[0].name}",
        "avx2_cpuinfer=ok",
    )


def verify_avx2_instruction_contract(shared_object: Path) -> None:
    """Reject EVEX-encoded instructions from the AVX2 shared object."""
    objdump = shutil.which("objdump")
    if objdump is None:
        raise RuntimeError("objdump is required to validate the AVX2 kt-kernel variant")

    # In 64-bit mode an instruction beginning with byte 0x62 uses the EVEX
    # prefix.  AVX2 uses VEX, so EVEX in executable code proves that the binary
    # requires an AVX-512-family feature regardless of the printed mnemonic.
    evex_instruction = re.compile(r"^\s*[0-9a-f]+:\s+62(?:\s|$)", re.IGNORECASE)
    process = subprocess.Popen(
        [objdump, "-d", str(shared_object)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    offending_line = None
    for line in process.stdout:
        if evex_instruction.search(line):
            offending_line = line.strip()
            process.terminate()
            break

    if offending_line is None:
        stderr = process.communicate()[1]
        if process.returncode != 0:
            raise RuntimeError(
                f"objdump failed while validating {shared_object.name}: {stderr.strip()}"
            )
        return

    process.communicate()
    raise RuntimeError(
        f"AVX2 kt-kernel variant contains an EVEX/AVX-512 instruction: {offending_line}"
    )


if __name__ == "__main__":
    verify_versions()
    verify_dependency_metadata()
    verify_imports()
