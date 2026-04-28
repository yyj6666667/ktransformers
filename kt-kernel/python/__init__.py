# KT-Kernel: High-performance kernel operations for KTransformers
# SPDX-License-Identifier: Apache-2.0

"""
KT-Kernel provides high-performance kernel operations for KTransformers,
including CPU-optimized MoE inference with AMX, AVX, and KML support.

The package automatically detects your CPU capabilities and loads the optimal
kernel variant (AMX, AVX512, or AVX2) at runtime.

Example usage:
    >>> from kt_kernel import KTMoEWrapper
    >>> wrapper = KTMoEWrapper(
    ...     layer_idx=0,
    ...     num_experts=8,
    ...     num_experts_per_tok=2,
    ...     hidden_size=4096,
    ...     moe_intermediate_size=14336,
    ...     num_gpu_experts=2,
    ...     cpuinfer_threads=32,
    ...     threadpool_count=2,
    ...     weight_path="/path/to/weights",
    ...     chunked_prefill_size=512,
    ...     method="AMXINT4"
    ... )

    Check which CPU variant is loaded:
    >>> import kt_kernel
    >>> print(kt_kernel.__cpu_variant__)  # 'amx', 'avx512', or 'avx2'

Environment Variables:
    KT_KERNEL_CPU_VARIANT: Override automatic detection ('amx', 'avx512', 'avx2')
    KT_KERNEL_DEBUG: Enable debug output ('1' to enable)
"""

from __future__ import annotations

# Detect CPU and load optimal extension variant
from ._cpu_detect import initialize as _initialize_cpu

_kt_kernel_ext, __cpu_variant__ = _initialize_cpu()

# Make the extension module available to other modules in this package
import sys

sys.modules["kt_kernel_ext"] = _kt_kernel_ext
# Also register under the fully-qualified `kt_kernel.kt_kernel_ext` name so
# that user-side `from kt_kernel.kt_kernel_ext.moe import ...` (or any other
# fully-qualified import) hits the cached module. Without this entry, on
# Windows where the .pyd lives at `kt_kernel/kt_kernel_ext.cp312-*.pyd`,
# Python's native import machinery would load it a *second* time as a
# package submodule and trip pybind11's "type already registered" guard.
# (Linux is unaffected when the multi-variant naming
# `_kt_kernel_ext_<variant>.cpython-*.so` is used because the file name on
# disk doesn't collide with `kt_kernel.kt_kernel_ext`.)
sys.modules["kt_kernel.kt_kernel_ext"] = _kt_kernel_ext

# pybind11 nested submodules (m.def_submodule(...)) live as attributes on
# the parent module but don't have a __path__, so Python's import machinery
# can't resolve `kt_kernel.kt_kernel_ext.moe` as a dotted import. Register
# every direct submodule under its fully-qualified name in sys.modules so
# `from kt_kernel.kt_kernel_ext.moe import ...` is a cache hit. Recursive,
# in case future builds add nested-nested submodules.
import types as _types

def _register_pybind_submodules(parent_full_name, parent_module):
    for name in dir(parent_module):
        if name.startswith("_"):
            continue
        sub = getattr(parent_module, name, None)
        if isinstance(sub, _types.ModuleType):
            full = f"{parent_full_name}.{name}"
            sys.modules[full] = sub
            _register_pybind_submodules(full, sub)

_register_pybind_submodules("kt_kernel_ext", _kt_kernel_ext)
_register_pybind_submodules("kt_kernel.kt_kernel_ext", _kt_kernel_ext)
del _register_pybind_submodules, _types

# Also expose kt_kernel_ext as an attribute for backward compatibility
kt_kernel_ext = _kt_kernel_ext

# Import main API
from .experts import KTMoEWrapper
from .experts_base import generate_gpu_experts_masks

def __getattr__(name):
    if name == "AMXSFTMoEWrapper":
        try:
            from .sft.amx import AMXSFTMoEWrapper
            return AMXSFTMoEWrapper
        except (ImportError, AttributeError):
            return None
    raise AttributeError(f"module 'kt_kernel' has no attribute {name!r}")

# Read version from package metadata (preferred) or fallback to project root
try:
    # Try to get version from installed package metadata (works in installed environment)
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("kt-kernel")
    except PackageNotFoundError:
        # Package not installed, try to read from source tree version.py
        import os

        _root_version_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "version.py")
        if os.path.exists(_root_version_file):
            _version_ns = {}
            with open(_root_version_file, "r", encoding="utf-8") as f:
                exec(f.read(), _version_ns)
            __version__ = _version_ns.get("__version__", "0.5.3")
        else:
            __version__ = "0.5.3"
except ImportError:
    # Python < 3.8, fallback to pkg_resources or hardcoded version
    try:
        from pkg_resources import get_distribution, DistributionNotFound

        try:
            __version__ = get_distribution("kt-kernel").version
        except DistributionNotFound:
            __version__ = "0.5.3"
    except ImportError:
        __version__ = "0.5.3"

__all__ = ["KTMoEWrapper", "AMXSFTMoEWrapper", "generate_gpu_experts_masks", "kt_kernel_ext", "__cpu_variant__", "__version__"]
