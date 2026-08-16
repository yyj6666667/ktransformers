# KT SFT artifact contracts
# SPDX-License-Identifier: Apache-2.0

"""Validated, framework-neutral artifact interfaces for KT SFT.

Transformers owns model construction and Trainer sequencing.  This module owns
the on-disk KT formats and returns immutable load plans which Transformers can
apply without knowing the manifest schema.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from safetensors import safe_open

from .weight_manifest import validate_persistent_int8_weights


KT_NON_EXPERT_MANIFEST_NAME = "kt_non_expert_manifest.json"
KT_NON_EXPERT_INDEX_NAME = "model.safetensors.index.json"
KT_ADAPTER_MANIFEST_NAME = "kt_adapter_manifest.json"
FUSED_EXPERT_LORA_NAME = "fused_expert_lora.safetensors"
KT_NON_EXPERT_MANIFEST_VERSION = 2
KT_ADAPTER_MANIFEST_VERSION = 1

_LEGACY_NON_EXPERT_VERSION = 1
_LEGACY_NON_EXPERT_PRODUCER = "llamafactory.prepare-kt-cache"
_NON_EXPERT_PRODUCER = "kt-kernel.prepare-non-expert-cache"
_STANDARD_ADAPTER_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
_FUSED_LORA_NAMES = (
    "gate_lora_a",
    "gate_lora_b",
    "up_lora_a",
    "up_lora_b",
    "down_lora_a",
    "down_lora_b",
)
_FP32_ROUTER_BIAS = re.compile(r"^model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$")
_ROUTED_EXPERT = re.compile(r"(?:^|\.)experts(?:\.|$)")
_ROUTED_EXPERT_PARAMETER = re.compile(
    r"\.experts\.(?:\d+\.|gate_up_proj(?:\.|$)|down_proj(?:\.|$)|gate_proj(?:\.|$)|up_proj(?:\.|$))"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PARAMETER_MARKER = "_is_kt_int8_routed_expert_base_parameter"
_RUNTIME_MODULE_MARKER = "_is_kt_routed_expert_runtime_module"
_RUNTIME_MODULE_PATHS = "_kt_routed_expert_runtime_module_paths"
_RUNTIME_MODULE_REFS = "_kt_routed_expert_runtime_module_refs"
_RUNTIME_TENSOR_CONTRACTS = "_kt_routed_expert_runtime_tensor_contracts"
_SUPPORTED_MOE_ARCHITECTURES = (
    "DeepseekV2",
    "DeepseekV3",
    "Qwen2Moe",
    "Qwen3Moe",
    "Qwen3_5Moe",
    "Glm4Moe",
    "Mixtral",
)
_EXPERT_WEIGHT_FORMATS = frozenset({"bf16", "int8", "fp8"})


class KTArtifactError(RuntimeError):
    """A KT artifact is incomplete, unsafe, or incompatible with the runtime."""


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def is_kt_routed_expert_parameter_name(name: str) -> bool:
    """Whether a checkpoint key is a routed expert base parameter owned by KT."""

    return isinstance(name, str) and _ROUTED_EXPERT_PARAMETER.search(name) is not None


def is_kt_supported_moe_model(model: Any) -> bool:
    """Return whether the model architecture is implemented by KT SFT wrappers."""

    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None)
    if not isinstance(architectures, (list, tuple)):
        return False
    return any(
        isinstance(architecture, str)
        and any(marker in architecture for marker in _SUPPORTED_MOE_ARCHITECTURES)
        for architecture in architectures
    )


def _deepseek_routed_paths(config: Any) -> tuple[str, ...]:
    layer_count = getattr(config, "num_hidden_layers", None)
    first_moe_layer = getattr(config, "first_k_dense_replace", None)
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or isinstance(first_moe_layer, bool)
        or not isinstance(first_moe_layer, int)
        or first_moe_layer < 0
        or first_moe_layer >= layer_count
    ):
        raise KTArtifactError("model config has an invalid routed-expert layer range")
    return tuple(f"model.layers.{index}.mlp.experts" for index in range(first_moe_layer, layer_count))


def _effective_runtime_shape(tensor: Any, name: str) -> tuple[int, ...]:
    shape = getattr(tensor, "_kt_original_shape", None) if getattr(tensor, "_kt_zero_storage", False) else tensor.shape
    try:
        resolved = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise KTArtifactError(f"{name} has invalid routed-expert shape metadata") from exc
    if not resolved or any(value <= 0 for value in resolved):
        raise KTArtifactError(f"{name} has invalid routed-expert shape {resolved}")
    return resolved


def _runtime_tensor_contract(path: str, module: Any) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    entries = []
    seen = set()
    for kind, tensors in (
        ("parameter", module.named_parameters(recurse=True, remove_duplicate=False)),
        ("buffer", module.named_buffers(recurse=True, remove_duplicate=False)),
    ):
        for name, tensor in tensors:
            key = (kind, name)
            if key in seen:
                raise KTArtifactError(f"{path} exposes duplicate routed-expert {kind} {name!r}")
            seen.add(key)
            entries.append((kind, name, _effective_runtime_shape(tensor, f"{path}.{name}")))
    if not any(kind == "parameter" for kind, _, _ in entries):
        raise KTArtifactError(f"routed-expert subtree {path!r} contains no parameters")
    return tuple(sorted(entries))


def _validate_runtime_expert_structure(path: str, experts: Any, moe_config: Any, hidden_size: Any) -> None:
    import torch

    dimensions = (moe_config.expert_num, moe_config.intermediate_size, hidden_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise KTArtifactError(f"{path} has invalid routed-expert dimensions {dimensions}")
    expert_num, intermediate_size, hidden_size = dimensions

    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if isinstance(gate_up, torch.nn.Parameter) or isinstance(down, torch.nn.Parameter):
        if not isinstance(gate_up, torch.nn.Parameter) or not isinstance(down, torch.nn.Parameter):
            raise KTArtifactError(f"{path} must register both fused gate_up_proj and down_proj parameters")
        expected = {
            "gate_up_proj": (expert_num, 2 * intermediate_size, hidden_size),
            "down_proj": (expert_num, hidden_size, intermediate_size),
        }
        for name, parameter in (("gate_up_proj", gate_up), ("down_proj", down)):
            actual = _effective_runtime_shape(parameter, f"{path}.{name}")
            if actual != expected[name]:
                raise KTArtifactError(f"{path}.{name} shape mismatch: expected={expected[name]}, actual={actual}")
        return

    children = tuple(experts.named_children())
    expected_names = tuple(str(index) for index in range(expert_num))
    if tuple(name for name, _ in children) != expected_names:
        raise KTArtifactError(
            f"{path} expert inventory mismatch: expected={list(expected_names)}, "
            f"actual={[name for name, _ in children]}"
        )
    gate_name, up_name, down_name = moe_config.weight_names
    expected_shapes = {
        gate_name: (intermediate_size, hidden_size),
        up_name: (intermediate_size, hidden_size),
        down_name: (hidden_size, intermediate_size),
    }
    for expert_name, expert in children:
        for projection_name, expected_shape in expected_shapes.items():
            projection = getattr(expert, projection_name, None)
            weight = getattr(projection, "weight", None)
            if not isinstance(weight, torch.nn.Parameter):
                raise KTArtifactError(f"{path}.{expert_name}.{projection_name} does not expose a weight Parameter")
            actual_shape = _effective_runtime_shape(weight, f"{path}.{expert_name}.{projection_name}.weight")
            if actual_shape != expected_shape:
                raise KTArtifactError(
                    f"{path}.{expert_name}.{projection_name}.weight shape mismatch: "
                    f"expected={expected_shape}, actual={actual_shape}"
                )


def _enumerate_runtime_routed_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    if not is_kt_supported_moe_model(model):
        return ()
    try:
        from .arch import _get_layers_prefix, _get_model_container_and_layers, get_moe_arch_config, get_moe_module

        config = model.config
        moe_config = get_moe_arch_config(config)
        _, layers = _get_model_container_and_layers(model, purpose="routed-expert ownership")
        layers_path = _get_layers_prefix(config)
        registered_layers = model.get_submodule(layers_path)
    except Exception as exc:
        raise KTArtifactError(f"could not enumerate KT routed-expert layers: {exc}") from exc
    if registered_layers is not layers:
        raise KTArtifactError(f"model layer path {layers_path!r} does not resolve to the discovered layer container")

    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", None)
    modules = []
    identities = set()
    for layer_index, layer in enumerate(layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue
        experts = getattr(moe_module, moe_config.experts_attr, None)
        if experts is None or not hasattr(experts, "named_parameters"):
            raise KTArtifactError(f"layer {layer_index} does not expose a registered routed-expert module")
        path = f"{layers_path}.{layer_index}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}"
        try:
            registered = model.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise KTArtifactError(f"missing routed-expert subtree {path!r}") from exc
        if registered is not experts:
            raise KTArtifactError(f"routed-expert subtree {path!r} does not preserve module identity")
        if id(experts) in identities:
            raise KTArtifactError(f"routed-expert subtree {path!r} shares a module with another layer")
        identities.add(id(experts))
        _validate_runtime_expert_structure(path, experts, moe_config, hidden_size)
        modules.append((path, experts))
    if not modules:
        raise KTArtifactError("supported KT MoE model contains no routed-expert layers")
    return tuple(modules)


def _validated_runtime_routed_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    metadata = (
        getattr(model, _RUNTIME_MODULE_PATHS, None),
        getattr(model, _RUNTIME_MODULE_REFS, None),
        getattr(model, _RUNTIME_TENSOR_CONTRACTS, None),
    )
    if metadata == (None, None, None):
        return ()
    if any(value is None for value in metadata):
        raise KTArtifactError("routed-expert runtime ownership metadata is incomplete")
    paths, module_refs, contracts = metadata
    if not isinstance(paths, tuple) or not isinstance(module_refs, tuple) or not isinstance(contracts, tuple):
        raise KTArtifactError("routed-expert runtime ownership metadata has invalid types")

    enumerated = _enumerate_runtime_routed_modules(model)
    expected_paths = tuple(path for path, _ in enumerated)
    if paths != expected_paths:
        raise KTArtifactError("routed-expert runtime ownership paths changed")
    if len(module_refs) != len(paths) or len(contracts) != len(paths):
        raise KTArtifactError("routed-expert runtime ownership metadata has inconsistent lengths")

    validated = []
    for index, ((path, current), claimed, contract) in enumerate(zip(enumerated, module_refs, contracts)):
        if current is not claimed:
            raise KTArtifactError(f"routed-expert subtree {path!r} changed module identity")
        if getattr(current, _RUNTIME_MODULE_MARKER, False) is not True:
            raise KTArtifactError(f"routed-expert subtree {path!r} lost its runtime ownership marker")
        if _runtime_tensor_contract(path, current) != contract:
            raise KTArtifactError(f"routed-expert subtree {path!r} changed its tensor contract")
        validated.append((paths[index], current))
    return tuple(validated)


def claim_kt_routed_expert_subtrees(model: Any) -> tuple[str, ...]:
    """Claim routed-expert subtrees that KT will own outside the framework state dict."""

    existing = (
        getattr(model, _RUNTIME_MODULE_PATHS, None),
        getattr(model, _RUNTIME_MODULE_REFS, None),
        getattr(model, _RUNTIME_TENSOR_CONTRACTS, None),
    )
    if existing != (None, None, None):
        return tuple(path for path, _ in _validated_runtime_routed_modules(model))

    modules = _enumerate_runtime_routed_modules(model)
    if not modules:
        return ()
    for path, module in modules:
        if getattr(module, _RUNTIME_MODULE_MARKER, False):
            raise KTArtifactError(f"routed-expert subtree {path!r} has an unowned runtime marker")

    paths = tuple(path for path, _ in modules)
    refs = tuple(module for _, module in modules)
    contracts = tuple(_runtime_tensor_contract(path, module) for path, module in modules)
    try:
        for _, module in modules:
            setattr(module, _RUNTIME_MODULE_MARKER, True)
        setattr(model, _RUNTIME_MODULE_PATHS, paths)
        setattr(model, _RUNTIME_MODULE_REFS, refs)
        setattr(model, _RUNTIME_TENSOR_CONTRACTS, contracts)
    except BaseException:
        for _, module in modules:
            with contextlib.suppress(AttributeError):
                delattr(module, _RUNTIME_MODULE_MARKER)
        for name in (_RUNTIME_MODULE_PATHS, _RUNTIME_MODULE_REFS, _RUNTIME_TENSOR_CONTRACTS):
            with contextlib.suppress(AttributeError):
                delattr(model, name)
        raise
    return paths


def mark_kt_int8_routed_expert_base_parameters(
    model: Any, plan: KTPretrainedLoadPlan | None
) -> tuple[str, ...]:
    """Mark native routed-expert tensors omitted from a validated load plan."""

    if plan is None:
        return ()
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "deepseek_v3":
        return ()
    paths = _deepseek_routed_paths(config)
    expected_shapes = {
        "gate_up_proj": (
            getattr(config, "n_routed_experts", None),
            2 * getattr(config, "moe_intermediate_size", 0),
            getattr(config, "hidden_size", None),
        ),
        "down_proj": (
            getattr(config, "n_routed_experts", None),
            getattr(config, "hidden_size", None),
            getattr(config, "moe_intermediate_size", None),
        ),
    }
    if any(any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape) for shape in expected_shapes.values()):
        raise KTArtifactError("model config has invalid routed-expert dimensions")
    modules = []
    parameters = []
    for path in paths:
        try:
            experts = model.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise KTArtifactError(f"missing routed-expert subtree {path!r}") from exc
        actual = dict(experts.named_parameters(recurse=True))
        if set(actual) != set(expected_shapes):
            raise KTArtifactError(
                f"{path} parameter contract mismatch: expected={sorted(expected_shapes)}, actual={sorted(actual)}"
            )
        for name, shape in expected_shapes.items():
            if tuple(actual[name].shape) != shape:
                raise KTArtifactError(
                    f"{path}.{name} shape mismatch: expected={shape}, actual={tuple(actual[name].shape)}"
                )
            parameters.append(actual[name])
        modules.append(experts)
    runtime_paths = tuple(path for path, _ in _enumerate_runtime_routed_modules(model))
    if runtime_paths != paths:
        raise KTArtifactError(
            "validated INT8 routed-expert paths differ from the runtime ownership contract: "
            f"artifact={paths}, runtime={runtime_paths}"
        )
    if claim_kt_routed_expert_subtrees(model) != paths:
        raise KTArtifactError("INT8 routed-expert ownership claim returned inconsistent paths")
    for parameter in parameters:
        setattr(parameter, _PARAMETER_MARKER, True)
    return paths


def is_kt_int8_routed_expert_base_parameter(parameter: Any) -> bool:
    return getattr(parameter, _PARAMETER_MARKER, False) is True


@contextlib.contextmanager
def project_kt_routed_experts_out_of_device_map(model: Any) -> Iterator[None]:
    """Temporarily project KT-owned routed experts to zero-sized meta tensors."""

    modules = _validated_runtime_routed_modules(model)
    if not modules:
        yield
        return
    import torch

    replacements = []
    try:
        for path, experts in modules:
            for module in experts.modules():
                for name, parameter in tuple(module._parameters.items()):
                    if parameter is None:
                        continue
                    if parameter.device.type != "meta":
                        raise KTArtifactError(f"{path}.{name} must be meta before device-map inference")
                    projected = torch.nn.Parameter(
                        torch.empty(0, dtype=parameter.dtype, device="meta"),
                        requires_grad=parameter.requires_grad,
                    )
                    module._parameters[name] = projected
                    replacements.append((module._parameters, name, parameter))
                for name, buffer in tuple(module._buffers.items()):
                    if buffer is None:
                        continue
                    if buffer.device.type != "meta":
                        raise KTArtifactError(f"{path}.{name} must be meta before device-map inference")
                    module._buffers[name] = torch.empty(0, dtype=buffer.dtype, device="meta")
                    replacements.append((module._buffers, name, buffer))
        yield
    finally:
        for registry, name, original in reversed(replacements):
            registry[name] = original


def prepare_kt_non_expert_device_map(model: Any, device_map: Any) -> Any:
    """Remove virtual expert placements and reject host offload of real tensors."""

    modules = _validated_runtime_routed_modules(model)
    if not modules:
        return device_map
    if not isinstance(device_map, dict):
        raise KTArtifactError("non-expert placement requires a resolved device-map dictionary")
    import torch

    paths = tuple(path for path, _ in modules)
    resolved = {
        name: device
        for name, device in device_map.items()
        if not any(name == path or name.startswith(f"{path}.") for path in paths)
    }
    if not resolved:
        raise KTArtifactError("device map became empty after removing routed experts")

    def is_host(device: Any) -> bool:
        if device == "disk":
            return True
        if isinstance(device, int):
            return False
        try:
            return torch.device(device).type in {"cpu", "meta"}
        except (RuntimeError, TypeError):
            return False

    host_entries = {name: device for name, device in resolved.items() if is_host(device)}
    if host_entries:
        raise KTArtifactError(
            "non-expert device map offloaded real tensors to CPU/disk: "
            + ", ".join(f"{name or '<root>'}={device}" for name, device in sorted(host_entries.items()))
        )
    return resolved


@contextlib.contextmanager
def hide_kt_routed_experts_from_dispatch(model: Any) -> Iterator[None]:
    """Temporarily unregister KT-owned subtrees from parent dispatch hooks."""

    modules = _validated_runtime_routed_modules(model)
    if not modules:
        yield
        return
    import torch

    replacements = []
    try:
        for path, experts in modules:
            parent_path, child_name = path.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            if parent._modules.get(child_name) is not experts:
                raise KTArtifactError(f"routed-expert subtree {path!r} changed before dispatch")
            parent._modules[child_name] = torch.nn.Module()
            replacements.append((parent, child_name, experts))
        yield
    finally:
        for parent, child_name, experts in reversed(replacements):
            parent._modules[child_name] = experts



__all__ = [
    "KTArtifactError",
    "claim_kt_routed_expert_subtrees",
    "hide_kt_routed_experts_from_dispatch",
    "is_kt_int8_routed_expert_base_parameter",
    "is_kt_routed_expert_parameter_name",
    "is_kt_supported_moe_model",
    "mark_kt_int8_routed_expert_base_parameters",
    "prepare_kt_non_expert_device_map",
    "project_kt_routed_experts_out_of_device_map",
]
