"""Reversible Trainer instrumentation for the KT step timeline harness.

This module deliberately sits outside LLaMA-Factory and Transformers.  It
wraps their public runtime objects only for the duration of one process and
restores every patched method from :meth:`InstrumentationManager.close`.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import importlib.util
import json
import os
import platform
import socket
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Any, Callable

from .recorder import TimelineRecorder
from .resctrl import HardwareCounterSampler
from .samplers import MemoryReader, MemorySampler, PerfStatReader, monotonic_raw_ns


class ProfileMode(str, enum.Enum):
    OFF = "off"
    PHASE = "phase"
    DEEP = "deep"

    @classmethod
    def parse(cls, value: str | "ProfileMode") -> "ProfileMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(f"KT_STEP_PROFILE_MODE must be one of: {choices}") from exc


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON record with one OS write.

    MemorySampler uses append mode as well.  Rank-isolated paths and O_APPEND
    keep records intact when its helper threads and phase-boundary hooks write
    concurrently.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _restore_instance_attribute(
    instance: Any, name: str, existed: bool, value: Any
) -> None:
    if existed:
        setattr(instance, name, value)
    else:
        try:
            delattr(instance, name)
        except AttributeError:
            pass


def _git_head(start: Path) -> str | None:
    """Resolve a Git HEAD without starting a subprocess."""

    for parent in (start, *start.parents):
        marker = parent / ".git"
        if marker.is_dir():
            git_dir = marker
        elif marker.is_file():
            try:
                prefix, raw_path = (
                    marker.read_text(encoding="utf-8").strip().split(":", 1)
                )
                if prefix != "gitdir":
                    continue
                git_dir = (parent / raw_path.strip()).resolve()
            except (OSError, ValueError):
                continue
        else:
            continue
        try:
            head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        except OSError:
            return None
        if not head.startswith("ref: "):
            return head or None
        ref = head.removeprefix("ref: ")
        candidates = [git_dir / ref]
        try:
            common = (
                git_dir / (git_dir / "commondir").read_text(encoding="ascii").strip()
            ).resolve()
            candidates.append(common / ref)
        except OSError:
            common = git_dir
        for candidate in candidates:
            try:
                return candidate.read_text(encoding="ascii").strip()
            except OSError:
                pass
        for packed in (git_dir / "packed-refs", common / "packed-refs"):
            try:
                lines = packed.read_text(encoding="ascii").splitlines()
            except OSError:
                continue
            for line in lines:
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref:
                        return commit
        return None
    return None


def _module_revision(module_name: str) -> dict[str, str] | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin:
        return None
    path = Path(origin).resolve()
    head = _git_head(path.parent)
    return {"path": str(path), "commit": head or "unknown"}


def _safe_token_count(inputs: Any) -> int | None:
    """Count local non-padding tokens without synchronizing an accelerator."""

    if not isinstance(inputs, dict):
        return None
    tensor = inputs.get("attention_mask")
    if tensor is not None:
        try:
            if (
                getattr(tensor, "device", None) is not None
                and tensor.device.type != "cpu"
            ):
                return None
            return int(tensor.sum().item())
        except Exception:
            return None
    tensor = inputs.get("input_ids")
    if tensor is None:
        return None
    try:
        if getattr(tensor, "device", None) is not None and tensor.device.type != "cpu":
            return None
        return int(tensor.numel())
    except Exception:
        return None


class ProfileRuntime:
    """Own the recorder and process-level memory sampling."""

    GPU_PHASES = {
        "data_prepare",
        "forward",
        "backward",
        "grad_clip",
        "optimizer",
        "kt_post_update",
        "scheduler",
        "zero_grad",
        "train_core",
    }

    def __init__(self, mode: ProfileMode) -> None:
        self.mode = mode
        self.rank = _rank()
        self.world_size = _world_size()
        configured_run_id = os.environ.get("KT_STEP_PROFILE_RUN_ID")
        if self.world_size > 1 and not configured_run_id:
            raise ValueError(
                "KT_STEP_PROFILE_RUN_ID is required for multi-rank profiling"
            )
        self.run_id = (
            configured_run_id
            or time.strftime("%Y%m%dT%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
        )
        root = Path(os.environ.get("KT_STEP_PROFILE_DIR", "kt_step_profile"))
        self.root = root
        self.out_dir = root / f"rank_{self.rank}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.out_dir / "memory_samples.jsonl"
        self.kt_profile_path = self.out_dir / "kt_step_profile.jsonl"
        self.warmup_steps = _env_int("KT_STEP_PROFILE_WARMUP_STEPS", 0)
        self.cuda_timing = _env_bool("KT_STEP_PROFILE_CUDA_EVENTS", True)
        self.boundary_memory = _env_bool("KT_STEP_PROFILE_BOUNDARY_MEMORY", True)
        self.recorder = TimelineRecorder(
            out_dir=self.root,
            run_id=self.run_id,
            rank=self.rank,
            world_size=self.world_size,
            mode=mode.value,
            warmup_steps=self.warmup_steps,
            enable_cuda_events=self.cuda_timing,
        )
        self.memory_reader = MemoryReader(rank=self.rank)
        self.memory_sampler = MemorySampler(
            out_path=self.memory_path,
            rank=self.rank,
            run_id=self.run_id,
            host_interval_ms=_env_int("KT_STEP_PROFILE_HOST_INTERVAL_MS", 20),
            nvml_interval_ms=_env_int("KT_STEP_PROFILE_NVML_INTERVAL_MS", 50),
            reader=self.memory_reader,
        )
        self.hardware_path = self.out_dir / "hardware_counters.jsonl"
        resctrl_group = os.environ.get("KT_STEP_PROFILE_RESCTRL_GROUP")
        self.hardware_sampler = (
            HardwareCounterSampler(
                out_path=self.hardware_path,
                rank=self.rank,
                run_id=self.run_id,
                resctrl_group=resctrl_group,
                interval_ms=_env_int("KT_STEP_PROFILE_HARDWARE_INTERVAL_MS", 100),
            )
            if resctrl_group
            else None
        )
        self._closed = False
        self._started = False
        self._boundary_counter = 0
        self._write_run_meta()

    def _write_run_meta(self) -> None:
        row: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "mode": self.mode.value,
            "clock": "CLOCK_MONOTONIC_RAW",
            "started_monotonic_raw_ns": monotonic_raw_ns(),
            "started_unix_ns": time.time_ns(),
            "python": sys.version,
            "platform": platform.platform(),
            "host_interval_ms": _env_int("KT_STEP_PROFILE_HOST_INTERVAL_MS", 20),
            "nvml_interval_ms": _env_int("KT_STEP_PROFILE_NVML_INTERVAL_MS", 50),
            "cuda_events": self.cuda_timing,
            "boundary_memory": self.boundary_memory,
            "measurement_contract": {
                "phase_wall": "CLOCK_MONOTONIC_RAW host interval; nested scopes remain separate",
                "cuda_stream": "CUDA event interval resolved after work submission; never added to host wall",
                "rss_cgroup_nvml": "resident-state samples correlated by timestamp, not allocation ownership",
                "torch_allocator": "PyTorch CUDA allocator state; native KT pools are reported separately",
            },
        }
        for name in (
            "KT_SOURCE_COMMIT",
            "LLAMA_FACTORY_SOURCE_COMMIT",
            "TRANSFORMERS_SOURCE_COMMIT",
            "ACCELERATE_SOURCE_COMMIT",
        ):
            if os.environ.get(name):
                row[name.lower()] = os.environ[name]
        self._run_meta = row
        self._flush_run_meta()

    def _flush_run_meta(self) -> None:
        path = self.out_dir / "run_meta.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._run_meta, indent=2, ensure_ascii=False) + "\n"
        )
        os.replace(temporary, path)

    def refresh_source_revisions(self) -> None:
        modules = {
            "ktransformers": "kt_kernel",
            "llama_factory": "llamafactory",
            "transformers": "transformers",
            "accelerate": "accelerate",
        }
        self._run_meta["sources"] = {
            label: revision
            for label, module in modules.items()
            if (revision := _module_revision(module)) is not None
        }
        self._flush_run_meta()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.memory_sampler.start()
        _append_jsonl(
            self.kt_profile_path,
            {
                "schema_version": 1,
                "record_type": "kt_step_profile_metadata",
                "clock": "CLOCK_MONOTONIC_RAW",
                "timestamp_ns": monotonic_raw_ns(),
                "run_id": self.run_id,
                "rank": self.rank,
                "status": "enabled",
            },
        )
        if self.hardware_sampler is not None:
            self.hardware_sampler.start()
        else:
            _append_jsonl(
                self.hardware_path,
                {
                    "schema_version": 1,
                    "record_type": "resctrl_mbm_sample",
                    "clock_name": "CLOCK_MONOTONIC_RAW",
                    "monotonic_raw_ns": monotonic_raw_ns(),
                    "run_id": self.run_id,
                    "rank": self.rank,
                    "status": "unavailable",
                    "errors": {
                        "resctrl": "KT_STEP_PROFILE_RESCTRL_GROUP is not configured"
                    },
                },
            )

    def clock_sync_marker(self) -> None:
        """Place a raw-clock anchor in both the ledger and deep traces."""

        raw_ns = monotonic_raw_ns()
        self.recorder.emit("clock_sync", raw_ns=raw_ns, clock="CLOCK_MONOTONIC_RAW")
        if self.mode is not ProfileMode.DEEP:
            return
        marker = f"kt.clock_sync.raw_ns={raw_ns}"
        try:
            import torch

            with torch.profiler.record_function(marker):
                pass
            if torch.cuda.is_available():
                torch.cuda.nvtx.mark(marker)
        except Exception as exc:
            self.recorder.emit(
                "instrumentation_warning", phase="clock_sync", error=repr(exc)
            )

    def _memory_boundary(
        self, name: str, boundary: str, recorder: TimelineRecorder, boundary_id: str
    ) -> None:
        if not self.boundary_memory:
            return
        snapshot = self.memory_reader.snapshot(
            include_process=True,
            include_cgroup=True,
            include_torch_cuda=True,
            include_nvml=False,
        )
        snapshot.update(
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "sample_kind": "phase_boundary",
                "phase": name,
                "boundary": boundary,
                "boundary_id": boundary_id,
                "step": recorder.current_step,
                "microbatch": recorder.current_microbatch,
            }
        )
        self.memory_sampler.emit(snapshot)

    @contextlib.contextmanager
    def scope(self, name: str):
        self._boundary_counter += 1
        boundary_id = f"{self.rank}:{os.getpid()}:{self._boundary_counter}"
        self._memory_boundary(name, "begin", self.recorder, boundary_id)
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    self.recorder.scope(
                        name,
                        cuda_timing=self.cuda_timing and name in self.GPU_PHASES,
                        metadata={
                            "boundary_id": boundary_id,
                            "overview": name == "train_core",
                        },
                    )
                )
                if self.mode is ProfileMode.DEEP:
                    try:
                        import torch

                        stack.enter_context(
                            torch.profiler.record_function(f"kt.step.{name}")
                        )
                        if torch.cuda.is_available():
                            torch.cuda.nvtx.range_push(f"kt.step.{name}")
                            stack.callback(torch.cuda.nvtx.range_pop)
                    except Exception as exc:
                        self.recorder.emit(
                            "instrumentation_warning", phase=name, error=repr(exc)
                        )
                yield
        finally:
            self._memory_boundary(name, "end", self.recorder, boundary_id)

    def collect_kt_profile(self, model: Any, step: int) -> None:
        try:
            from kt_kernel.sft.profiler import collect_kt_sft_profile

            profile = collect_kt_sft_profile(model, reset=True)
            _append_jsonl(
                self.kt_profile_path,
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "rank": self.rank,
                    "pid": os.getpid(),
                    "timestamp_ns": monotonic_raw_ns(),
                    "clock": "CLOCK_MONOTONIC_RAW",
                    "step": step,
                    "profile": profile,
                },
            )
        except Exception as exc:
            self.recorder.emit("kt_profile_error", step=step, error=repr(exc))

    def close(self, *, partial: bool) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self.memory_sampler.stop()
        self.memory_reader.close()
        if self.hardware_sampler is not None:
            self.hardware_sampler.stop()
        perf_path = os.environ.get("KT_STEP_PROFILE_PERF_STAT_FILE")
        if perf_path:
            perf = PerfStatReader(perf_path).snapshot()
            perf.update({"run_id": self.run_id, "rank": self.rank})
            _append_jsonl(self.hardware_path, perf)
        allocation_path = self.out_dir / "allocation_events.jsonl"
        if not allocation_path.exists():
            _append_jsonl(
                allocation_path,
                {
                    "schema_version": 1,
                    "record_type": "torch_allocation_metadata",
                    "clock_name": "CLOCK_MONOTONIC_RAW",
                    "status": "pending_postprocess"
                    if self.mode is ProfileMode.DEEP
                    else "not_collected",
                    "reason": (
                        "run trace_allocations.py against the Torch deep trace"
                        if self.mode is ProfileMode.DEEP
                        else "allocation ownership requires deep mode"
                    ),
                },
            )
        self.recorder.close(partial=partial)
        try:
            from .kt_events import read_events, summarize_pool_events

            pool_path = self.out_dir / "kt_pool_events.jsonl"
            pool_summary = (
                summarize_pool_events(read_events(pool_path))
                if pool_path.exists()
                else {
                    "schema": "kt.sft.pool_event.v1",
                    "status": "unavailable",
                    "reason": "no KT pool event file",
                }
            )
            (self.out_dir / "kt_pool_summary.json").write_text(
                json.dumps(pool_summary, indent=2, ensure_ascii=False) + "\n"
            )
        except Exception as exc:
            (self.out_dir / "kt_pool_summary.json").write_text(
                json.dumps(
                    {
                        "schema": "kt.sft.pool_event.v1",
                        "status": "partial",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
                + "\n"
            )
        try:
            from .memory_report import write_reports

            write_reports(self.out_dir)
        except Exception as exc:
            (self.out_dir / "postprocess_error.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "partial",
                        "stage": "memory_report",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
                + "\n"
            )


class TrainerBinding:
    def __init__(
        self, manager: "InstrumentationManager", trainer: Any, trainer_id: int
    ) -> None:
        self.manager = manager
        self.runtime = manager.runtime
        self.trainer_ref = weakref.ref(trainer)
        self.trainer_id = trainer_id
        self.model: Any = None
        self.step_open = False
        self.step = -1
        self.microbatch = 0
        self.local_tokens = 0
        self.local_token_count_complete = True
        self.token_start: int | float | None = None
        self.train_core_context: Any = None
        self.callback: Any = None
        self.instance_restores: list[Callable[[], None]] = []

    @property
    def recorder(self) -> TimelineRecorder:
        return self.runtime.recorder

    def begin_step(self, trainer: Any) -> None:
        active = self.manager.active_binding
        if active is not None and active is not self and active.step_open:
            raise RuntimeError(
                "concurrent or nested Trainer profiling is not supported"
            )
        if self.step_open:
            self.finish_step(trainer, partial=True)
        self.step = (
            int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0) + 1
        )
        self.microbatch = 0
        self.local_tokens = 0
        self.local_token_count_complete = True
        self.token_start = getattr(
            getattr(trainer, "state", None), "num_input_tokens_seen", None
        )
        self.recorder.begin_step(self.step, trainer_id=self.trainer_id)
        self.runtime.clock_sync_marker()
        self.step_open = True
        self.manager.active_binding = self

    def mark_microbatch(self, tokens: int | None) -> int:
        self.microbatch += 1
        if tokens is None:
            self.local_token_count_complete = False
        else:
            self.local_tokens += int(tokens)
        self.recorder.mark_microbatch(
            self.microbatch - 1,
            tokens=tokens,
            token_scope="rank_local",
        )
        return self.microbatch

    def ensure_train_core(self) -> None:
        if self.train_core_context is None:
            self.train_core_context = self.runtime.scope("train_core")
            self.train_core_context.__enter__()

    def close_train_core(self) -> None:
        if self.train_core_context is None:
            return
        context = self.train_core_context
        self.train_core_context = None
        context.__exit__(None, None, None)

    def finish_step(self, trainer: Any, *, partial: bool = False) -> None:
        if not self.step_open:
            return
        self.close_train_core()
        token_end = getattr(
            getattr(trainer, "state", None), "num_input_tokens_seen", None
        )
        observed_tokens = None
        token_scope = "unknown"
        tracking_mode = getattr(
            getattr(trainer, "args", None), "include_num_input_tokens_seen", None
        )
        if (
            tracking_mode == "non_padding"
            and isinstance(self.token_start, (int, float))
            and isinstance(token_end, (int, float))
            and token_end >= self.token_start
        ):
            observed_tokens = int(token_end - self.token_start)
            token_scope = "world_aggregated"
        elif self.local_token_count_complete and self.microbatch > 0:
            observed_tokens = self.local_tokens
            token_scope = "rank_local"
        self.recorder.emit(
            "step_tokens",
            step=self.step,
            observed_non_padding_tokens=observed_tokens,
            source=(
                "TrainerState.num_input_tokens_seen"
                if token_scope == "world_aggregated"
                else "rank-local attention_mask"
                if token_scope == "rank_local"
                else "unavailable"
            ),
            token_scope=token_scope,
            microbatches=self.microbatch,
            partial=partial,
        )
        self.runtime.collect_kt_profile(self.model, self.step)
        self.recorder.end_step(
            self.step,
            partial=partial,
            observed_tokens=observed_tokens,
            token_scope=token_scope,
        )
        self.step_open = False
        if self.manager.active_binding is self:
            self.manager.active_binding = None

    def wrap_runtime_objects(self, trainer: Any) -> None:
        self.model = getattr(trainer, "model", None)
        self._wrap_optimizer(getattr(trainer, "optimizer", None))
        self._wrap_model_zero_grad(self.model)
        self.manager.register_scheduler(getattr(trainer, "lr_scheduler", None), self)
        self._wrap_kt_repack(self.model)

    def _wrap_model_zero_grad(self, model: Any) -> None:
        if (
            model is None
            or not hasattr(model, "zero_grad")
            or model in self.manager.wrapped_models
        ):
            return
        self.manager.wrapped_models.add(model)
        namespace = getattr(model, "__dict__", {})
        existed = "zero_grad" in namespace
        instance_value = namespace.get("zero_grad")
        original = model.zero_grad

        @functools.wraps(original)
        def profiled_model_zero_grad(*args, **kwargs):
            if not self.step_open:
                return original(*args, **kwargs)
            try:
                with self.runtime.scope("zero_grad"):
                    return original(*args, **kwargs)
            finally:
                self.close_train_core()
                self.runtime.clock_sync_marker()

        model.zero_grad = profiled_model_zero_grad
        self.instance_restores.append(
            lambda: _restore_instance_attribute(
                model, "zero_grad", existed, instance_value
            )
        )

    def _wrap_kt_repack(self, model: Any) -> None:
        if model is None:
            return
        try:
            from kt_kernel.sft.lora import _find_kt_wrappers

            wrappers = _find_kt_wrappers(model) or []
        except Exception:
            return
        for layer in wrappers:
            inner = getattr(layer, "wrapper", None)
            if inner is None or not hasattr(inner, "update_base_weights"):
                continue
            original = inner.update_base_weights
            if getattr(original, "_kt_step_profile_wrapped", False):
                continue

            @functools.wraps(original)
            def profiled_repack(*args, __original=original, **kwargs):
                if not self.step_open:
                    return __original(*args, **kwargs)
                with self.runtime.scope("kt_repack"):
                    return __original(*args, **kwargs)

            profiled_repack._kt_step_profile_wrapped = True
            namespace = getattr(inner, "__dict__", {})
            existed = "update_base_weights" in namespace
            instance_value = namespace.get("update_base_weights")
            inner.update_base_weights = profiled_repack
            self.instance_restores.append(
                lambda inner=inner, existed=existed, instance_value=instance_value: (
                    _restore_instance_attribute(
                        inner, "update_base_weights", existed, instance_value
                    )
                )
            )

    def _wrap_optimizer(self, optimizer: Any) -> None:
        if optimizer is None:
            return
        owner = self.manager.optimizer_bindings.get(optimizer)
        if owner is self:
            return
        if owner is not None:
            raise RuntimeError(
                "sharing one optimizer across profiled Trainers is not supported"
            )
        self.manager.optimizer_bindings[optimizer] = self
        namespace = getattr(optimizer, "__dict__", {})
        step_existed = "step" in namespace
        step_instance_value = namespace.get("step")
        zero_existed = "zero_grad" in namespace
        zero_instance_value = namespace.get("zero_grad")
        original_step = optimizer.step

        @functools.wraps(original_step)
        def profiled_step(*args, **kwargs):
            if not self.step_open:
                return original_step(*args, **kwargs)
            with self.runtime.scope("optimizer"):
                return original_step(*args, **kwargs)

        optimizer.step = profiled_step
        original_zero_grad = optimizer.zero_grad

        @functools.wraps(original_zero_grad)
        def profiled_zero_grad(*args, **kwargs):
            if not self.step_open:
                return original_zero_grad(*args, **kwargs)
            try:
                with self.runtime.scope("zero_grad"):
                    return original_zero_grad(*args, **kwargs)
            finally:
                self.close_train_core()
                self.runtime.clock_sync_marker()

        optimizer.zero_grad = profiled_zero_grad

        def restore() -> None:
            _restore_instance_attribute(
                optimizer, "step", step_existed, step_instance_value
            )
            _restore_instance_attribute(
                optimizer, "zero_grad", zero_existed, zero_instance_value
            )

        self.instance_restores.append(restore)

    def restore(self) -> None:
        while self.instance_restores:
            self.instance_restores.pop()()
        trainer = self.trainer_ref()
        if trainer is not None and self.callback is not None:
            if hasattr(trainer, "remove_callback"):
                try:
                    trainer.remove_callback(self.callback)
                except (KeyError, ValueError):
                    pass
            else:
                callbacks = getattr(trainer, "callbacks", None)
                if isinstance(callbacks, list) and self.callback in callbacks:
                    callbacks.remove(self.callback)
        self.callback = None


class InstrumentationManager:
    def __init__(self, mode: ProfileMode) -> None:
        self.mode = mode
        self.runtime = ProfileRuntime(mode) if mode is not ProfileMode.OFF else None
        self.active_binding: TrainerBinding | None = None
        self.bindings: weakref.WeakKeyDictionary[Any, TrainerBinding] = (
            weakref.WeakKeyDictionary()
        )
        self.optimizer_bindings: weakref.WeakKeyDictionary[Any, TrainerBinding] = (
            weakref.WeakKeyDictionary()
        )
        self.wrapped_models: weakref.WeakSet[Any] = weakref.WeakSet()
        self.scheduler_bindings: weakref.WeakKeyDictionary[Any, TrainerBinding] = (
            weakref.WeakKeyDictionary()
        )
        self.class_restores: list[Callable[[], None]] = []
        self._closed = False
        self._trainer_counter = 0

    def install(self) -> "InstrumentationManager":
        if self.mode is ProfileMode.OFF:
            return self
        assert self.runtime is not None
        os.environ.setdefault("KT_SFT_PROFILE", "1")
        os.environ.setdefault("SFT_POOL_LOG", "1")
        # One training process owns each rank directory. Never inherit a shared
        # pool-log filename across ranks.
        os.environ["SFT_POOL_LOG_FILE"] = str(
            self.runtime.out_dir / "kt_pool_events.jsonl"
        )
        import transformers.trainer as trainer_module
        from transformers import Trainer, TrainerCallback

        manager = self

        class TimelineCallback(TrainerCallback):
            def __init__(self, binding: TrainerBinding) -> None:
                self.binding = binding

            def on_train_begin(self, args, state, control, model=None, **kwargs):
                trainer = self.binding.trainer_ref()
                if trainer is not None:
                    self.binding.runtime.start()
                    self.binding.runtime.refresh_source_revisions()
                    self.binding.model = (
                        model if model is not None else getattr(trainer, "model", None)
                    )
                    self.binding.wrap_runtime_objects(trainer)
                    try:
                        from kt_kernel.sft.profiler import reset_kt_sft_profile

                        reset_kt_sft_profile(self.binding.model)
                    except Exception as exc:
                        self.binding.recorder.emit(
                            "kt_profile_reset_error", error=repr(exc)
                        )

            def on_train_end(self, args, state, control, **kwargs):
                trainer = self.binding.trainer_ref()
                if trainer is not None:
                    self.binding.finish_step(trainer, partial=True)

        original_init = Trainer.__init__

        @functools.wraps(original_init)
        def patched_init(trainer, *args, **kwargs):
            original_init(trainer, *args, **kwargs)
            manager._trainer_counter += 1
            binding = TrainerBinding(manager, trainer, manager._trainer_counter)
            manager.bindings[trainer] = binding
            callback = TimelineCallback(binding)
            binding.callback = callback
            trainer.add_callback(callback)

        self._patch(Trainer, "__init__", patched_init)

        original_get_batch = Trainer.get_batch_samples

        @functools.wraps(original_get_batch)
        def profiled_get_batch(trainer, *args, **kwargs):
            binding = manager.bindings.get(trainer)
            if binding is None:
                return original_get_batch(trainer, *args, **kwargs)
            binding.begin_step(trainer)
            with binding.runtime.scope("dataloader"):
                return original_get_batch(trainer, *args, **kwargs)

        self._patch(Trainer, "get_batch_samples", profiled_get_batch)

        original_training_step = Trainer.training_step

        @functools.wraps(original_training_step)
        def profiled_training_step(trainer, model, inputs, *args, **kwargs):
            binding = manager.bindings.get(trainer)
            if binding is None:
                return original_training_step(trainer, model, inputs, *args, **kwargs)
            binding.mark_microbatch(_safe_token_count(inputs))
            binding.ensure_train_core()
            original_prepare = trainer._prepare_inputs
            original_compute_loss = trainer.compute_loss
            original_backward = trainer.accelerator.backward
            trainer_namespace = getattr(trainer, "__dict__", {})
            prepare_existed = "_prepare_inputs" in trainer_namespace
            prepare_instance_value = trainer_namespace.get("_prepare_inputs")
            compute_existed = "compute_loss" in trainer_namespace
            compute_instance_value = trainer_namespace.get("compute_loss")
            accelerator_namespace = getattr(trainer.accelerator, "__dict__", {})
            backward_existed = "backward" in accelerator_namespace
            backward_instance_value = accelerator_namespace.get("backward")

            @functools.wraps(original_prepare)
            def profiled_prepare(*inner_args, **inner_kwargs):
                with binding.runtime.scope("data_prepare"):
                    return original_prepare(*inner_args, **inner_kwargs)

            @functools.wraps(original_compute_loss)
            def profiled_compute_loss(*inner_args, **inner_kwargs):
                with binding.runtime.scope("forward"):
                    return original_compute_loss(*inner_args, **inner_kwargs)

            @functools.wraps(original_backward)
            def profiled_backward(*inner_args, **inner_kwargs):
                with binding.runtime.scope("backward"):
                    return original_backward(*inner_args, **inner_kwargs)

            trainer._prepare_inputs = profiled_prepare
            trainer.compute_loss = profiled_compute_loss
            trainer.accelerator.backward = profiled_backward
            try:
                return original_training_step(trainer, model, inputs, *args, **kwargs)
            finally:
                _restore_instance_attribute(
                    trainer, "_prepare_inputs", prepare_existed, prepare_instance_value
                )
                _restore_instance_attribute(
                    trainer, "compute_loss", compute_existed, compute_instance_value
                )
                _restore_instance_attribute(
                    trainer.accelerator,
                    "backward",
                    backward_existed,
                    backward_instance_value,
                )
                try:
                    if bool(trainer.accelerator.gradient_state.sync_gradients):
                        binding.recorder.mark_microbatch(None)
                except Exception as exc:
                    binding.recorder.emit("microbatch_clear_warning", error=repr(exc))

        self._patch(Trainer, "training_step", profiled_training_step)

        original_clip = Trainer._clip_grad_norm

        @functools.wraps(original_clip)
        def profiled_clip(trainer, *args, **kwargs):
            binding = manager.bindings.get(trainer)
            if binding is None:
                return original_clip(trainer, *args, **kwargs)
            with binding.runtime.scope("grad_clip"):
                return original_clip(trainer, *args, **kwargs)

        self._patch(Trainer, "_clip_grad_norm", profiled_clip)

        if hasattr(Trainer, "_get_grad_norm"):
            original_get_grad_norm = Trainer._get_grad_norm

            @functools.wraps(original_get_grad_norm)
            def profiled_get_grad_norm(trainer, *args, **kwargs):
                binding = manager.bindings.get(trainer)
                if binding is None or not binding.step_open:
                    return original_get_grad_norm(trainer, *args, **kwargs)
                supplied_grad_norm = kwargs.get("grad_norm")
                if "grad_norm" not in kwargs and len(args) > 1:
                    supplied_grad_norm = args[1]
                if supplied_grad_norm is not None:
                    return original_get_grad_norm(trainer, *args, **kwargs)
                with binding.runtime.scope("grad_clip"):
                    return original_get_grad_norm(trainer, *args, **kwargs)

            self._patch(Trainer, "_get_grad_norm", profiled_get_grad_norm)

        original_maybe_log = Trainer._maybe_log_save_evaluate

        @functools.wraps(original_maybe_log)
        def profiled_maybe_log(trainer, *args, **kwargs):
            binding = manager.bindings.get(trainer)
            if binding is None or not binding.step_open:
                return original_maybe_log(trainer, *args, **kwargs)
            binding.close_train_core()
            failed = True
            try:
                with binding.runtime.scope("log_save_eval"):
                    result = original_maybe_log(trainer, *args, **kwargs)
                failed = False
                return result
            finally:
                binding.finish_step(trainer, partial=failed)

        self._patch(Trainer, "_maybe_log_save_evaluate", profiled_maybe_log)

        update_fn = getattr(trainer_module, "update_kt_lora_pointers", None)
        if update_fn is not None:

            @functools.wraps(update_fn)
            def profiled_update(*args, **kwargs):
                binding = manager.active_binding
                if binding is None:
                    return update_fn(*args, **kwargs)
                with binding.runtime.scope("kt_post_update"):
                    return update_fn(*args, **kwargs)

            self._patch(trainer_module, "update_kt_lora_pointers", profiled_update)

        return self

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        original = getattr(owner, name)
        setattr(owner, name, replacement)
        self.class_restores.append(
            lambda owner=owner, name=name, original=original: setattr(
                owner, name, original
            )
        )

    def register_scheduler(self, scheduler: Any, binding: TrainerBinding) -> None:
        if scheduler is None:
            return
        self.scheduler_bindings[scheduler] = binding
        cls = type(scheduler)
        marker = f"_kt_timeline_original_step_{id(self)}"
        if hasattr(cls, marker):
            return
        original = cls.step
        bindings = self.scheduler_bindings

        @functools.wraps(original)
        def profiled_scheduler(instance, *args, **kwargs):
            owner = bindings.get(instance)
            if owner is None or not owner.step_open:
                return original(instance, *args, **kwargs)
            with owner.runtime.scope("scheduler"):
                return original(instance, *args, **kwargs)

        setattr(cls, marker, original)
        setattr(cls, "step", profiled_scheduler)

        def restore() -> None:
            setattr(cls, "step", original)
            try:
                delattr(cls, marker)
            except AttributeError:
                pass

        self.class_restores.append(restore)

    def close(self, *, partial: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        for trainer, binding in list(self.bindings.items()):
            try:
                if binding.step_open:
                    binding.finish_step(trainer, partial=True)
            except Exception as exc:
                errors.append(
                    f"finish Trainer {binding.trainer_id}: {type(exc).__name__}: {exc}"
                )
            try:
                binding.restore()
            except Exception as exc:
                errors.append(
                    f"restore Trainer {binding.trainer_id}: {type(exc).__name__}: {exc}"
                )
        while self.class_restores:
            try:
                self.class_restores.pop()()
            except Exception as exc:
                errors.append(f"restore class patch: {type(exc).__name__}: {exc}")
        if self.runtime is not None:
            try:
                self.runtime.close(partial=partial or bool(errors))
            except Exception as exc:
                errors.append(f"close runtime: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("KT step profiler cleanup errors: " + "; ".join(errors))


class _DisabledManager:
    def close(self, *, partial: bool = False) -> None:
        del partial


_INSTALL_LOCK = threading.Lock()
_ACTIVE_MANAGER: InstrumentationManager | None = None


def install(
    mode: str | ProfileMode = "phase",
) -> InstrumentationManager | _DisabledManager:
    """Install profiling once in this process and return its lifecycle owner."""

    parsed = ProfileMode.parse(mode)
    if parsed is ProfileMode.OFF:
        return _DisabledManager()
    global _ACTIVE_MANAGER
    with _INSTALL_LOCK:
        if _ACTIVE_MANAGER is not None and not _ACTIVE_MANAGER._closed:
            raise RuntimeError("KT step profiling is already installed in this process")
        _ACTIVE_MANAGER = InstrumentationManager(parsed).install()
        return _ACTIVE_MANAGER
