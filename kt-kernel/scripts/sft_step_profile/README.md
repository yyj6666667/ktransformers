# KT SFT step timeline profiler

This harness records one optimizer window on a common
`CLOCK_MONOTONIC_RAW` time axis without changing LLaMA-Factory,
Transformers, or Accelerate. It is intended to answer three different
questions without mixing their units:

1. Where wall time is spent: dataloader, each GAS microbatch's data prepare,
   forward and backward, gradient norm/clip, optimizer, KT post-update,
   scheduler, zero-grad, and log/save/eval.
2. When resident memory changes: process RSS, cgroup-v2 memory, PyTorch CUDA
   allocated/reserved bytes, and device-wide NVML used bytes.
3. Which subsystem can own an allocation or byte count: PyTorch allocator
   events from a deep trace and KT native pool events. RSS/cgroup/NVML changes
   are temporal correlations, not proof of allocation ownership.

The profiler never synchronizes CUDA at each phase boundary. Optional CUDA
events are queried non-blockingly after the step and may remain `pending` if
the stream has not completed. Host wall and CUDA-stream duration are separate
columns and must not be added together.

## Modes

- `off`: no Trainer patching and no sampling. Use this for the performance
  baseline.
- `phase`: low-overhead phase ledger, 20 ms host/cgroup/PyTorch samples,
  50 ms NVML samples, KT counters, and structured native-pool events.
- `deep`: everything in `phase`, plus `record_function` and NVTX ranges for a
  Torch Profiler/Nsight Systems capture. A deep run is diagnostic and is not a
  performance result.

## Launch

Put the checked-out sources ahead of site packages, then use the regular
Accelerate config and LLaMA-Factory YAML. The wrapper passes all arguments to
LLaMA-Factory unchanged.

```bash
export PYTHONPATH=/path/to/LLaMA-Factory/src:/path/to/transformers/src:/path/to/accelerate/src:/path/to/ktransformers/kt-kernel
export KT_STEP_PROFILE_MODE=phase
export KT_STEP_PROFILE_DIR=/path/to/results/phase
export KT_STEP_PROFILE_RUN_ID=qwen3-30b-b1-gas2-s512-phase
export KT_STEP_PROFILE_WARMUP_STEPS=2

accelerate launch --config_file /path/to/accelerate.yaml \
  /path/to/ktransformers/kt-kernel/scripts/sft_step_profile/profile_train_entrypoint.py \
  /path/to/train.yaml
```

For an `off` comparison, change only `KT_STEP_PROFILE_MODE=off`. This mode does
not create profiler artifacts. Run the same number of warmup and measured
steps, then derive both runs' per-step wall times from successive cumulative
`train_runtime` values in each `trainer_state.json`. Compare the medians after
discarding identical warmup steps; do not compare an all-step mean with the
phase ledger's measured-step median. `phase` overhead should be at most 5%.

For `deep`, enable LLaMA-Factory's existing upstream Torch Profiler settings
in a copy of the experiment YAML:

```yaml
enable_torch_profiler: true
profiler_output_dir: /path/to/results/deep/torch
profiler_wait_steps: 2
profiler_warmup_steps: 1
profiler_active_steps: 1
profiler_repeat: 1
profiler_record_shapes: true
profiler_profile_memory: true
profiler_with_stack: true
```

An Nsight Systems capture can wrap the same command. It does not require GPU
performance-counter access:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true -o /path/to/results/deep/nsys \
  accelerate launch --config_file /path/to/accelerate.yaml \
  /path/to/ktransformers/kt-kernel/scripts/sft_step_profile/profile_train_entrypoint.py \
  /path/to/deep.yaml
```

## CPU counters and memory bandwidth

The harness only reads hardware interfaces. It never mounts resctrl, moves a
PID into a control group, invokes sudo, or changes NVIDIA driver settings.

Core counters can be collected around the launcher with at most four events
to avoid excessive multiplexing:

```bash
perf stat --no-big-num -x, \
  -e cycles,instructions,cache-misses,context-switches \
  -o /path/to/results/phase/perf.csv -- \
  accelerate launch ...

# Run only after perf and training have exited.
python -m sft_step_profile.append_perf_stat \
  --perf-stat /path/to/results/phase/perf.csv \
  --output /path/to/results/phase/rank_0/hardware_counters.jsonl \
  --run-id qwen3-30b-b1-gas2-s512-phase --rank 0
```

On AMD hosts with resctrl MBM support, an administrator can mount resctrl and
create a monitor group before the run. Assign the training PIDs to that group,
then set `KT_STEP_PROFILE_RESCTRL_GROUP` to its path relative to the resctrl
root (for example `kt_sft`). The sampler reports
MBM total/local byte deltas and rates in `hardware_counters.jsonl`. These are
measured last-level-cache miss traffic estimates; they are not separate DRAM
read/write counters.

GPU allocator/device memory is available now. GPU SM and HBM hardware counters
are deliberately reported as unavailable when NVIDIA profiling is
administrator-only; the profiler does not reload the driver on a shared host.

## Artifact contract

Each rank writes an isolated `rank_<rank>/` directory:

- `run_meta.json`: clock, mode, source revisions, sampling contract.
- `phase_events.jsonl`: raw nested scope and marker events.
- `memory_samples.jsonl`: periodic and phase-boundary memory state.
- `allocation_events.jsonl`: deep-trace allocation/free events after
  post-processing.
- `kt_pool_events.jsonl`: structured KT native pool resize/use events.
- `hardware_counters.jsonl`: perf/resctrl samples and availability.
- `kt_step_profile.jsonl`: per-step KT C++ stage counters.
- `step_index.json`, `step_summary.json`, `step_summary.csv`: indexed step
  summaries.
- `timeline.trace.json`: a Chrome/Perfetto-compatible host timeline.

Torch and Nsight traces remain separate because they have different overhead
and clock domains. Deep mode emits raw-clock anchor markers named
`kt.clock_sync.raw_ns=<value>` so post-processing can align them rather than
assuming their timestamps are directly comparable.

## Interpretation rules

- Throughput uses the observed non-padding token delta from Trainer state,
  never `cutoff_len * batch_size`.
- `iteration_wall` spans batch acquisition through log/save/eval.
  `train_core` spans the first microbatch through zero-grad. They are overview
  spans, not additive phases.
- Leaf phase wall times plus `other` explain the iteration wall. Nested KT
  kernel timers are drill-down data and must not be added to their parent.
- A positive boundary delta means the counter is higher at phase end; a
  negative delta is a valid release. A sampled peak can be missed when a spike
  is shorter than the sampling period.
- KT shared pools are deduplicated by NUMA node using the maximum capacity.
  Per-layer cache/persistent pools are summed by layer and NUMA node.
