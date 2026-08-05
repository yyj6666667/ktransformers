#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[dsv4-entrypoint] %s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  exec "$@"
fi

MODEL_PATH="${MODEL_PATH:-/model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
KT_GPU_EXPERTS="${KT_GPU_EXPERTS:-0}"
KT_GPU_PREFILL_TOKEN_THRESHOLD="${KT_GPU_PREFILL_TOKEN_THRESHOLD:-2048}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$CHUNKED_PREFILL_SIZE}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1200}"
ENABLE_MTP="${ENABLE_MTP:-0}"

[[ "$TP" =~ ^[1-9][0-9]*$ ]] ||
  die "TP must be a positive integer"

[[ -r "$MODEL_PATH/config.json" ]] ||
  die "missing model config: $MODEL_PATH/config.json (mount the model directory at /model)"
find "$MODEL_PATH" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name '*.gguf' \) -print -quit | grep -q . ||
  die "no model weight files found below $MODEL_PATH"

# nvidia-smi is useful for diagnostics but is not the CUDA capability contract:
# stripped NVIDIA runtimes can expose CUDA correctly without shipping this tool.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi is unavailable; relying on PyTorch CUDA validation"
elif ! nvidia-smi -L >/dev/null 2>&1; then
  log "nvidia-smi did not report a GPU; relying on PyTorch CUDA validation"
fi

GPU_INFO="$(
  TP="$TP" python - <<'PY'
import os
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")

tp = int(os.environ["TP"])
visible = torch.cuda.device_count()
if visible < tp:
    raise SystemExit(
        f"TP={tp} requires at least {tp} visible GPU(s), but PyTorch sees {visible}. "
        "Set GPU_DEVICE to a comma-separated CUDA device list, for example GPU_DEVICE=0,1."
    )

supported = {
    (8, 0): ("8.0", "8.0"),
    (8, 6): ("8.6", "8.6"),
    (8, 9): ("8.9", "8.9"),
    (9, 0): ("9.0+PTX", "9.0a"),
    (10, 0): ("10.0+PTX", "10.0a"),
    (12, 0): ("12.0+PTX", "12.0a"),
}

torch_archs = []
flashinfer_archs = []
gpu_summary = []
for device_index in range(tp):
    major, minor = torch.cuda.get_device_capability(device_index)
    arches = supported.get((major, minor))
    if arches is None:
        raise SystemExit(
            f"unsupported GPU compute capability SM_{major}{minor} on visible GPU {device_index}; "
            "supported: 80 86 89 90 100 120"
        )
    torch_arch, flashinfer_arch = arches
    if torch_arch not in torch_archs:
        torch_archs.append(torch_arch)
    if flashinfer_arch not in flashinfer_archs:
        flashinfer_archs.append(flashinfer_arch)
    name = torch.cuda.get_device_name(device_index).replace(" ", "_").replace("|", "_")
    gpu_summary.append(f"{device_index}:{name}:SM_{major}{minor}")

print(
    "|".join(
        (
            str(visible),
            ";".join(torch_archs),
            ";".join(flashinfer_archs),
            ",".join(gpu_summary),
        )
    )
)
PY
)" || die "PyTorch cannot access CUDA; expose an NVIDIA GPU to the container"
IFS='|' read -r VISIBLE_GPU_COUNT DEFAULT_TORCH_ARCH DEFAULT_FLASHINFER_ARCH GPU_SUMMARY <<< "$GPU_INFO"
[[ "$VISIBLE_GPU_COUNT" =~ ^[0-9]+$ ]] ||
  die "could not determine the number of visible GPUs"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$DEFAULT_TORCH_ARCH}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-$DEFAULT_FLASHINFER_ARCH}"
export SGLANG_DSV4_MODE="${SGLANG_DSV4_MODE:-2604}"
export SGLANG_DSV4_2604_SUBMODE="${SGLANG_DSV4_2604_SUBMODE:-2604B}"

# The image is built with all kt-kernel variants.  DSV4 MXFP4 has an AVX2
# backend, so AVX512/AMX are performance upgrades rather than requirements.
grep -qm1 -w avx2 /proc/cpuinfo && grep -qm1 -w fma /proc/cpuinfo ||
  die "DeepSeek-V4 CPU offload requires an x86-64 CPU with AVX2 and FMA"
if ! grep -qm1 -w avx512f /proc/cpuinfo; then
  log "AVX512F is unavailable; using the included AVX2 MXFP4 backend (lower throughput)"
fi

MEM_TOTAL_KIB="$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo)"
RECOMMENDED_MEMORY_KIB=$((256 * 1024 * 1024))
if [[ "$MEM_TOTAL_KIB" =~ ^[0-9]+$ ]] && (( MEM_TOTAL_KIB < RECOMMENDED_MEMORY_KIB )); then
  log "WARNING: detected $((MEM_TOTAL_KIB / 1024 / 1024)) GiB RAM; 256 GiB is recommended. Continuing, but reduce context/concurrency if the host runs out of memory."
fi

PHYSICAL_CORES="$(
  lscpu -p=CORE,SOCKET |
    awk -F, '!/^#/ {print $1 "," $2}' |
    sort -u |
    wc -l
)"
NUMA_NODES="$(
  lscpu -p=NODE |
    awk -F, '!/^#/ && $1 >= 0 {print $1}' |
    sort -u |
    wc -l
)"
(( PHYSICAL_CORES > 0 )) || die "could not detect physical CPU cores"
(( NUMA_NODES > 0 )) || NUMA_NODES=1

if [[ -z "${KT_CPUINFER_THREADS:-}" ]]; then
  if (( PHYSICAL_CORES >= 64 )); then
    KT_CPUINFER_THREADS=$((PHYSICAL_CORES - 4))
  else
    KT_CPUINFER_THREADS="$PHYSICAL_CORES"
  fi
fi
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-$NUMA_NODES}"

for value_name in PORT CONTEXT_LENGTH KT_GPU_EXPERTS KT_CPUINFER_THREADS \
  KT_THREADPOOL_COUNT KT_GPU_PREFILL_TOKEN_THRESHOLD CHUNKED_PREFILL_SIZE \
  MAX_PREFILL_TOKENS WATCHDOG_TIMEOUT; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$value_name must be a non-negative integer"
done
[[ "$MEM_FRACTION" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] ||
  die "MEM_FRACTION must be between 0 and 1"
[[ "$ENABLE_MTP" == 0 || "$ENABLE_MTP" == 1 ]] ||
  die "ENABLE_MTP must be 0 or 1"

(( CHUNKED_PREFILL_SIZE % 256 == 0 )) ||
  die "CHUNKED_PREFILL_SIZE must be a multiple of 256 for DeepSeek-V4 paged KV cache"

# The layerwise slots are physically allocated by the first long request, but
# their capacity is held out of KV-cache profiling.  Keep the KV budget itself
# automatic.  The default SWA ratio gives a 4096-token prefill chunk enough
# SWA KV pages while retaining a useful full-attention KV budget.
if (( KT_GPU_PREFILL_TOKEN_THRESHOLD > 0 )); then
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
  SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.4}"
else
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
fi

for value_name in MAX_RUNNING_REQUESTS; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$value_name must be a non-negative integer"
done
if [[ -n "$MAX_TOTAL_TOKENS" ]]; then
  [[ "$MAX_TOTAL_TOKENS" =~ ^[0-9]+$ ]] ||
    die "MAX_TOTAL_TOKENS must be a non-negative integer"
fi
if [[ -n "$SWA_FULL_TOKENS_RATIO" ]]; then
  [[ "$SWA_FULL_TOKENS_RATIO" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] ||
    die "SWA_FULL_TOKENS_RATIO must be between 0 and 1"
fi

cmd=(
  python -m sglang.launch_server
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL_PATH"
  --kt-weight-path "$MODEL_PATH"
  --kt-method MXFP4
  --kt-num-gpu-experts "$KT_GPU_EXPERTS"
  --kt-cpuinfer "$KT_CPUINFER_THREADS"
  --kt-threadpool-count "$KT_THREADPOOL_COUNT"
  --kt-gpu-prefill-token-threshold "$KT_GPU_PREFILL_TOKEN_THRESHOLD"
  --kt-enable-dynamic-expert-update
  --tensor-parallel-size "$TP"
  --context-length "$CONTEXT_LENGTH"
  --attention-backend flashinfer
  --mem-fraction-static "$MEM_FRACTION"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --watchdog-timeout "$WATCHDOG_TIMEOUT"
  --disable-shared-experts-fusion
  --trust-remote-code
  --cuda-graph-bs 1
  --cuda-graph-max-bs 1
  --disable-radix-cache
  --skip-server-warmup
)

if [[ -n "$MAX_TOTAL_TOKENS" ]]; then
  cmd+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

if [[ -n "$SWA_FULL_TOKENS_RATIO" ]]; then
  cmd+=(--swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO")
fi

if [[ "$ENABLE_MTP" == 1 ]]; then
  cmd+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --speculative-moe-runner-backend auto
  )
fi
cmd+=("$@")

numa_prefix=()
if command -v numactl >/dev/null 2>&1 &&
  numactl --interleave=all true >/dev/null 2>&1; then
  numa_prefix=(numactl --interleave=all)
else
  log "NUMA interleave is unavailable; continuing without numactl"
fi

log "GPUs=${GPU_SUMMARY}, visible_gpus=${VISIBLE_GPU_COUNT}, TP=${TP}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}, torch_arch=${TORCH_CUDA_ARCH_LIST}, flashinfer_arch=${FLASHINFER_CUDA_ARCH_LIST}"
log "CPU cores=${PHYSICAL_CORES}, cpuinfer=${KT_CPUINFER_THREADS}, NUMA pools=${KT_THREADPOOL_COUNT}"
printf '[dsv4-entrypoint] launching:' >&2
printf ' %q' "${numa_prefix[@]}" "${cmd[@]}" >&2
printf '\n' >&2

exec "${numa_prefix[@]}" "${cmd[@]}"
