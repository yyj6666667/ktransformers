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

[[ -r "$MODEL_PATH/config.json" ]] ||
  die "missing model config: $MODEL_PATH/config.json (mount the model directory at /model)"
find "$MODEL_PATH" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name '*.gguf' \) -print -quit | grep -q . ||
  die "no model weight files found below $MODEL_PATH"

command -v nvidia-smi >/dev/null 2>&1 ||
  die "nvidia-smi is unavailable; start the container with an NVIDIA CDI device"
nvidia-smi -L >/dev/null 2>&1 ||
  die "no NVIDIA GPU is visible inside the container"

read -r SM_CODE GPU_NAME < <(
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor}", torch.cuda.get_device_name(0).replace(" ", "_"))
PY
)

case "$SM_CODE" in
  80)  DEFAULT_TORCH_ARCH="8.0";      DEFAULT_FLASHINFER_ARCH="8.0" ;;
  86)  DEFAULT_TORCH_ARCH="8.6";      DEFAULT_FLASHINFER_ARCH="8.6" ;;
  89)  DEFAULT_TORCH_ARCH="8.9";      DEFAULT_FLASHINFER_ARCH="8.9" ;;
  90)  DEFAULT_TORCH_ARCH="9.0+PTX";  DEFAULT_FLASHINFER_ARCH="9.0a" ;;
  100) DEFAULT_TORCH_ARCH="10.0+PTX"; DEFAULT_FLASHINFER_ARCH="10.0a" ;;
  120) DEFAULT_TORCH_ARCH="12.0+PTX"; DEFAULT_FLASHINFER_ARCH="12.0a" ;;
  *) die "unsupported GPU compute capability SM_${SM_CODE}; supported: 80 86 89 90 100 120" ;;
esac

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$DEFAULT_TORCH_ARCH}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-$DEFAULT_FLASHINFER_ARCH}"
export SGLANG_DSV4_MODE="${SGLANG_DSV4_MODE:-2604}"
export SGLANG_DSV4_2604_SUBMODE="${SGLANG_DSV4_2604_SUBMODE:-2604B}"

grep -qm1 -w avx512f /proc/cpuinfo ||
  die "DeepSeek-V4 CPU offload requires an x86 CPU with AVX512F"

MEM_TOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
MIN_MEMORY_KIB=$((256 * 1024 * 1024))
(( MEM_TOTAL_KIB >= MIN_MEMORY_KIB )) ||
  die "at least 256 GiB system memory is required"

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

for value_name in PORT TP CONTEXT_LENGTH KT_GPU_EXPERTS KT_CPUINFER_THREADS \
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

log "GPU=${GPU_NAME//_/ } SM_${SM_CODE}, torch_arch=${TORCH_CUDA_ARCH_LIST}, flashinfer_arch=${FLASHINFER_CUDA_ARCH_LIST}"
log "CPU cores=${PHYSICAL_CORES}, cpuinfer=${KT_CPUINFER_THREADS}, NUMA pools=${KT_THREADPOOL_COUNT}"
printf '[dsv4-entrypoint] launching:' >&2
printf ' %q' "${numa_prefix[@]}" "${cmd[@]}" >&2
printf '\n' >&2

exec "${numa_prefix[@]}" "${cmd[@]}"
