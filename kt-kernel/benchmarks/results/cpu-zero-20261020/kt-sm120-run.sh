#!/usr/bin/env bash
set -euo pipefail
root=/dev/shm/yyj-qwen35-35b-wbw-20261019
experiment=$root/fp32-down-merge
variant=$1
shift
export KT_MERGE_FIXTURE_ROOT=$root KT_KERNEL_CPU_VARIANT=avx512_bf16
export PYTHONPATH=$experiment/overlays/$variant
export LD_LIBRARY_PATH=/mnt/runner_yyj/_work/_tool/Python/3.12.13/x64/lib:/usr/local/cuda-12.8/lib64
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 HF_HUB_DISABLE_PROGRESS_BARS=1
script=$1
shift
exec "$root/actor-kt/venv/bin/python" "$experiment/cpu-zero-20261020/scripts/$script" "$@"
