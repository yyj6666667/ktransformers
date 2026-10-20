#!/usr/bin/env bash
set -euo pipefail
root=/dev/shm/yyj-qwen35-35b-wbw-20261019
zero=$root/fp32-down-merge/cpu-zero-20261020
export KT_MERGE_FIXTURE_ROOT=$root KT_KERNEL_CPU_VARIANT=avx512_bf16
export PYTHONPATH=$root/fp32-down-merge/overlays/sm120-v5:$zero/scripts
export LD_LIBRARY_PATH=/mnt/runner_yyj/_work/_tool/Python/3.12.13/x64/lib:/usr/local/cuda-12.8/lib64
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$root/actor-kt/venv/bin/python" -m pytest -q "$zero/scripts/test_sm120_bf16_compat.py" "$zero/scripts/test_bf16_fp32_merge.py" > "$zero/regression-final.log" 2>&1
cat "$zero/regression-final.log"
"$root/actor-kt/venv/bin/python" "$zero/scripts/cpu_zero_cpp_silu.py" --header "$zero/scripts/sm120_bf16_math.hpp" --output "$zero/cpp-silu-final" > "$zero/cpp-silu-final.log" 2>&1
cat "$zero/cpp-silu-final/results.json"
