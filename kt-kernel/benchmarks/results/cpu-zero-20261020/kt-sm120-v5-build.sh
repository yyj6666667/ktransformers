#!/usr/bin/env bash
set -euo pipefail
experiment=/dev/shm/yyj-qwen35-35b-wbw-20261019/fp32-down-merge
zero=$experiment/cpu-zero-20261020
cp -a "$experiment/source-sm120-v4" "$experiment/source-sm120-v5"
cp "$zero/scripts/sm120_bf16_math.hpp" "$zero/scripts/sm120_bf16_moe.hpp" "$experiment/source-sm120-v5/kt-kernel/operators/avx512/"
cp "$zero/scripts/ext_bindings.cpp" "$experiment/source-sm120-v5/kt-kernel/"
bash "$zero/scripts/build-used.sh" sm120-v5
mkdir -p "$experiment/overlays/sm120-v5"
cp -a "$experiment/overlays/after-v3/kt_kernel" "$experiment/overlays/sm120-v5/"
cp "$experiment/build-sm120-v5/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so" "$experiment/overlays/sm120-v5/kt_kernel/_kt_kernel_ext_avx512_bf16.cpython-312-x86_64-linux-gnu.so"
