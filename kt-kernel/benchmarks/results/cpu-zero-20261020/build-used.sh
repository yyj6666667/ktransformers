#!/usr/bin/env bash
set -euo pipefail
experiment=/dev/shm/yyj-qwen35-35b-wbw-20261019/fp32-down-merge
variant=${1:-before}
export LD_LIBRARY_PATH=/mnt/runner_yyj/_work/_tool/Python/3.12.13/x64/lib:/usr/local/cuda-12.8/lib64
python_bin=/dev/shm/yyj-qwen35-35b-wbw-20261019/actor-kt/venv/bin/python
source_dir=$experiment/source-$variant
if [[ $variant == before && ! -f $source_dir/kt-kernel/CMakeLists.txt ]]; then
  tar -xzf "$experiment/kt-fp32-merge-before-source.tar.gz" -C "$source_dir"
fi
cmake -S "$source_dir/kt-kernel" -B "$experiment/build-$variant" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$python_bin" \
  -DLLAMA_NATIVE=OFF -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON \
  -DLLAMA_AVX512=ON -DLLAMA_AVX512_FANCY_SIMD=ON \
  -DLLAMA_AVX512_VNNI=ON -DLLAMA_AVX512_BF16=ON -DLLAMA_AVX512_VBMI=ON \
  -DKTRANSFORMERS_CPU_USE_AMX_AVX512=ON -DKTRANSFORMERS_CPU_USE_AMX=OFF \
  -DKTRANSFORMERS_USE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DKTRANSFORMERS_CUDA_STATIC_RUNTIME=OFF \
  -DCPUINFER_ENABLE_LTO=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  > "$experiment/logs/configure-$variant.log" 2>&1
cmake --build "$experiment/build-$variant" --target kt_kernel_ext --parallel 12 \
  > "$experiment/logs/build-$variant.log" 2>&1
date -Is > "$experiment/build-$variant/complete.txt"
