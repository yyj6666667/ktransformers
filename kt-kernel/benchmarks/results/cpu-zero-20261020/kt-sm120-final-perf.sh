#!/usr/bin/env bash
set -euo pipefail
zero=/dev/shm/yyj-qwen35-35b-wbw-20261019/fp32-down-merge/cpu-zero-20261020
mkdir -p "$zero/final-performance"
variants=(before after-v3 sm120-v5 sm120-v5 after-v3 before)
for index in "${!variants[@]}"; do
  variant=${variants[$index]}
  family=inference
  if [[ $variant == sm120-v5 ]]; then family=sm120; fi
  KT_SM120_INTERVAL=1 bash "$zero/scripts/kt-sm120-run.sh" "$variant" bench_sm120_compat.py     --family "$family" --fixture "$zero/performance-fixture.pt" --tokens 1,4,128,512     --warmup 20 --iterations 60 --tag "$index-$variant" --output "$zero/final-performance/$index-$variant.json"
done
KT_SM120_INTERVAL=0 bash "$zero/scripts/kt-sm120-run.sh" sm120-v5 bench_sm120_compat.py     --family sm120 --fixture "$zero/performance-fixture.pt" --tokens 1,4,128,512     --warmup 20 --iterations 60 --tag final-no-interval --output "$zero/final-performance/6-sm120-v5-no-interval.json"
