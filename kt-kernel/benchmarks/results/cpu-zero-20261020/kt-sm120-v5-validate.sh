#!/usr/bin/env bash
set -euo pipefail
zero=/dev/shm/yyj-qwen35-35b-wbw-20261019/fp32-down-merge/cpu-zero-20261020
export KT_SM120_INTERVAL=1
bash "$zero/scripts/kt-sm120-run.sh" sm120-v5 cpu_zero_operator_sweep.py --output "$zero/sweep-v5-all"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash "$zero/scripts/kt-sm120-run.sh" sm120-v5 replay_kt_to_golden.py --stages 12_kt_sm120 --kt-sm120 --cpu-prefill-reference "$zero/golden-prefill-fixtures/prefill-experts" --max-positions 32 --output "$zero/full-kt-v5-32"
bash "$zero/scripts/kt-sm120-run.sh" sm120-v5 bench_sm120_compat.py --family sm120 --fixture "$zero/performance-fixture.pt" --tokens 1,4,128,512 --warmup 5 --iterations 10 --tag v5-discovery --output "$zero/bench-v5-discovery.json"
