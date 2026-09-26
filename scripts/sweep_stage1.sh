#!/bin/bash
# Stage 1 access patterns. 2^28 32-bit elements (1 GiB in, 1 GiB out) for contiguous,
# strided, gather, and offset; 2^26 structs (1 GiB AoS) for AoS vs SoA. All runs flush
# L2 before each rep; contiguous also runs warm.
# Submit: sbatch --mem=16G --time=00:30:00 scripts/run_gpu.sh scripts/sweep_stage1.sh
set -euo pipefail
out=${1:-results/stage1/patterns.csv}
B=./build/mem_patterns
N=268435456
NS=67108864
run() { "$@" --csv "$out" || echo "SWEEP_FAIL rc=$? cmd=$*"; }
run $B $N --pattern contiguous
run $B $N --pattern contiguous --flush
for s in 1 2 4 8 16 32 64; do run $B $N --pattern strided --stride $s --flush; done
run $B $N --pattern gather --flush
for o in $(seq 0 32); do run $B $N --pattern offset --offset $o --flush; done
for p in aos_x soa_x aos_sum soa_sum; do run $B $NS --pattern $p --flush; done
