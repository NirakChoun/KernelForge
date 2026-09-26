#!/bin/bash
# Stage 0 fixed sizes (elements): three powers of two and one size that is not a
# multiple of the block size. Warm, cold, and (for the two small sizes) 100 launches
# per event pair.
# Submit: sbatch scripts/run_gpu.sh scripts/fixed_stage0.sh
set -euo pipefail
out=${1:-results/stage0/fixed_sizes.csv}
run() { "$@" --csv "$out" || echo "FIXED_FAIL rc=$? cmd=$*"; }
for n in 1048576 16777216 67108864 1000003; do
  for cmd in "./build/vector_add $n" "./build/saxpy $n" "./build/copy $n --mode kernel" "./build/copy $n --mode memcpy"; do
    run $cmd
    run $cmd --flush
    if [ "$n" -le 1048576 ]; then run $cmd --launches 100; fi
  done
done
