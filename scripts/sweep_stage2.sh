#!/bin/bash
# Stage 2 reduction versions 1 to 7 at powers of two from 2^20 to 2^28 elements plus two
# sizes that are not multiples of the block size. L2 flushed before each rep.
# Submit: sbatch --mem=16G --time=01:00:00 scripts/run_gpu.sh scripts/sweep_stage2.sh
set -euo pipefail
out=${1:-results/stage2/reduce.csv}
run() { "$@" --csv "$out" || echo "SWEEP_FAIL rc=$? cmd=$*"; }
for n in 1048576 4194304 16777216 67108864 268435456 1000003 16777233; do
  for v in 1 2 3 4 5 6 7; do run ./build/reduce $n --version $v --flush; done
done
