#!/bin/bash
# Stage 4 reduction tree-phase microbenchmark: first pass only of v2 to v5 and of the
# load-only baseline (v0). L2-resident sizes (4 and 16 MiB) warm with 100 launches per
# event pair; DRAM sizes (256 MiB and 1 GiB) with L2 flushed.
# Submit: sbatch --mem=16G --time=00:30:00 scripts/run_gpu.sh scripts/reduce_tree_stage4.sh
set -euo pipefail
out=${1:-results/stage4/reduce_tree.csv}
run() { "$@" --csv "$out" || echo "RUN_FAIL rc=$? cmd=$*"; }
for n in 1048576 4194304; do
  for v in 0 2 3 4 5; do run ./build/reduce $n --version $v --first-pass --launches 100; done
done
for n in 67108864 268435456; do
  for v in 0 2 3 4 5; do run ./build/reduce $n --version $v --first-pass --flush; done
done
