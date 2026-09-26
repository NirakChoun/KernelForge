#!/bin/bash
# Stage 4: map the v1 (naive SGEMM) size effect. Square M = N = K for every size from
# 1016 to 1032 and 4088 to 4104; v2 (coalesced) runs at the same sizes as a control.
# Submit: sbatch --mem=16G --time=01:30:00 scripts/run_gpu.sh scripts/naive_sweep_stage4.sh
set -euo pipefail
out=${1:-results/stage4/naive_sweep.csv}
run() { "$@" --csv "$out" || echo "RUN_FAIL rc=$? cmd=$*"; }
for n in $(seq 1016 1032) $(seq 4088 4104); do
  for v in 1 2; do run ./build/sgemm $n --version $v; done
done
