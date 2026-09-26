#!/bin/bash
# Stage 3 SGEMM: cuBLAS (version 0) and versions 1 to 6 with default configurations at
# square sizes 256 to 8192, plus sizes that are not tile multiples. Warm (no flush).
# Submit: sbatch --mem=16G --time=03:00:00 scripts/run_gpu.sh scripts/sweep_stage3.sh
set -euo pipefail
out=${1:-results/stage3/sgemm.csv}
run() { "$@" --csv "$out" || echo "SWEEP_FAIL rc=$? cmd=$*"; }
for dims in "256" "512" "1024" "2048" "4096" "8192" "1000" "1023" "4097" "777 --ncols 1111 --k 333"; do
  for v in 0 1 2 3 4 5 6; do run ./build/sgemm $dims --version $v; done
done
