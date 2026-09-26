#!/bin/bash
# Stage 7: Triton vector add, softmax, matmul (python/stage7_triton.py), then the Stage 3
# CUDA SGEMM versions and cuBLAS at the same matmul shapes in the same job, so the
# comparison is not across jobs.
# Submit: sbatch --mem=32G --time=01:30:00 scripts/run_gpu.sh scripts/stage7_triton.sh
set -euo pipefail
(cd python && ../.venv/bin/python stage7_triton.py --outdir results/stage7)
out=results/stage7/cuda_sgemm_same_job.csv
run() { "$@" --csv "$out" || echo "RUN_FAIL rc=$? cmd=$*"; }
for dims in "1024" "2048" "4096" "8192" "1000" "4097" "777 --ncols 1111 --k 333"; do
  run ./build/sgemm $dims --version 0
  run ./build/sgemm $dims --version 6
  run ./build/sgemm $dims --version 5 --cfg 128x128x16x8x8
done
