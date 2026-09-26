#!/bin/bash
# Stage 3 tile and block size sweep for versions 3 to 5 at 1024, 2048, 4096. Warm.
# Submit: sbatch --mem=16G --time=01:00:00 scripts/run_gpu.sh scripts/tile_sweep_stage3.sh
set -euo pipefail
out=${1:-results/stage3/tile_sweep.csv}
run() { "$@" --csv "$out" || echo "SWEEP_FAIL rc=$? cmd=$*"; }
for n in 1024 2048 4096; do
  for c in 8 16 32; do run ./build/sgemm $n --version 3 --cfg $c; done
  for c in 64x64x8x8 32x32x8x4 64x64x16x8 128x64x8x8 64x64x8x16; do run ./build/sgemm $n --version 4 --cfg $c; done
  for c in 128x128x8x8x8 64x64x8x8x8 128x64x8x8x8 64x64x8x4x4 128x128x16x8x8; do run ./build/sgemm $n --version 5 --cfg $c; done
done
