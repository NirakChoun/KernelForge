#!/bin/bash
# Stage 10 capstone: Triton vs hand-written CUDA for matmul and softmax, with GPU clocks,
# power, and temperature sampled every 100 ms by nvidia-smi in the background. The
# driver writes start/end markers per run; scripts/clock_summary.py then gives the
# clock of every run in results/stage10/clock_summary.csv (joined on `label`).
# Submit: sbatch --mem=64G --time=01:30:00 scripts/run_gpu.sh scripts/stage10_capstone.sh
set -euo pipefail
out=results/stage10
if compgen -G "$out/*.csv" > /dev/null; then
  echo "$out already has results; move them before rerunning" >&2
  exit 1
fi
mkdir -p "$out"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
  --format=csv,nounits -lms 100 > "$out/gpu_log.csv" &
logger=$!
trap 'kill $logger 2>/dev/null || true' EXIT
echo "timestamp,event,label" > "$out/gpu_log_markers.csv"
sleep 2  # idle baseline
(cd python && ../.venv/bin/python stage10_capstone.py --outdir "$out" "$@")
sleep 1
kill $logger 2>/dev/null || true
python3 scripts/clock_summary.py "$out" > /dev/null
echo "clock summary: $(wc -l < "$out/clock_summary.csv") lines"
# Static SASS of the CUDA SGEMM kernels compared in the capstone (v6 both paths, v5 128x128x16x8x8).
cuobjdump -sass build/sgemm | c++filt | python3 scripts/sass_stats.py \
  | grep -E '^"void sgemm_(vec<128, 128, 8, 8, 8|2d<128, 128, 16, 8, 8)|^function' > "$out/sass_ops_cuda_sgemm.csv"
