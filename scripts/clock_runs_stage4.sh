#!/bin/bash
# Stage 4: SGEMM and streaming runs with GPU clocks, power, and temperature sampled
# every 100 ms by nvidia-smi in the background. Each run is bracketed by start/end
# markers in the same timestamp format as the nvidia-smi log, so samples can be
# attributed to runs (scripts/clock_summary.py).
# Submit: sbatch --mem=16G --time=01:00:00 scripts/run_gpu.sh scripts/clock_runs_stage4.sh
set -euo pipefail
out=results/stage4
mkdir -p "$out"
log=$out/gpu_log.csv
markers=$out/gpu_log_markers.csv
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu \
  --format=csv,nounits -lms 100 > "$log" &
logger=$!
trap 'kill $logger 2>/dev/null || true' EXIT
echo "timestamp,event,label" > "$markers"
stamp() { date "+%Y/%m/%d %H:%M:%S.%3N"; }
run() {  # run <label> <csv> <cmd...>
  local label=$1 csv=$2
  shift 2
  echo "$(stamp),start,$label" >> "$markers"
  "$@" --csv "$csv" || echo "RUN_FAIL rc=$? label=$label"
  echo "$(stamp),end,$label" >> "$markers"
}
sleep 2  # idle baseline
for v in 0 4 5 6; do
  run "sgemm_v${v}_4096" "$out/clocked_sgemm.csv" ./build/sgemm 4096 --version $v --reps 1000
  run "sgemm_v${v}_8192" "$out/clocked_sgemm.csv" ./build/sgemm 8192 --version $v --reps 200
done
# 2 GiB of total data per launch, warm; long runs so the clock settles.
run "vector_add_2GiB" "$out/clocked_stream.csv" ./build/vector_add 178956970 --reps 3000
run "copy_2GiB" "$out/clocked_stream.csv" ./build/copy 268435456 --mode kernel --reps 3000
run "reduce_v5_2p28" "$out/clocked_stream.csv" ./build/reduce 268435456 --version 5 --reps 3000 --no-csv
sleep 2
