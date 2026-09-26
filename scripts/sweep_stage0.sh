#!/bin/bash
# Stage 0 size sweep: total data moved per launch from 1 MiB to 2 GiB, powers of two
# plus the 1.5x points between them for resolution around the 128 MiB L2.
# Each size runs warm (no flush), cold (--flush), and, up to 32 MiB, warm with 100
# launches per event pair. Runs sequentially because the CSV writer is single-writer.
# Submit: sbatch --time=01:00:00 --mem=16G scripts/run_gpu.sh scripts/sweep_stage0.sh
set -euo pipefail
out=${1:-results/stage0/sweep.csv}
mib=$((1 << 20))
sizes=()
for p in 0 1 2 3 4 5 6 7 8 9 10 11; do
  sizes+=($(( (1 << p) * mib )))
  [ $p -lt 11 ] && sizes+=($(( 3 * (1 << p) * mib / 2 )))
done
run() {  # run <binary args...>; failure is logged, sweep continues
  "$@" --csv "$out" || echo "SWEEP_FAIL rc=$? cmd=$*"
}
for total in "${sizes[@]}"; do
  va_n=$(( total / 12 ))   # vector add moves 3 floats per element
  cp_n=$(( total / 8 ))    # copy moves 2 floats per element
  for cmd in "./build/vector_add $va_n" "./build/copy $cp_n --mode kernel" "./build/copy $cp_n --mode memcpy"; do
    run $cmd
    run $cmd --flush
    if [ "$total" -le $(( 32 * mib )) ]; then run $cmd --launches 100; fi
  done
done
