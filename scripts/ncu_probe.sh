#!/bin/bash
# Checks whether this node allows access to GPU performance counters.
# Run inside a Slurm job: sbatch -p low --gpus=<type>:1 --time=00:01:00 scripts/run_gpu.sh scripts/ncu_probe.sh
# Needs a multi-arch build in build-probe/ so the binary runs on every GPU type:
#   cmake -S . -B build-probe -DCMAKE_CUDA_ARCHITECTURES="80;86;89;120" && cmake --build build-probe --target vector_add
set -uo pipefail
out=$(ncu --metrics sm__cycles_elapsed.avg ./build-probe/vector_add 4096 --no-csv 2>&1)
rc=$?
echo "$out" | grep -E "==ERROR==|sm__cycles_elapsed|vector_add\(" || true
if echo "$out" | grep -q ERR_NVGPUCTRPERM; then result=DENIED
elif [ $rc -eq 0 ] && echo "$out" | grep -q sm__cycles_elapsed.avg; then result=ALLOWED
else result="ERROR(rc=$rc)"; fi
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "PROBE_RESULT node=$(hostname) job=${SLURM_JOB_ID:-none} gpu=\"$gpu\" counters=$result"
