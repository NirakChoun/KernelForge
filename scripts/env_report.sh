#!/bin/bash
echo "=== KernelForge environment report ==="
echo "date:      $(date -Iseconds)"
echo "host:      $(hostname)"
echo "job:       ${SLURM_JOB_ID:-none}   partition: ${SLURM_JOB_PARTITION:-none}"
echo "--- GPU ---"
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,driver_version --format=csv
echo "--- toolchain ---"
nvcc --version | tail -2
gcc --version | head -1
cmake --version | head -1
echo "ncu:  $(command -v ncu  || echo 'not found')"
echo "nsys: $(command -v nsys || echo 'not found')"
