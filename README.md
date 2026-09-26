# KernelForge

GPU kernel optimization and performance analysis suite.

Central question: why is a GPU workload slow, what should change, and what
hardware/software mechanism explains the resulting performance change?

**Status:** Stage 0 (harness and baselines) complete. Stage 1 (memory access patterns) in progress. Progress log: `docs/progress.md`.

**Primary environment:** UC Davis HPC (Hive), NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition (sm_120, 188 SMs, 128 MiB L2), CUDA 13.3, GCC 13.2, CMake 3.28.
Nsight Compute counters are not accessible on Hive nodes probed so far; analysis
uses event timing, Nsight Systems, ptxas reports, occupancy API, and SASS.

## Build and run

```bash
source scripts/modules.sh
cmake -S . -B build && cmake --build build -j
sbatch scripts/run_gpu.sh ./build/vector_add 67108864 --flush
```

Common flags: `<n>`, `--flush` (evict L2 before each timed rep), `--launches K`
(K launches per event pair), `--csv PATH` or `--no-csv`.

## Results

Stage 0, 2 GiB total data, L2 flushed, median of 100 reps:

| Kernel | Bandwidth (GB/s) | % of 1792.1 GB/s nominal |
|---|---|---|
| vector_add | 1528.5 | 85.3 |
| copy kernel | 1493.7 | 83.3 |
| cudaMemcpy device-to-device | 1468.6 | 81.9 |

Details: `docs/stage0.md`.
