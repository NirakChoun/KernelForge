# KernelForge

GPU kernel optimization and performance analysis suite.

Central question: why is a GPU workload slow, what should change, and what
hardware/software mechanism explains the resulting performance change?

**Status:** Stages 0 to 3 (harness and baselines, memory access patterns, parallel reduction, SGEMM) measured and documented; interpretation sections pending review. Progress log: `docs/progress.md`.

**Primary environment:** UC Davis HPC (Hive), NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition (sm_120, 188 SMs, 128 MiB L2), CUDA 13.3, GCC 13.2, CMake 3.28.
Nsight Compute counters are not accessible on the Hive nodes probed so far; analysis
uses event timing, Nsight Systems, ptxas reports, the occupancy API, and SASS.

## Build and run

```bash
source scripts/modules.sh
cmake -S . -B build && cmake --build build -j
sbatch scripts/run_gpu.sh ./build/vector_add 67108864 --flush
```

Common flags: `<n>`, `--flush` (evict L2 before each timed rep), `--launches K`
(K launches per event pair), `--csv PATH` or `--no-csv`. Every binary checks
correctness before timing and writes one CSV row per configuration to `results/`.

## Results

All numbers are medians of 100 timed reps after 10 warm-ups, on the GPU above.

Stage 0, streaming bandwidth, 2 GiB total data, L2 flushed (`docs/stage0.md`):

| Kernel | GB/s | % of 1792.1 GB/s nominal |
|---|---|---|
| vector_add | 1528.5 | 85.3 |
| copy kernel | 1493.7 | 83.3 |
| cudaMemcpy device-to-device | 1468.6 | 81.9 |

Stage 1, access patterns, 2^28 uint32 elements, L2 flushed (`docs/stage1.md`):

| Pattern | Effective GB/s | % of 1530 GB/s achievable |
|---|---|---|
| contiguous | 1493.7 | 97.6 |
| stride 2 / 4 / 8 | 1019.0 / 621.0 / 352.2 | 66.6 / 40.6 / 23.0 |
| random gather | 180.5 | 11.8 |
| AoS one field / SoA one field | 618.3 / 1489.5 | 40.4 / 97.4 |

Stage 2, float sum reduction, 2^28 elements, L2 flushed (`docs/stage2.md`):

| Version | GB/s | % of CUB |
|---|---|---|
| v1 atomicAdd | 2.8 | 0.2 |
| v3 sequential addressing | 896.2 | 58.3 |
| v5 warp-shuffle last warp | 1528.5 | 99.4 |
| CUB DeviceReduce::Sum | 1537.5 | 100.0 |

Stage 3, FP32 SGEMM, M = N = K = 8192 (`docs/stage3.md`):

| Version | GFLOP/s | % of cuBLAS |
|---|---|---|
| v1 naive | 832.0 | 1.6 |
| v3 shared-memory tiling | 6944.6 | 13.6 |
| v5 2D register blocking | 19573.3 | 38.2 |
| v6 float4 vectorized | 37066.3 | 72.4 |
| cuBLAS | 51188.3 | 100.0 |
