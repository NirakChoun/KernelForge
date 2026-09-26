# KernelForge

GPU kernel optimization and performance analysis suite.

Central question: why is a GPU workload slow, what should change, and what hardware or software mechanism explains the measured result?

**Status:** Stages 0 to 8 and the Stage 10 capstone are measured and documented; Stage 9 was skipped by choice. Interpretation sections (`Interpretation TODO(Nirak)`) are pending the owner's review. Documentation index: [`docs/index.md`](docs/index.md). Progress log with commits, failures, and pending profiling: [`docs/progress.md`](docs/progress.md).

## Environment

All results come from one GPU type on the UC Davis Hive cluster, without Nsight Compute counters.

| Item | Value |
|---|---|
| Cluster | UC Davis HPC (Hive), Slurm account `publicgrp`, partition `high`, `--gpus=6000_blackwell:1` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, sm_120, 188 SMs, 128 MiB L2, 512-bit bus, 1792.1 GB/s nominal DRAM peak |
| Measured limits | about 1530 GB/s achievable DRAM bandwidth (L2 flushed); 300 W power limit under load; SM clock 1425 to 2347 MHz observed under load; memory clock 13365 MHz |
| Toolchain | CUDA 13.3, GCC 13.2, CMake 3.28.1 (`scripts/modules.sh`); driver 580.167.08 |
| Python | 3.10, PyTorch 2.14.0+cu130, Triton 3.8.0 (`requirements.txt`) |
| Profiling | Nsight Compute counters are denied (`ERR_NVGPUCTRPERM`) on every Hive node probed. Analysis uses CUDA event timing, Nsight Systems, `ptxas -v`, the CUDA occupancy API, `cuobjdump` SASS, and `nvidia-smi` clock logs |

## Results

Hand-written streaming and reduction kernels reach the achievable DRAM bandwidth (about 1530 GB/s). Hand-written FP32 SGEMM reaches 72.4% of cuBLAS at 8192; autotuned Triton reaches 92.6% to 93.2% at 4096 and 8192.

All numbers are medians of at least 100 timed reps after at least 10 warm-ups, on the GPU above. Every kernel passes a correctness check against a reference before it is timed.

Stage 0, streaming bandwidth, 2 GiB total data, L2 flushed ([`docs/stage0.md`](docs/stage0.md)):

| Kernel | GB/s | % of 1792.1 GB/s nominal |
|---|---|---|
| vector_add | 1528.5 | 85.3 |
| copy kernel | 1493.7 | 83.3 |
| cudaMemcpy device-to-device | 1468.6 | 81.9 |

Stage 1, access patterns, 2^28 uint32 elements, L2 flushed ([`docs/stage1.md`](docs/stage1.md)):

| Pattern | Effective GB/s | % of 1530 GB/s achievable |
|---|---|---|
| contiguous | 1493.7 | 97.6 |
| stride 2 / 4 / 8 | 1019.0 / 621.0 / 352.2 | 66.6 / 40.6 / 23.0 |
| random gather | 180.5 | 11.8 |
| AoS one field / SoA one field | 618.3 / 1489.5 | 40.4 / 97.4 |

Stage 2, float sum reduction, 2^28 elements, L2 flushed ([`docs/stage2.md`](docs/stage2.md)):

| Version | GB/s | % of CUB |
|---|---|---|
| v1 atomicAdd | 2.8 | 0.2 |
| v3 sequential addressing | 896.2 | 58.3 |
| v5 warp-shuffle last warp | 1528.5 | 99.4 |
| CUB DeviceReduce::Sum | 1537.5 | 100.0 |

Stage 3, FP32 SGEMM, M = N = K = 8192, cuBLAS verified FP32 (no TF32) ([`docs/stage3.md`](docs/stage3.md)):

| Version | GFLOP/s | % of cuBLAS |
|---|---|---|
| v1 naive | 832.0 | 1.6 |
| v3 shared-memory tiling | 6944.6 | 13.6 |
| v5 2D register blocking | 19573.3 | 38.2 |
| v6 float4 vectorized | 37066.3 | 72.4 |
| cuBLAS | 51188.3 | 100.0 |

Stages 4 to 6, roofline, clocks, and resources ([`docs/roofline.md`](docs/roofline.md)):

| Result | Value |
|---|---|
| FP32 peak | 188 x 128 x 2 x SM clock (148.7 TFLOPS at 3090 MHz) |
| Under load | 300 W, SM clocks 1515 to 2332 MHz |
| SGEMM at 8192, % of peak at observed clock | cuBLAS 71.4%, v6 47.5% |
| Memory-bound kernels (vector add, SAXPY, reduction v5 to v7) | 99.6% to 100.5% of 1530 GB/s |
| Occupancy | the fastest SGEMM tile configuration has the lowest theoretical occupancy (0.333) |

Stage 7, Triton, job 24047985 ([`docs/stage7.md`](docs/stage7.md)):

| Kernel | Result |
|---|---|
| vector add, 2^26, L2 flushed | 1512.4 GB/s (98.8% of 1530) |
| matmul 4096, FP32 (`input_precision="ieee"`) | 47127.3 GFLOP/s, 93.2% of cuBLAS; hand-written v6 73.6% |
| matmul 8192 | 46334.9 GFLOP/s, 92.6% of cuBLAS; v6 76.0% |
| matmul 777 x 1111 x 333 | 18714.9 GFLOP/s, 141.9% of cuBLAS |

Stage 8, softmax and RMSNorm, 2^26 elements, row lengths 128 to 16384, L2 flushed ([`docs/stage8.md`](docs/stage8.md)):

| Kernel | GB/s range (% of 1530) |
|---|---|
| CUDA softmax / RMSNorm | 1417.0 to 1472.6 (92.6 to 96.2) / 1417.0 to 1481.2 (92.6 to 96.8) |
| Triton softmax / RMSNorm | 1464.5 to 1498.0 (95.7 to 97.9) / 1424.7 to 1498.0 (93.1 to 97.9) |
| PyTorch softmax / `F.rms_norm` | 1448.3 to 1489.5 (94.7 to 97.3) / 766.5 to 1464.5 (50.1 to 95.7) |

Stage 10 capstone, when Triton matches or beats hand-written CUDA, job 24049608 ([`docs/capstone.md`](docs/capstone.md)):

| Result | Value |
|---|---|
| Triton vs v6 matmul | ahead at all 13 shapes (1.06x to 5.98x raw); at 8192, 62.0% vs 47.4% of clock-adjusted FP32 peak (cuBLAS 70.4%) |
| Triton vs cuBLAS matmul | ahead in raw GFLOP/s at 512, 1024, 1536, 777 x 1111 x 333, 1000 x 3000 x 2000; ahead per clock only at 512 and 777 x 1111 x 333 |
| Small shapes | v6's 128 x 128 tiles give 16 to 64 blocks for 188 SMs; Triton's autotuned tiles give 128 to 234 |
| Non-multiple-of-16 shapes | Triton emits 32-bit global accesses and 255-register kernels with spill; 81.3% of cuBLAS at 4097 vs 94.9% at 4096 |
| Softmax | Triton faster than the CUDA kernel at all 13 row lengths, 95.4% to 97.9% vs 92.1% to 95.7% of 1530 GB/s |

## Repository layout

CUDA sources and the shared harness are in `src/` and `include/`, Triton and Python drivers in `python/`, run scripts in `scripts/`, and every measured number in `results/`.

| Path | Contents |
|---|---|
| `src/` | CUDA benchmarks (`vector_add`, `saxpy`, `copy`, `mem_patterns`, `reduce`, `sgemm`, `cublas_fp32_check`, `device_info`) and `ml_kernels.cu` (shared library for Stage 8 and 10) |
| `include/kf/` | shared harness: error checks, event timer, bench loop with L2 flush, CSV writer |
| `python/` | Triton kernels, the Python timing harness (`kfbench.py`), and drivers for Stages 7, 8, and 10 |
| `scripts/` | Slurm wrapper, environment report, per-stage run scripts, plotting, roofline, SASS and clock summaries |
| `results/stageN/` | CSV results (each row carries GPU, job ID, date, CUDA version, build type, size, launch configuration, warm-up and rep counts, median, min, standard deviation, metric), plots, ptxas and SASS reports |
| `docs/` | stage documents, roofline analysis, capstone, progress log |

## How to reproduce

Build on a login node, then submit each stage's run script through Slurm.

On a Hive login node (build and submit only; all GPU work runs through Slurm, one job at a time):

```bash
git clone git@github.com:NirakChoun/KernelForge.git && cd KernelForge
source scripts/modules.sh                       # cmake/3.28.1 gcc/13.2.0 cuda/13.3.0
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cmake -S . -B build && cmake --build build -j 2
```

Every GPU run goes through `scripts/run_gpu.sh` (account `publicgrp`, partition `high`, one `6000_blackwell` GPU), which loads the modules and prints an environment report before running its arguments:

```bash
sbatch scripts/run_gpu.sh ./build/vector_add 67108864 --flush                      # single run
sbatch --mem=16G --time=01:00:00 scripts/run_gpu.sh scripts/sweep_stage0.sh         # Stage 0 sweep
sbatch scripts/run_gpu.sh scripts/fixed_stage0.sh                                   # Stage 0 fixed sizes
sbatch --mem=16G --time=00:30:00 scripts/run_gpu.sh scripts/sweep_stage1.sh         # Stage 1
sbatch --mem=16G --time=01:00:00 scripts/run_gpu.sh scripts/sweep_stage2.sh         # Stage 2
sbatch --mem=16G --time=03:00:00 scripts/run_gpu.sh scripts/sweep_stage3.sh         # Stage 3
sbatch --mem=16G --time=01:00:00 scripts/run_gpu.sh scripts/tile_sweep_stage3.sh    # Stage 3 tile sweep
sbatch --mem=16G --time=01:00:00 scripts/run_gpu.sh scripts/clock_runs_stage4.sh    # Stage 4 clocks and power
sbatch --mem=16G --time=00:30:00 scripts/run_gpu.sh scripts/reduce_tree_stage4.sh   # Stage 4 reduction tree phase
sbatch --mem=16G --time=01:30:00 scripts/run_gpu.sh scripts/naive_sweep_stage4.sh   # Stage 4 naive SGEMM sweep
sbatch --mem=32G --time=01:30:00 scripts/run_gpu.sh scripts/stage7_triton.sh        # Stage 7
sbatch --mem=32G --time=01:00:00 scripts/run_gpu.sh scripts/stage8_ml.sh            # Stage 8
sbatch --mem=64G --time=01:30:00 scripts/run_gpu.sh scripts/stage10_capstone.sh     # Stage 10
```

- Result CSVs are appended to, so move a stage's existing results before rerunning it (the Stage 10 script refuses to run over existing results).
- Binaries take `<n>`, `--flush` (evict L2 before each timed rep), `--launches K`, `--warmup W`, `--reps R`, and `--csv PATH` or `--no-csv`; `sgemm` also takes `--version V`, `--cfg C`, `--ncols N`, `--k K`.
- Plots and tables: `.venv/bin/python scripts/plot.py <stage>...`, `.venv/bin/python scripts/roofline.py`, `python3 scripts/clock_summary.py <results dir>`, `.venv/bin/python scripts/md_table.py <csv> <cols>`.
- Static reports: `scripts/kernel_report.sh <stage> <target>...` (ptxas and SASS), `scripts/nsys_kern_sum.sh` (Nsight Systems kernel summary).
- Profiling commands that need counter access are listed under Pending profiling in `docs/progress.md`.
