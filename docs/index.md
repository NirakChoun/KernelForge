# KernelForge documentation index

| Stage | Document | Topic |
|---|---|---|
| 0 | [stage0.md](stage0.md) | Harness, streaming baselines (vector add, SAXPY, copy), achievable DRAM bandwidth |
| 1 | [stage1.md](stage1.md) | Memory access patterns: stride, offset, gather, AoS vs SoA |
| 2 | [stage2.md](stage2.md) | Parallel reduction, v1 to v6 and CUB |
| 3 | [stage3.md](stage3.md) | FP32 SGEMM v1 to v6 against cuBLAS, tile sweep, cuBLAS FP32 verification |
| 4 to 6 | [roofline.md](roofline.md) | Roofline, clocks and power under load, occupancy and resource analysis |
| 7 | [stage7.md](stage7.md) | Triton vector add, softmax, matmul; autotuning and lowering stages |
| 8 | [stage8.md](stage8.md) | Softmax and RMSNorm in CUDA, Triton, and PyTorch |
| 9 | skipped | Skipped by choice; see [progress.md](progress.md) |
| 10 | [capstone.md](capstone.md) | When Triton matches or beats hand-written CUDA, and why |
| - | [progress.md](progress.md) | Stage reports, commits, counter access probes, pending profiling |

Results for each stage are in `results/stageN/`; the repository README has the headline numbers and the reproduction steps.
