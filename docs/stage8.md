# Stage 8: ML kernels (softmax and RMSNorm)

## Question

How close do row-wise softmax and RMSNorm in hand-written CUDA, Triton, and PyTorch get to the 1530 GB/s achievable DRAM bandwidth across row lengths from 128 to 16384, including lengths that are not powers of two?

## Setup

Three implementations of each operation (CUDA, Triton, PyTorch) run on the same 256 MiB inputs with L2 flushed, after a check against an FP64 reference.

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 188 SMs, hive-dc-7-4-58, `high` |
| Software | PyTorch 2.14.0+cu130, Triton 3.8.0; CUDA kernels built with CUDA 13.3, Release, sm_120 |
| Code | `src/ml_kernels.cu` (built as `build/libkf_ml.so`, called through ctypes), `python/triton_kernels.py`, `python/stage8_ml.py` |
| Runs | `sbatch --mem=32G --time=01:00:00 scripts/run_gpu.sh scripts/stage8_ml.sh`: job 24048749 (row lengths 128 to 16384, 1000, 3000, 5000); job 24049119 (`--cols 1001,4099`) |
| Earlier failed run | job 24048053: uncommitted earlier version of the CUDA kernels, failed correctness (Observation 7) |
| Problem | FP32, 2^26 elements per shape (256 MiB in, 256 MiB out), rows = 2^26 // cols. Input `randn`; RMSNorm weight uniform [0.5, 1.5), eps 1e-6 |
| Timing | CUDA events, 10 warm-up, 100 reps, median; 256 MiB L2 flush before every rep |
| Bandwidth | compulsory bytes / median: 8 B per element (read x, write y) plus 4 cols B for the RMSNorm weight |
| Output | `results/stage8/ml_kernels.csv` (78 rows) |

Implementations:

| Name | Description |
|---|---|
| cuda_softmax / cuda_rmsnorm | one block per row, threads = smallest power of two >= cols / 16, clamped to 32..1024. Pass 1 computes the row statistic (online max and sum of exponentials; sum of squares), block reduction with warp shuffles, pass 2 re-reads the row and writes. float4 path when cols % 4 == 0, scalar path otherwise |
| triton_softmax / triton_rmsnorm | one program per row, the whole row loaded once into registers (BLOCK = next power of two of cols, masked), num_warps autotuned over {1, 2, 4, 8, 16, 32} |
| torch_softmax / torch_rmsnorm | `torch.softmax(x, dim=-1)`, `F.rms_norm(x, (cols,), w, eps)` |

Correctness, before timing: every implementation against an FP64 reference computed by PyTorch from the same input, per element |y - ref| <= 1e-7 + 1e-5 |ref|. All 78 checks passed. Largest normalized error: 1.03e-6 (triton_softmax, 4099 columns). Row lengths 1000, 3000, 5000, 1001, 4099 are not powers of two; 1001 and 4099 are not multiples of 4 and use the CUDA scalar path.

## Results

CUDA and Triton run at 92.6% to 97.9% of 1530 GB/s at every row length, with Triton equal or ahead in all but one case. PyTorch softmax stays at 94.7% to 97.3%; PyTorch `F.rms_norm` drops to 50.1% at 1001 columns.

GB/s and % of 1530 GB/s, L2 flushed. Threads per block for CUDA; Triton num_warps chosen by the autotuner.

### Softmax

| Cols | Rows | CUDA | Triton | PyTorch | CUDA threads / regs | Triton warps / regs |
|---|---|---|---|---|---|---|
| 128 | 524288 | 1464.4 (95.7) | 1481.0 (96.8) | 1472.7 (96.3) | 32 / 35 | 1 / 16 |
| 256 | 262144 | 1472.6 (96.2) | 1489.5 (97.3) | 1464.5 (95.7) | 32 / 35 | 4 / 16 |
| 512 | 131072 | 1448.3 (94.7) | 1481.0 (96.8) | 1464.5 (95.7) | 32 / 35 | 1 / 38 |
| 1000 | 67108 | 1440.3 (94.1) | 1472.7 (96.3) | 1456.3 (95.2) | 64 / 35 | 16 / 17 |
| 1001 | 67041 | 1417.0 (92.6) | 1472.7 (96.3) | 1456.3 (95.2) | 64 / 24 (scalar) | 16 / 17 |
| 1024 | 65536 | 1448.2 (94.7) | 1481.0 (96.8) | 1472.7 (96.3) | 64 / 35 | 1 / 54 |
| 2048 | 32768 | 1456.4 (95.2) | 1489.5 (97.3) | 1448.3 (94.7) | 128 / 35 | 16 / 18 |
| 3000 | 22369 | 1448.3 (94.7) | 1468.6 (96.0) | 1456.3 (95.2) | 256 / 35 | 16 / 30 |
| 4096 | 16384 | 1448.3 (94.7) | 1498.0 (97.9) | 1472.7 (96.3) | 256 / 35 | 32 / 17 |
| 4099 | 16372 | 1440.4 (94.1) | 1464.5 (95.7) | 1456.4 (95.2) | 512 / 24 (scalar) | 32 / 24 |
| 5000 | 13421 | 1456.3 (95.2) | 1472.6 (96.3) | 1456.3 (95.2) | 512 / 35 | 32 / 24 |
| 8192 | 8192 | 1456.4 (95.2) | 1489.5 (97.3) | 1489.5 (97.3) | 512 / 35 | 32 / 20 |
| 16384 | 4096 | 1464.5 (95.7) | 1481.0 (96.8) | 1464.4 (95.7) | 1024 / 35 | 32 / 34 |

### RMSNorm

| Cols | Rows | CUDA | Triton | PyTorch | CUDA threads / regs | Triton warps / regs |
|---|---|---|---|---|---|---|
| 128 | 524288 | 1481.0 (96.8) | 1481.0 (96.8) | 1365.3 (89.2) | 32 / 40 | 2 / 16 |
| 256 | 262144 | 1464.5 (95.7) | 1472.6 (96.2) | 1464.5 (95.7) | 32 / 40 | 8 / 15 |
| 512 | 131072 | 1464.5 (95.7) | 1424.7 (93.1) | 1464.5 (95.7) | 32 / 40 | 16 / 15 |
| 1000 | 67108 | 1456.3 (95.2) | 1472.6 (96.3) | 1448.3 (94.7) | 64 / 40 | 16 / 17 |
| 1001 | 67041 | 1417.0 (92.6) | 1472.7 (96.3) | 766.5 (50.1) | 64 / 18 (scalar) | 16 / 17 |
| 1024 | 65536 | 1464.5 (95.7) | 1481.1 (96.8) | 1456.4 (95.2) | 64 / 40 | 4 / 29 |
| 2048 | 32768 | 1472.7 (96.3) | 1481.1 (96.8) | 1456.4 (95.2) | 128 / 40 | 1 / 117 |
| 3000 | 22369 | 1456.3 (95.2) | 1472.7 (96.3) | 1448.3 (94.7) | 256 / 40 | 16 / 28 |
| 4096 | 16384 | 1464.5 (95.7) | 1498.0 (97.9) | 1358.3 (88.8) | 256 / 40 | 32 / 20 |
| 4099 | 16372 | 1448.4 (94.7) | 1464.5 (95.7) | 960.3 (62.8) | 512 / 18 (scalar) | 32 / 30 |
| 5000 | 13421 | 1456.3 (95.2) | 1472.7 (96.3) | 1327.3 (86.8) | 512 / 40 | 32 / 30 |
| 8192 | 8192 | 1464.6 (95.7) | 1489.5 (97.4) | 1065.7 (69.7) | 512 / 40 | 32 / 28 |
| 16384 | 4096 | 1481.2 (96.8) | 1481.2 (96.8) | 985.6 (64.4) | 1024 / 40 | 32 / 44 |

## Observations

The spread between implementations is a few percent except for `F.rms_norm`, and the CUDA scalar path is the slowest CUDA case.

1. CUDA and Triton softmax and RMSNorm run at 92.6% to 97.9% of 1530 GB/s at every row length from 128 to 16384.
2. Triton is equal to or faster than CUDA at every row length for softmax, and at every row length except 512 for RMSNorm (1424.7 vs 1464.5 GB/s).
3. The CUDA scalar path (1001 columns) is the slowest CUDA result: 1417.0 GB/s for both operations, against 1440.3 to 1456.3 GB/s for the float4 path at 1000 columns. At 4099 columns the scalar path runs at 1440.4 (softmax) and 1448.4 GB/s (RMSNorm).
4. PyTorch softmax stays within 94.7% to 97.3% of 1530 GB/s. PyTorch `F.rms_norm` falls to 50.1% at 1001 columns, 62.8% at 4099, 64.4% at 16384, and 69.7% at 8192.
5. The CUDA kernels read each row twice (statistic pass, then output pass) and still reach 94.1% to 96.8% of 1530 GB/s with the float4 path; the Triton kernels read each row once.
6. The Triton autotuner chose 1 warp for RMSNorm at 2048 columns, giving 117 registers per thread; softmax at 1024 columns also chose 1 warp (54 registers).
7. Correctness history: in an earlier, uncommitted version of the code, cuda_softmax and cuda_rmsnorm failed at every row length below 16384 (max normalized error 0.329 to 0.969) and passed at 16384, where the block has 1024 threads. The kernels were changed before commit; the committed version passed all checks.

## Interpretation TODO(Nirak)

- Whether the second read of each row in the CUDA kernels is served from L2, given that the float4 path is at most 3.4% slower than Triton's single read (softmax, 4096 columns).
- Why `F.rms_norm` loses bandwidth at large and non-multiple-of-4 row lengths while `torch.softmax` does not.
- Whether the remaining 2% to 7% gap to 1530 GB/s is launch and tail effects of one block per row or DRAM efficiency.

## Open questions

The main gaps are the unrecorded CUDA fix, timer resolution, the `F.rms_norm` dispatch, and DRAM byte counts.

1. The diff that fixed the CUDA reduction between the failed run and the committed version was not recorded by the previous session; the failure pattern (all block sizes below 1024) points at the block reduction for partial warp counts. Not reconstructed.
2. Timer resolution: median times are 0.358400 to 0.378880 ms for every CUDA and Triton run. Standard deviations are 0.0008 to 0.0023 ms. 51 of the 52 CUDA and Triton medians are multiples of 2.048 us to within 0.04 us (the other is a multiple of 1.024 us), so adjacent entries (for example 1464.5 and 1472.6 GB/s) differ by one 2.048 us step, about 0.6%. Differences below one step are not resolved at this problem size.
3. `F.rms_norm` at 1001 columns (766.5 GB/s) is half of `torch.softmax` at the same shape (1456.3 GB/s). The kernel PyTorch dispatches was not identified; Nsight Systems would name it.
4. Triton RMSNorm at 512 columns chose 16 warps for a 512-element row (one element per thread) and is the only case below CUDA.
5. SM clock was not logged in these jobs. Stage 4 recorded 300 W and 1515 to 2332 MHz under load; memory-bound kernels depend on the memory clock (13365 MHz under load in Stage 4).
6. DRAM bytes (to confirm the compulsory-byte count and the L2 hit on the second read) need `ncu` counters; pending counter access.
