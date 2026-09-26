# Stage 7: Triton

## Question

How close do autotuned Triton kernels for vector add, softmax, and FP32 matmul get to the hand-written CUDA kernels of Stages 0 and 3 and to cuBLAS, and what does the Triton compiler generate to get there?

## Setup

Three autotuned Triton kernels are checked against FP64 or exact references and timed with the same harness rules as the CUDA stages; the Stage 3 CUDA SGEMMs and cuBLAS run in the same job for the matmul comparison.

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 188 SMs, hive-dc-7-4-58, `high` |
| Software | PyTorch 2.14.0+cu130, Triton 3.8.0 (driver reports CUDA 13.0); CUDA kernels built with CUDA 13.3, Release, sm_120 |
| Code | `python/triton_kernels.py` (kernels and autotune spaces), `python/stage7_triton.py` (driver), `python/kfbench.py` (timing harness) |
| Run | `sbatch --mem=32G --time=01:30:00 scripts/run_gpu.sh scripts/stage7_triton.sh`, job 24047985, 2026-09-26 |
| Same-job CUDA baseline | the same script runs `build/sgemm` versions 0 (cuBLAS), 6, and 5 (128x128x16x8x8) at the Triton matmul shapes; `results/stage7/cuda_sgemm_same_job.csv` |
| Timing | CUDA events, 10 warm-up, 100 reps, median. A `torch.cuda._sleep` spin precedes each start event so Python launch latency is not timed. Vector add is timed warm and with a 256 MiB L2 flush; softmax and matmul warm |
| Precision | matmul `tl.dot(..., input_precision="ieee")`; PyTorch cuBLAS with `allow_tf32 = False`. The Triton SASS has 2048 FFMA and no HMMA or other MMA opcode; the PTX has no `mma` instruction |

Autotune spaces:

| Kernel | Key | Configurations |
|---|---|---|
| vector add | `n` | BLOCK in {1024, 2048, 4096, 8192} x num_warps in {4, 8} (8) |
| softmax | `n_cols` | num_warps in {1, 2, 4, 8, 16, 32}; BLOCK = next power of two of the row length (6) |
| matmul | `M, N, K` | 12 (BM, BN, BK, num_warps, num_stages) tuples listed in `MATMUL_SPACE`, GROUP_M = 8 (12) |

Correctness, before timing (all passed):

- Vector add: bit-exact against `x + y` in PyTorch, including n = 1000003.
- Softmax: max relative error against an FP64 PyTorch softmax <= 1e-5; worst 9.43e-7. Row length 3000 is not a power of two.
- Matmul: per element |c - ref| <= 16 sqrt(K) 2^-24 (|A||B|)_ij against an FP64 product, the Stage 3 criterion. Worst 3.82e-7 of (|A||B|)_ij at 8192 (tolerance 8.63e-5). Shapes 1000, 4097, and 777 x 1111 x 333 are not multiples of any tile size.

## Results

Triton matmul reaches 93.2% of cuBLAS at 4096 against 73.6% for hand-written v6, and Triton vector add reaches 98.8% of achievable bandwidth. Triton's matmul uses more registers and shared memory than v6 and a cp.async pipeline.

### Vector add (`results/stage7/vector_add.csv`)

| n | Autotuned config | Regs | L2 | Median (ms) | GB/s |
|---|---|---|---|---|---|
| 2^20 | BLOCK=1024, 8 warps | 18 | warm | 0.004128 | 3048.2 (launch-bound) |
| 2^20 | | | flushed | 0.010240 | 1228.8 (launch-bound) |
| 2^24 | BLOCK=4096, 8 warps | 42 | warm | 0.116736 | 1724.6 |
| 2^24 | | | flushed | 0.133120 | 1512.4 |
| 2^26 | BLOCK=4096, 8 warps | 42 | warm | 0.528384 | 1524.1 |
| 2^26 | | | flushed | 0.532480 | 1512.4 |
| 1000003 | BLOCK=2048, 4 warps | 80 | flushed | 0.008224 | 1459.1 (launch-bound) |

At 2^26 flushed, Triton vector add reaches 1512.4 GB/s, 98.8% of the 1530 GB/s achievable bandwidth. The Stage 0 CUDA vector add measured 1528.5 GB/s (flushed, 2 GiB total data). No kernel uses shared memory.

### Softmax (`results/stage7/softmax.csv`, warm, 2^24 elements)

| Rows x cols | Autotuned num_warps | Regs | Smem (B) | Median (ms) | GB/s (8 B/element) |
|---|---|---|---|---|---|
| 16384 x 1024 | 1 | 54 | 0 | 0.024576 | 5461.3 |
| 4096 x 4096 | 4 | 48 | 16 | 0.026592 | 5047.3 |
| 1024 x 16384 | 4 | 164 | 16 | 0.030720 | 4369.1 |
| 5592 x 3000 | 32 | 17 | 128 | 0.051200 | 2621.3 |

The 128 MiB working set (64 MiB in, 64 MiB out) fits in the 128 MiB L2 and the runs were warm, so these are not DRAM bandwidths. L2-flushed softmax and the CUDA comparison are in Stage 8.

### Matmul

From `results/stage7/matmul.csv` and `cuda_sgemm_same_job.csv` (job 24047985), warm. GFLOP/s = 2 M N K / median. Percentages are of `cublasSgemm` from `build/sgemm` in the same job.

| Shape | cuBLAS (`build/sgemm`) | cuBLAS (`torch.matmul`) | Triton | CUDA v6 | CUDA v5 |
|---|---|---|---|---|---|
| 1024 | 37056.2 | 38836.1 | 38802.5 (104.7%) | 13130.3 (35.4%) | 10292.8 (27.8%) |
| 2048 | 51987.1 | 53087.2 | 44855.1 (86.3%) | 34050.3 (65.5%) | 26363.7 (50.7%) |
| 4096 | 50557.9 | 50080.7 | 47127.3 (93.2%) | 37209.4 (73.6%) | 30174.5 (59.7%) |
| 8192 | 50029.8 | 50010.1 | 46334.9 (92.6%) | 38032.9 (76.0%) | 30468.2 (60.9%) |
| 1000 | 34473.2 | 36169.0 | 28709.2 (83.3%) | 12542.6 (36.4%) | 9736.0 (28.2%) |
| 4097 | 45829.2 | 45178.4 | 36174.2 (78.9%) | 34265.9 (74.8%) | 29098.2 (63.5%) |
| 777 x 1111 x 333 | 13191.1 | 14036.2 | 18714.9 (141.9%) | 8283.2 (62.8%) | 7048.4 (53.4%) |

Against `torch.matmul` in the same process, Triton is at 94.1% at 4096 and 92.7% at 8192.

Autotuned configuration and compiled resources (`results/stage7/matmul_autotune.csv`: 84 rows, 12 configurations x 7 shapes):

| Shape | Chosen (BM, BN, BK, warps, stages) | Regs | Spill (B) | Smem (B) | Best / worst autotune time (ms) |
|---|---|---|---|---|---|
| 1024 | 128, 64, 64, 8, 3 | 158 | 0 | 98304 | 0.0573 / 0.1963 |
| 2048 | 128, 64, 32, 4, 4 | 238 | 0 | 73728 | 0.4074 / 0.5284 |
| 4096 | 128, 128, 32, 8, 3 | 220 | 0 | 65536 | 2.9702 / 4.0837 |
| 8192 | 128, 128, 32, 8, 3 | 220 | 0 | 65536 | 23.2003 / 36.5492 |
| 1000 | 128, 64, 64, 8, 3 | 255 | 6 | 98304 | 0.0800 / 0.3686 |
| 4097 | 128, 128, 32, 8, 3 | 255 | 6 | 65536 | 3.5267 / 6.4584 |
| 777 x 1111 x 333 | 64, 64, 32, 4, 4 | 168 | 2 | 49152 | 0.0328 / 0.1246 |

The "Spill" column is Triton's `n_spills`. Worst configurations: 128x256x16 or 256x128x16 at 1024, 1000, and 777 x 1111 x 333; 32x64x32 (4 warps, 5 stages) at 4096 and 8192; 128x128x32 with 4 warps at 2048 and 4097.

Resources at 4096 against the Stage 3 kernels (Stage 3 values from `results/stage3/ptxas_sgemm.txt`):

| Kernel | Threads/block | Output tile | Regs | Smem (B) | Blocks/SM by regs (derived) |
|---|---|---|---|---|---|
| Triton, 128x128x32, 8 warps, 3 stages | 256 | 128 x 128 | 220 | 65536 | 1 |
| CUDA v6, 128x128x8x8x8 | 256 | 128 x 128 | 94 | 8192 | 2 |
| CUDA v5, 128x128x16x8x8 | 256 | 128 x 128 | 97 | 16384 | 2 |

Blocks per SM are derived from the register count and the 64K-register file; v6 and v5 match the Stage 3 occupancy API value of 0.333 (16 of 48 warps).

### Lowering stages for matmul at 4096 (`results/stage7/matmul_ir/`)

The block-level `tt.dot` becomes a pipelined loop of cp.async copies, 128-bit shared loads, and FP32 FMAs; no MMA instruction appears at any stage.

| Stage | File | Lines | What changes |
|---|---|---|---|
| Triton IR | `matmul_4096.ttir` | 208 | block-level `tt.load` / `tt.dot` / `tt.store` on tensors, no thread mapping |
| TritonGPU IR | `matmul_4096.ttgir` | 280 | layouts assigned (`#ttg.blocked`, e.g. `sizePerThread = [4, 4]`, `warpsPerCTA = [8, 1]`); shared buffers `2x128x32xf32` (A) and `2x32x128xf32` (B); 6 `async_copy_global_to_local` ops for the software pipeline |
| LLVM IR | `matmul_4096.ll` | 1626 | explicit per-thread address arithmetic and FP32 FMAs |
| PTX | `matmul_4096.ptx` | 3795 | 32 `cp.async`, 78 `ld.shared.v4`, 1984 `fma.rn.f32`, 0 `mma` |

The two shared buffers per operand (num_stages = 3 gives 2 buffers) total 2 x 128 x 32 x 4 + 2 x 32 x 128 x 4 = 65536 bytes, the reported shared memory.

Static SASS opcode counts (`cuobjdump -sass`, counts in the binary, not executed counts):

| Opcode class | Triton matmul (4096 config) | CUDA v6, float4 path | CUDA v6, scalar path |
|---|---|---|---|
| FFMA | 2048 | 64 | 64 |
| Global load | 24 LDGSTS.E.BYPASS.128 (global to shared, async) | 2 LDG.E.128.CONSTANT | 8 LDG.E.CONSTANT |
| Shared store | 16 STS.128 | 4 STS | 4 STS |
| Shared load | 176 LDS.128, 9 LDS | 4 LDS.128 | 4 LDS.128 |
| Global store | 16 STG.E.128 | 16 STG.E.128 | 64 STG.E |
| Barriers | 11 BAR.SYNC, 6 LDGDEPBAR, 2 DEPBAR.LE | 2 BAR.SYNC | 2 BAR.SYNC |

## Observations

Triton matmul leads v6 at every shape and reaches 92.6% to 93.2% of cuBLAS at 4096 and 8192; it falls further behind at 4097 and 1000, where its compile uses 255 registers with spill.

1. Triton matmul is ahead of hand-written v6 at every shape: 1.27x at 4096, 1.22x at 8192, 2.96x at 1024, 1.06x at 4097.
2. Triton reaches 92.6% to 93.2% of `cublasSgemm` at 4096 and 8192, 86.3% at 2048, and 104.7% at 1024.
3. At 777 x 1111 x 333, Triton (18714.9 GFLOP/s) is 1.42x faster than `cublasSgemm` and 1.33x faster than `torch.matmul`.
4. At 4097, Triton drops to 78.9% of cuBLAS and the compiled kernel uses 255 registers with 6 bytes of spill, against 220 registers and no spill for the same configuration at 4096. The same happens at 1000 (255 registers, 6 B spill).
5. The autotuner picked the fastest configuration it timed at every shape (rank 1 of 12). The spread between best and worst configuration is 1.30x at 2048 and 4.6x at 1000.
6. The Triton matmul uses 2.3x the registers and 8x the shared memory of v6 and fits one 256-thread block per SM by registers, against two for v6.
7. The Triton SASS moves A and B from global to shared memory with `LDGSTS.E.BYPASS.128` (cp.async, 128-bit, L1 bypass) and reads shared memory with 176 `LDS.128`. v6 uses synchronous `LDG.E.128` into registers followed by `STS`.
8. The Triton inner loop is unrolled to 2048 FFMA in the binary against 64 in v6.
9. Triton vector add at 2^26 (1512.4 GB/s flushed) is within 1.1% of the Stage 0 CUDA vector add (1528.5 GB/s).
10. Row-softmax autotuning chose 1 warp at 1024 columns, 4 warps at 4096 and 16384, and 32 warps at 3000.

## Interpretation TODO(Nirak)

- Why Triton's larger register and shared-memory footprint (one block per SM) still beats v6 at two blocks per SM.
- What the cp.async pipeline (num_stages = 3) contributes versus the register tile shape; Stage 10 compares these directly.
- Why cuBLAS loses to Triton at 777 x 1111 x 333 and at 1024.
- Whether the 255-register, spilling compile at 4097 and 1000 explains the lower fraction of cuBLAS there.

## Open questions

The open items concern the two cuBLAS paths, register growth at non-divisible shapes, and softmax at non-power-of-two rows; counter-based checks are pending counter access.

1. At 1024 Triton is 4.7% faster than `cublasSgemm` from `build/sgemm` but equal to `torch.matmul` (38836.1). The two cuBLAS paths differ by 4.8% at 1024 and by less than 1.2% at 4096 and 8192. Possible different cuBLAS kernel selection through PyTorch; not checked with Nsight Systems.
2. The 4097 and 1000 compiles use 255 registers with spill while the same configuration at 4096 uses 220. Triton specializes on argument divisibility by 16; the non-divisible shapes may take a different masking path. Not confirmed.
3. Softmax at 3000 columns runs at 2621.3 GB/s against 4369.1 to 5461.3 GB/s for power-of-two rows with warm L2; the padded BLOCK = 4096 masks 27% of each row. Stage 8 measures this with L2 flushed.
4. `matmul_autotune.csv` records autotuner timings without the GPU, job, and date columns that the other result files carry; the job is 24047985.
5. Counter-based checks (achieved occupancy, shared-memory bank conflicts, stall reasons for Triton vs v6) are pending counter access; see `docs/progress.md`.
