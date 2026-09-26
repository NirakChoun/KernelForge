# Stage 3: SGEMM

## Question

How much of cuBLAS FP32 GEMM throughput does each classic optimization step recover on this GPU, and how do tile and block sizes change the result?

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 188 SMs, hive-dc-7-5-58, `high` |
| Toolchain / build | CUDA 13.3, cuBLAS from the same toolkit (`CUDA::cublas`), Release, `-lineinfo`, sm_120 |
| Binary | `build/sgemm` (`src/sgemm.cu`), `--version 0..6` (0 = cuBLAS), `--cfg` for tile configurations |
| Problem | row-major FP32 C = A x B, A is M x K, B is K x N, alpha 1, beta 0; A, B uniform [-1, 1) |
| Sizes | square 256, 512, 1024, 2048, 4096, 8192; non-multiples 1000, 1023, 4097; rectangular M = 777, N = 1111, K = 333 |
| Runs | `scripts/sweep_stage3.sh` (job 24038801, all versions, default configurations); `scripts/tile_sweep_stage3.sh` (job 24038802, versions 3 to 5 at 1024, 2048, 4096) |
| Timing | 10 warm-up, 100 reps, median; warm (no L2 flush) |
| GFLOP/s | 2 M N K / median time |
| cuBLAS | `cublasSgemm` with `CUBLAS_DEFAULT_MATH` (FP32, no TF32); row-major product computed as column-major C^T = B^T A^T |

Versions (default configurations):

| Version | Description | Block | Default configuration |
|---|---|---|---|
| v1_naive | one thread per output, `threadIdx.x` selects the row | 32 x 32 | |
| v2_coalesced | one thread per output, `threadIdx.x` selects the column | 32 x 32 | |
| v3_smem_tiling | T x T output tile, A and B tiles in shared memory | T x T | T = 32 |
| v4_1d_regblock | BM x BN block tile, BK-deep shared tiles, TM outputs per thread in one column | BM BN / TM | 64 x 64 x 8 x 8 (512 threads) |
| v5_2d_regblock | TM x TN outputs per thread from TM + TN register values | (BM/TM)(BN/TN) | 128 x 128 x 8 x 8 x 8 (256 threads) |
| v6_vectorized | v5 with float4 global loads, A transposed in shared memory, float4 shared loads and C stores; scalar fallback when K or N is not a multiple of 4 | 256 | 128 x 128 x 8 x 8 x 8 |

All kernels guard every load and store, so any M, N, K works. The v6 float4 path ran at 256 to 8192 and 1000; the scalar fallback ran at 1023, 4097, and 777 x 1111 x 333.

Correctness, before timing:

- Every size: kernel vs cuBLAS.
- M N K <= 2^30: kernel vs a CPU loop in double, and cuBLAS vs the same CPU loop.
- Per element: |c - ref| <= 16 sqrt(K) 2^-24 (|A||B|)_ij, where |A||B| is the product of the element-wise absolute values (computed with cuBLAS, or in double on the CPU). This is far above the rounding error of either summation order and below the error from one wrong or missing product term for K <= 8192. It would also fail cuBLAS if TF32 were used (about 2^-11 relative error).

All 70 configurations of the main run passed (70 kernel-vs-cuBLAS checks, 42 kernel-vs-CPU, 42 cuBLAS-vs-CPU). Largest error across all checks: 3.94e-7 of (|A||B|)_ij, against a tolerance of 1.74e-5 at that K (333). A separate test job ran every version and configuration at 1, 17, 128, 1000, 1023, and 777 x 1111 x 333 before the commits; all 330 runs passed.

### cuBLAS baseline precision check

The cuBLAS baseline is pure FP32. Verified on 2026-09-26 in job 24039529 (same GPU type, `high`), output in `results/stage3/cublas_fp32_check.txt` and `results/stage3/nsys_cublas_*.csv`:

1. Math mode: `src/sgemm.cu` calls `cublasSetMathMode(h, CUBLAS_DEFAULT_MATH)` and `cublasSgemm` (FP32 inputs, FP32 compute). `src/cublas_fp32_check.cu` reads the mode back after `cublasCreate`: 0 (`CUBLAS_DEFAULT_MATH`). cuBLAS version 130501.
2. Environment: inside the job, `NVIDIA_TF32_OVERRIDE` and `CUBLAS_EMULATION_STRATEGY` are unset, and `env | grep -iE "tf32|cublas"` prints nothing. Slurm jobs inherit the submitting shell environment, which has neither variable.
3. Numerics: cuBLAS SGEMM compared with a cuBLAS DGEMM reference on the same inputs, max_ij |C - C_fp64| / (|A||B|)_ij:

| n | Default math (as in Stage 3) | TF32 tensor-op math forced |
|---|---|---|
| 1024 | 1.023e-07 | 5.110e-05 |
| 4096 | 9.819e-08 | 2.892e-05 |
| 8192 | 2.178e-07 | 2.154e-05 |

4. Kernel: Nsight Systems shows every cuBLAS launch at 4096 and 8192 in the Stage 3 configuration is `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`, a SIMT (CUDA core) FP32 kernel, not a tensor-op kernel. Median duration 2648198 ns at 4096 and 21563385 ns at 8192 under nsys.

The Stage 3 percentages of cuBLAS stand as reported.

Static analysis (`results/stage3/ptxas_sgemm.txt`, `sass_ops_sgemm.csv`, `sass_sgemm_v5_v6.txt`): no kernel spills or uses local memory. Static SASS instruction counts per kernel. Loop unrolling differs between kernels, so these are counts in the binary, not per-iteration or executed counts:

| Kernel | Regs | Static smem (B) | Theo. occupancy | Notes |
|---|---|---|---|---|
| v1 sgemm_naive | 31 | 0 | 0.667 | 1024 threads per block, 1 block per SM; 30 `LDG.E.CONSTANT`, 15 `FFMA` |
| v2 sgemm_coalesced | 32 | 0 | 0.667 | same instruction mix as v1 |
| v3 sgemm_smem<32> | 38 | 8192 | 0.667 | 32 `FFMA`, 32 `LDS`, 8 `LDS.128` |
| v4 sgemm_1d<64,64,8,8> | 48 | 4096 | 0.667 | 64 `FFMA`, 8 `LDS`, 16 `LDS.128` |
| v5 sgemm_2d<128,128,8,8,8> | 89 | 8192 | 0.333 | 64 `FFMA`, 8 scalar `LDS` (A column) + 2 `LDS.128` (B row); 64 scalar `STG.E` |
| v6 sgemm_vec<...,true> | 94 | 8192 | 0.333 | 64 `FFMA`, 4 `LDS.128`, 2 `LDG.E.128.CONSTANT`, 16 `STG.E.128` |
| v6 sgemm_vec<...,false> | 95 | 8192 | 0.333 | 8 scalar `LDG.E.CONSTANT`, 64 scalar `STG.E` |

## Results

Main run, `results/stage3/sgemm.csv`. GFLOP/s and percentage of cuBLAS at the same size.

| Version | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| cublas | 2818.8 | 13357.7 | 37005.2 | 52014.8 | 51082.8 | 51188.3 |
| v1_naive | 271.8 (9.6%) | 557.4 (4.2%) | 751.9 (2.0%) | 821.0 (1.6%) | 824.6 (1.6%) | 832.0 (1.6%) |
| v2_coalesced | 1691.3 (60.0%) | 3988.9 (29.9%) | 5602.2 (15.1%) | 6143.7 (11.8%) | 5768.3 (11.3%) | 4253.6 (8.3%) |
| v3_smem_tiling | 2407.8 (85.4%) | 5289.2 (39.6%) | 7640.8 (20.6%) | 8481.4 (16.3%) | 8197.5 (16.0%) | 6944.6 (13.6%) |
| v4_1d_regblock | 1108.4 (39.3%) | 4723.3 (35.4%) | 14284.6 (38.6%) | 20240.6 (38.9%) | 19141.1 (37.5%) | 17342.8 (33.9%) |
| v5_2d_regblock | 312.7 (11.1%) | 1334.2 (10.0%) | 5576.1 (15.1%) | 18485.1 (35.5%) | 23039.8 (45.1%) | 19573.3 (38.2%) |
| v6_vectorized | 718.2 (25.5%) | 3094.9 (23.2%) | 12908.0 (34.9%) | 33414.5 (64.2%) | 36845.6 (72.1%) | 37066.3 (72.4%) |

Median time (ms) at 4096 and 8192:

| Version | 4096 | 8192 |
|---|---|---|
| cublas | 2.690512 | 21.479729 |
| v1_naive | 166.666077 | 1321.596191 |
| v2_coalesced | 23.826464 | 258.487915 |
| v3_smem_tiling | 16.765953 | 158.326752 |
| v4_1d_regblock | 7.180288 | 63.398560 |
| v5_2d_regblock | 5.965280 | 56.174015 |
| v6_vectorized | 3.730128 | 29.663343 |

Sizes that are not multiples of the tile or of 4 (GFLOP/s, % of cuBLAS):

| Version | 1000 | 1023 | 4097 | 777 x 1111 x 333 |
|---|---|---|---|---|
| cublas | 33386.8 | 34633.8 | 46382.9 | 12670.2 |
| v1_naive | 1939.9 (5.8%) | 2459.6 (7.1%) | 2725.3 (5.9%) | 2023.0 (16.0%) |
| v2_coalesced | 5357.4 (16.0%) | 5645.7 (16.3%) | 5684.4 (12.3%) | 5317.8 (42.0%) |
| v3_smem_tiling | 7135.9 (21.4%) | 7548.8 (21.8%) | 8024.0 (17.3%) | 6718.9 (53.0%) |
| v4_1d_regblock | 13721.2 (41.1%) | 14171.9 (40.9%) | 18272.1 (39.4%) | 11313.8 (89.3%) |
| v5_2d_regblock | 5339.1 (16.0%) | 5555.4 (16.0%) | 22074.3 (47.6%) | 4175.8 (33.0%) |
| v6_vectorized | 12370.1 (37.1%) float4 | 12038.9 (34.8%) scalar | 33874.9 (73.0%) scalar | 8187.0 (64.6%) scalar |

Tile and block size sweep, `results/stage3/tile_sweep.csv` (GFLOP/s; regs per thread, theoretical occupancy from the occupancy API):

| Version | Configuration | Threads | Regs | Blocks/SM | Theo. occ. | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|
| v3 | T = 8 | 64 | 40 | 24 | 1.000 | 6347.8 | 6137.6 | 5777.7 |
| v3 | T = 16 | 256 | 40 | 6 | 1.000 | 8524.5 | 8909.8 | 8269.2 |
| v3 | T = 32 | 1024 | 38 | 1 | 0.667 | 7758.2 | 8618.3 | 8311.1 |
| v4 | 64x64x8x8 | 512 | 48 | 2 | 0.667 | 14532.0 | 20428.1 | 19309.7 |
| v4 | 32x32x8x4 | 256 | 40 | 6 | 1.000 | 13658.1 | 16733.0 | 15048.8 |
| v4 | 64x64x16x8 | 512 | 40 | 3 | 1.000 | 13777.2 | 20702.6 | 20062.7 |
| v4 | 128x64x8x8 | 1024 | 64 | 1 | 0.667 | 13317.9 | 18323.2 | 16652.2 |
| v4 | 64x64x8x16 | 256 | 40 | 6 | 1.000 | 9474.6 | 16331.7 | 16897.5 |
| v5 | 128x128x8x8x8 | 256 | 89 | 2 | 0.333 | 5662.5 | 18763.8 | 23269.4 |
| v5 | 64x64x8x8x8 | 64 | 96 | 10 | 0.417 | 8645.8 | 23290.6 | 23350.3 |
| v5 | 128x64x8x8x8 | 128 | 95 | 5 | 0.417 | 7277.4 | 21167.1 | 23111.7 |
| v5 | 64x64x8x4x4 | 256 | 44 | 5 | 0.833 | 12461.0 | 22248.6 | 24368.7 |
| v5 | 128x128x16x8x8 | 256 | 97 | 2 | 0.333 | 10259.7 | 26322.4 | 30351.3 |

Plots: `results/stage3/sgemm_gflops.png`, `sgemm_pct_cublas.png`, `tile_sweep.png`, `occupancy_vs_gflops.png`.

## Observations

- At 4096, each step changes the median time by: v1 to v2 7.0x faster, v2 to v3 1.42x, v3 to v4 2.33x, v4 to v5 1.20x, v5 to v6 1.60x. v6 is 72.1% of cuBLAS at 4096 and 72.4% at 8192.
- cuBLAS reaches 52014.8 GFLOP/s at 2048 and stays within 1.8% of that at 4096 and 8192.
- v1 is 3.3x faster at 4097 than at 4096 (2725.3 vs 824.6 GFLOP/s) and 3.3x faster at 1023 than at 1024 (2459.6 vs 751.9). v2 to v5 change by at most 4.5% between these neighbouring sizes. v6 (scalar fallback at 1023 and 4097) is 6.7% and 8.1% lower than at 1024 and 4096, and cuBLAS is 6.4% and 9.2% lower.
- From 4096 to 8192, throughput drops 26.3% for v2, 15.3% for v3, 15.0% for v5, and 9.4% for v4; it rises 0.6% for v6 and 0.2% for cuBLAS.
- v5 (default) is slower than v4 at 1024 (5576.1 vs 14284.6 GFLOP/s) and 2048 (18485.1 vs 20240.6) and faster from 4096. At 1024 the v5 default launches 64 blocks; the GPU has 188 SMs.
- v6 over v5 (default configurations): 2.31x at 1024, 1.81x at 2048, 1.60x at 4096, 1.89x at 8192.
- For the rectangular 777 x 1111 x 333 case, v4 reaches 89.3% of cuBLAS; cuBLAS itself runs at 12670.2 GFLOP/s there.
- Tile sweep at 4096: the fastest configuration is v5 128x128x16x8x8 (30351.3 GFLOP/s, 30.4% above the v5 default), which has the lowest theoretical occupancy in the sweep (0.333, 97 registers, 2 blocks per SM). The three v3 configurations have theoretical occupancy 0.667 to 1.0 and are the slowest (5777.7 to 8311.1).
- Within v4 at 4096, moving from BK = 8 to BK = 16 at 64x64 tiles raises throughput by 3.9% and reduces registers from 48 to 40.
- The v4 and v5 default configurations in the tile sweep job and the main job differ by 0.9% and 1.0%.

## Interpretation

TODO(Nirak)

## Open questions

- Why is v1 3.3x faster at 1023 and 4097 than at 1024 and 4096? In v1 the 32 threads of a warp read A from 32 rows K floats apart; with K a power of two those addresses share low-order bits. Test: v1 at K = 4096 with a padded row stride (lda = 4097) versus lda = 4096; DRAM and L2 sector counts (pending counter access).
- Why do v2 to v5 lose 9% to 26% from 4096 to 8192 while v6 and cuBLAS do not? Is it L2 reuse of A and B across concurrently running blocks (one row of B is 32 KiB at 8192)? Test: L2 hit rate by size (pending counter access), or a block-order swizzle.
- v5 defaults to 2 resident blocks per SM (89 registers x 256 threads). At 1024 only 64 blocks exist for 188 SMs. Is the small-size gap to v4 explained by blocks per SM rather than per-thread efficiency? Test: v5 at 1024 with 64x64 tiles (8645.8 GFLOP/s in the sweep, 1.5x the default).
- The fastest configuration has the lowest occupancy. What limits the higher-occupancy v4 and v3 configurations: shared memory bandwidth (loads per FFMA), or instruction issue? Test: ncu shared memory throughput and warp stall reasons (pending counter access); count LDS per FFMA from the SASS listings.
- v6 gains 1.6x to 2.3x over v5 while issuing the same 64 FFMA per inner loop body. How much of that comes from the float4 global loads, the transposed A in shared memory (4 `LDS.128` instead of 8 `LDS` + 2 `LDS.128`), and the float4 stores? Test: enable each change separately.
- Would v6 with the BK = 16 configuration that was fastest for v5 close more of the remaining 28% gap to cuBLAS?
