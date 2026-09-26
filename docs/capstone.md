# Capstone: when does Triton match or beat hand-written CUDA, and why?

## Question

For FP32 matmul and row softmax on the RTX PRO 6000 Blackwell Max-Q, at which shapes does Triton's generated code match or beat the hand-written CUDA kernels of Stages 3 and 8 (and cuBLAS), and which differences in the generated code, resources, and chosen configuration account for the result?

Starting point (Stage 7, job 24047985): Triton matmul at 94.1% of cuBLAS at 4096, hand-written v6 at 73.6%.

## Method

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 188 SMs, `high`, job 24049608, 2026-09-26 |
| Software | PyTorch 2.14.0+cu130, Triton 3.8.0; CUDA kernels CUDA 13.3, Release, sm_120 |
| Run | `sbatch --mem=64G --time=01:30:00 scripts/run_gpu.sh scripts/stage10_capstone.sh` (driver `python/stage10_capstone.py`), 4 min 22 s |
| Matmul kernels | Triton `matmul_kernel` autotuned over the 12-entry `MATMUL_SPACE`; cuBLAS through `torch.matmul` (TF32 off); `build/sgemm` versions 0 (`cublasSgemm`), 6 (128x128x8x8x8, float4 when K and N are multiples of 4), 5 (128x128x16x8x8) |
| Matmul shapes | square 512, 1000, 1024, 1536, 2048, 3000, 4096, 4097, 6144, 8192; M x N x K = 777 x 1111 x 333, 1000 x 3000 x 2000, 4096 x 4096 x 1000 |
| Matmul ablation | Triton with fixed configurations at 4096 and 4097: 128x128x32 8 warps with num_stages 1 to 4; 128x128x16 8 warps with 1 and 3 stages; 128x128x32 4 warps; 64x64x32 4 warps; 128x64x32 4 warps 4 stages |
| Softmax | Triton `softmax_kernel` (num_warps autotuned) vs `build/libkf_ml.so` (Stage 8 CUDA), 2^26 FP32 elements, row lengths 128, 512, 1000, 1001, 1024, 2048, 3000, 4096, 4099, 6000, 8192, 12000, 16384 |
| Timing | before each timed run the kernel runs back to back for at least 0.5 s (for `build/sgemm`: warm-up count sized to 0.5 s from the Triton time), then 10 warm-up and 100 timed reps, median. Matmul warm; softmax L2 flushed before each rep |
| Clocks | `nvidia-smi` every 100 ms (SM clock, memory clock, power, temperature); start/end markers per run; `scripts/clock_summary.py` gives the median SM clock over samples with power >= 50% of the run maximum. `results/stage10/clock_summary.csv`, joined on `label` |
| Static analysis | registers, spills, shared memory per compiled kernel; resident blocks per SM from `cuOccupancyMaxActiveBlocksPerMultiprocessor` (Triton) and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (CUDA); static SASS opcode counts from `cuobjdump -sass` (`sass_summary.csv`, `sass_ops.csv`, `sass_ops_cuda_sgemm.csv`) |
| Clock-adjusted peak | 188 x 128 x 2 x median SM clock; "% of clock peak" = GFLOP/s / that peak |

Correctness: every Triton matmul (13 autotuned, 18 fixed) passed the Stage 3 criterion |c - ref| <= 16 sqrt(K) 2^-24 (|A||B|)_ij against FP64; every `build/sgemm` run passed kernel-vs-cuBLAS (and CPU where M N K <= 2^30); every softmax passed |y - ref| <= 1e-7 + 1e-5 |ref| against FP64. 120 checks, 0 failures.

## Results

### Matmul, autotuned Triton vs CUDA and cuBLAS

GFLOP/s, percentage of `cublasSgemm` (`build/sgemm`) in the same job, and median SM clock (MHz) of each run.

| Shape | cublasSgemm | torch.matmul | Triton | CUDA v6 | CUDA v5 |
|---|---|---|---|---|---|
| 512 | 13888.4 @2332 | 16384.0 (118.0) @2317 | 18724.6 (134.8) @2332 | 3133.6 (22.6) @2340 | 2389.2 (17.2) @2340 |
| 1000 | 31281.3 @1935 | 36169.0 (115.6) @1935 | 27889.3 (89.2) @2280 | 12547.7 (40.1) @2332 | 9694.4 (31.0) @2340 |
| 1024 | 34681.6 @1920 | 37449.1 (108.0) @1920 | 37428.3 (107.9) @2295 | 13132.9 (37.9) @2336 | 10294.3 (29.7) @2340 |
| 1536 | 37090.4 @1822 | 42114.6 (113.5) @1822 | 44787.9 (120.8) @2332 | 28933.6 (78.0) @2280 | 23154.0 (62.4) @2318 |
| 2048 | 47249.4 @1710 | 50835.2 (107.6) @1740 | 43457.3 (92.0) @2093 | 31808.9 (67.3) @2190 | 25769.6 (54.5) @2212 |
| 3000 | 47124.4 @1852 | 48422.5 (102.8) @1665 | 41260.7 (87.6) @2160 | 33292.6 (70.6) @2100 | 26890.5 (57.1) @2257 |
| 4096 | 48090.0 @1477 | 48734.7 (101.3) @1575 | 45652.3 (94.9) @2325 | 36510.2 (75.9) @2295 | 29753.4 (61.9) @1882 |
| 4097 | 43798.2 @1792 | 44446.1 (101.5) @1725 | 35627.3 (81.3) @2183 | 33680.3 (76.9) @2212 | 28728.2 (65.6) @2021 |
| 6144 | 46681.5 @1492 | 46931.7 (100.5) @1485 | 44925.5 (96.2) @1552 | 35280.6 (75.6) @1755 | 29852.0 (63.9) @1755 |
| 8192 | 48536.7 @1432 | 48612.0 (100.2) @1432 | 45232.1 (93.2) @1515 | 37654.7 (77.6) @1650 | 30164.6 (62.1) @1695 |
| 777 x 1111 x 333 | 13171.8 @2325 | 14036.2 (106.6) @2182 | 18714.9 (142.1) @2340 | 8287.1 (62.9) @2340 | 7051.1 (53.5) @2340 |
| 1000 x 3000 x 2000 | 38528.7 @1665 | 42144.3 (109.4) @1665 | 40676.9 (105.6) @2340 | 25020.0 (64.9) @2220 | 19457.8 (50.5) @2265 |
| 4096 x 4096 x 1000 | 44322.3 @2010 | 46676.0 (105.3) @1687 | 41162.6 (92.9) @2332 | 36801.8 (83.0) @1875 | 28289.0 (63.8) @2212 |

Same runs as % of the peak at each run's median clock:

| Shape | cublasSgemm | torch.matmul | Triton | CUDA v6 | CUDA v5 | Triton / v6 raw | Triton / v6 per clock | Triton / cublasSgemm per clock |
|---|---|---|---|---|---|---|---|---|
| 512 | 12.4 | 14.7 | 16.7 | 2.8 | 2.1 | 5.98 | 6.00 | 1.35 |
| 1000 | 33.6 | 38.8 | 25.4 | 11.2 | 8.6 | 2.22 | 2.27 | 0.76 |
| 1024 | 37.5 | 40.5 | 33.9 | 11.7 | 9.1 | 2.85 | 2.90 | 0.90 |
| 1536 | 42.3 | 48.0 | 39.9 | 26.4 | 20.8 | 1.55 | 1.51 | 0.94 |
| 2048 | 57.4 | 60.7 | 43.2 | 30.2 | 24.2 | 1.37 | 1.43 | 0.75 |
| 3000 | 52.9 | 60.4 | 39.7 | 32.9 | 24.8 | 1.24 | 1.20 | 0.75 |
| 4096 | 67.7 | 64.3 | 40.8 | 33.1 | 32.8 | 1.25 | 1.23 | 0.60 |
| 4097 | 50.8 | 53.5 | 33.9 | 31.6 | 29.5 | 1.06 | 1.07 | 0.67 |
| 6144 | 65.0 | 65.7 | 60.1 | 41.8 | 35.3 | 1.27 | 1.44 | 0.93 |
| 8192 | 70.4 | 70.5 | 62.0 | 47.4 | 37.0 | 1.20 | 1.31 | 0.88 |
| 777 x 1111 x 333 | 11.8 | 13.4 | 16.6 | 7.4 | 6.3 | 2.26 | 2.26 | 1.41 |
| 1000 x 3000 x 2000 | 48.1 | 52.6 | 36.1 | 23.4 | 17.8 | 1.63 | 1.54 | 0.75 |
| 4096 x 4096 x 1000 | 45.8 | 57.5 | 36.7 | 40.8 | 26.6 | 1.12 | 0.90 | 0.80 |

At 8192 every run was at the 300 W limit (median power 300.0 W, 30 to 46 active samples); at 6144 the runs were at 219.6 to 300.4 W; at the other shapes the runs had 5 to 49 active samples (most below 25) at 107 to 307 W median power.

### Grid size against SM count

Blocks per SM from the occupancy API; waves = blocks / (188 x blocks per SM). v6 always uses 128 x 128 tiles, 256 threads, 2 blocks per SM.

| Shape | Triton tile (BM x BN x BK, warps, stages) | Triton blocks | Triton blocks/SM | Triton waves | v6 blocks | v6 waves |
|---|---|---|---|---|---|---|
| 512 | 32 x 64 x 32, 4, 5 | 128 | 2 | 0.34 | 16 | 0.04 |
| 1000 | 128 x 64 x 64, 8, 3 | 128 | 1 | 0.68 | 64 | 0.17 |
| 1024 | 128 x 64 x 64, 8, 3 | 128 | 1 | 0.68 | 64 | 0.17 |
| 1536 | 128 x 128 x 32, 8, 3 | 144 | 1 | 0.77 | 144 | 0.38 |
| 2048 | 128 x 64 x 32, 4, 4 | 512 | 1 | 2.72 | 256 | 0.68 |
| 3000 | 64 x 256 x 32, 8, 3 | 564 | 1 | 3.00 | 576 | 1.53 |
| 4096 | 128 x 128 x 32, 8, 3 | 1024 | 1 | 5.45 | 1024 | 2.72 |
| 4097 | 128 x 128 x 32, 8, 3 | 1089 | 1 | 5.79 | 1089 | 2.90 |
| 6144 | 128 x 128 x 32, 8, 3 | 2304 | 1 | 12.26 | 2304 | 6.13 |
| 8192 | 128 x 128 x 32, 8, 3 | 4096 | 1 | 21.79 | 4096 | 10.89 |
| 777 x 1111 x 333 | 64 x 64 x 32, 4, 4 | 234 | 2 | 0.62 | 63 | 0.17 |
| 1000 x 3000 x 2000 | 128 x 64 x 64, 8, 3 | 376 | 1 | 2.00 | 192 | 0.51 |
| 4096 x 4096 x 1000 | 128 x 128 x 32, 8, 3 | 1024 | 1 | 5.45 | 1024 | 2.72 |

The autotuner chose its fastest timed configuration at all 13 shapes. Best-to-worst autotune time spread: 6.37x at 512, 4.65x at 1000, 1.28x at 2048, 1.58x at 8192 (`matmul_autotune.csv`).

### Generated code: resources and memory instruction widths

Static SASS counts (counts in the binary, not executed). LDGSTS = cp.async global-to-shared copy.

| Kernel | Regs | Spill (B) | Smem (B) | FFMA | Global loads | Shared loads | Global stores | BAR |
|---|---|---|---|---|---|---|---|---|
| Triton 4096, 6144, 8192, 1536 | 220 | 0 | 65536 | 2048 | 24 LDGSTS.128 | 176 LDS.128, 9 LDS | 16 STG.128 | 11 |
| Triton 1024 | 158 | 0 | 98304 | 2048 | 36 LDGSTS.128 | 200 LDS.128, 16 LDS | 8 STG.128 | 7 |
| Triton 1000 | 255 | 6 | 98304 | 2048 | 144 LDGSTS.32 | 192 LDS.128, 16 LDS | 32 STG.32 | 7 |
| Triton 4097 | 255 | 6 | 65536 | 2048 | 96 LDGSTS.32 | 160 LDS.128, 9 LDS | 64 STG.32 | 19 |
| Triton 3000 | 232 | 0 | 81920 | 2048 | 120 LDGSTS.32 | 160 LDS.128, 9 LDS | 64 STG.32 | 35 |
| Triton 4096 x 4096 x 1000 | 255 | 4 | 65536 | 2048 | 48 LDGSTS.32, 12 LDGSTS.128 | 176 LDS.128, 9 LDS | 16 STG.128 | 11 |
| Triton 1000 x 3000 x 2000 | 166 | 0 | 98304 | 2048 | 48 LDGSTS.32, 24 LDGSTS.128 | 192 LDS.128, 16 LDS | 32 STG.32 | 7 |
| CUDA v6 float4 path | 94 | 0 | 8192 | 64 | 2 LDG.128 | 4 LDS.128 | 16 STG.128 | 2 |
| CUDA v6 scalar path | 95 | 0 | 8192 | 64 | 8 LDG.32 | 4 LDS.128 | 64 STG.32 | 2 |
| CUDA v5 | 97 | 0 | 16384 | 64 | 14 LDG.32 | 2 LDS.128, 8 LDS | 64 STG.32 | 2 |

No Triton or CUDA matmul kernel contains an MMA opcode. Triton used 128-bit global accesses only where the dimension feeding that access is a multiple of 16 (1024, 1536, 2048, 4096, 6144, 8192; the K = 2000 operand of 1000 x 3000 x 2000; the N = 4096 operand of 4096 x 4096 x 1000) and 32-bit accesses where it is a multiple of 4 or 8 but not 16 (1000, 3000) or odd (4097). v6 takes its float4 path whenever K and N are multiples of 4 (1000 and 3000 included).

### Matmul ablation (fixed Triton configurations, `matmul_ablation.csv`)

| Config (BM x BN x BK, warps, stages) | 4096 GFLOP/s @MHz | % clock peak | Regs / spill / smem / blocks per SM | 4097 GFLOP/s @MHz | % clock peak | Regs / spill |
|---|---|---|---|---|---|---|
| 128x128x32, 8, 1 | 39945.8 @1860 | 44.6 | 224 / 0 / 32768 / 1 | 37602.0 @1935 | 40.4 | 254 / 0 |
| 128x128x32, 8, 2 | 42393.5 @2107 | 41.8 | 220 / 0 / 32768 / 1 | 35703.1 @1935 | 38.3 | 255 / 4 |
| 128x128x32, 8, 3 (autotuned choice) | 45528.4 @2205 | 42.9 | 220 / 0 / 65536 / 1 | 35551.9 @2340 | 31.6 | 255 / 6 |
| 128x128x32, 8, 4 | 45435.9 @2021 | 46.7 | 220 / 0 / 98304 / 1 | 35893.9 @2033 | 36.7 | 255 / 6 |
| 128x128x16, 8, 1 | 37412.6 @2340 | 33.2 | 196 / 0 / 16384 / 1 | 36281.7 @1972 | 38.2 | 230 / 0 |
| 128x128x16, 8, 3 | 43520.7 @2340 | 38.6 | 214 / 0 / 32768 / 1 | 36057.7 @2340 | 32.0 | 255 / 4 |
| 128x128x32, 4, 3 | 38271.4 @1995 | 39.9 | 255 / 42 / 65536 / 1 | 21360.7 @2205 | 20.1 | 255 / 116 |
| 64x64x32, 4, 3 | 39909.9 @1725 | 48.1 | 96 / 4 / 32768 / 3 | 31463.1 @1792 | 36.5 | 208 / 0 |
| 128x64x32, 4, 4 | 42655.6 @2197 | 40.3 | 238 / 0 / 73728 / 1 | 29907.8 @2224 | 27.9 | 255 / 4 |

With num_stages = 1 the SASS has no LDGSTS: global loads are LDG.128 (4096) or LDG.32 (4097) followed by STS.128. With 2, 3, 4 stages the LDGSTS count is 16, 24, 32 at 4096 and the shared memory 32768, 65536, 98304 B.

### Softmax, Triton vs CUDA (`softmax.csv`, L2 flushed)

| Cols | CUDA GB/s (% of 1530) | Triton GB/s (% of 1530) | CUDA threads / regs / blocks per SM / global access | Triton warps / regs / blocks per SM / global access |
|---|---|---|---|---|
| 128 | 1447.9 (94.6) | 1472.7 (96.3) | 32 / 35 / 24 / LDG.128 | 1 / 16 / 24 / LDG.128 |
| 512 | 1440.2 (94.1) | 1472.6 (96.2) | 32 / 35 / 24 / LDG.128 | 8 / 15 / 6 / LDG.64 |
| 1000 | 1424.6 (93.1) | 1460.2 (95.4) | 64 / 35 / 24 / LDG.128 | 16 / 17 / 3 / LDG.32 |
| 1001 | 1409.2 (92.1) | 1464.3 (95.7) | 64 / 24 / 24 / LDG.32 | 16 / 17 / 3 / LDG.32 |
| 1024 | 1432.4 (93.6) | 1472.6 (96.2) | 64 / 35 / 24 / LDG.128 | 16 / 15 / 3 / LDG.64 |
| 2048 | 1448.1 (94.6) | 1481.0 (96.8) | 128 / 35 / 12 / LDG.128 | 8 / 22 / 6 / LDG.128 |
| 3000 | 1440.2 (94.1) | 1464.3 (95.7) | 256 / 35 / 6 / LDG.128 | 8 / 48 / 5 / LDG.32 |
| 4096 | 1448.2 (94.7) | 1481.0 (96.8) | 256 / 35 / 6 / LDG.128 | 8 / 36 / 6 / LDG.128 |
| 4099 | 1432.2 (93.6) | 1464.4 (95.7) | 512 / 24 / 3 / LDG.32 | 8 / 79 / 3 / LDG.32 |
| 6000 | 1448.1 (94.6) | 1480.9 (96.8) | 512 / 35 / 3 / LDG.128 | 8 / 64 / 4 / LDG.128 |
| 8192 | 1448.2 (94.7) | 1481.0 (96.8) | 512 / 35 / 3 / LDG.128 | 2 / 167 / 6 / LDG.128 |
| 12000 | 1456.1 (95.2) | 1480.9 (96.8) | 1024 / 35 / 1 / LDG.128 | 32 / 34 / 1 / LDG.128 |
| 16384 | 1464.4 (95.7) | 1498.0 (97.9) | 1024 / 35 / 1 / LDG.128 | 4 / 164 / 3 / LDG.128 |

Memory clock was 13365 MHz in every softmax run. The CUDA kernel reads each row twice (statistic pass, output pass); the Triton kernel loads the row once into registers.

## Observations

1. Triton beats hand-written v6 at all 13 shapes in raw GFLOP/s (1.06x to 5.98x) and at 12 of 13 per clock; the exception per clock is 4096 x 4096 x 1000 (0.90).
2. Triton beats `cublasSgemm` in raw GFLOP/s at 512 (134.8%), 1024 (107.9%), 1536 (120.8%), 777 x 1111 x 333 (142.1%), and 1000 x 3000 x 2000 (105.6%). Per clock it is ahead only at 512 (1.35) and 777 x 1111 x 333 (1.41); at 1024, 1536, and 1000 x 3000 x 2000 the Triton runs had a median SM clock 375 to 675 MHz higher than the cuBLAS runs.
3. At the shapes where v6 is furthest behind (512, 1000, 1024, 777 x 1111 x 333) its 128 x 128 tiles give 16 to 64 blocks for 188 SMs (0.04 to 0.17 waves). Triton's chosen tiles give 128 to 234 blocks (0.34 to 0.68 waves).
4. At 8192, where all runs held 300 W and had 30 or more clock samples, per-clock efficiency is cuBLAS 70.4%, Triton 62.0%, v6 47.4%, v5 37.0%. Triton and v6 use the same 128 x 128 output tile and 256 threads there; Triton has BK = 32, 220 registers, 65536 B of shared memory, 1 block per SM, and cp.async (LDGSTS.128) double buffering; v6 has BK = 8, 94 registers, 8192 B, 2 blocks per SM, and synchronous LDG.128.
5. Non-multiple-of-16 dimensions change Triton's code more than v6's: at 1000, 3000, and 4097 Triton's global accesses are 32-bit, register use rises to 232 to 255 with up to 6 B of spill, and Triton falls from 107.9% (1024) to 89.2% (1000) and from 94.9% (4096) to 81.3% (4097) of `cublasSgemm`. v6 goes from 75.9% to 76.9% between 4096 and 4097 and keeps float4 at 1000 and 3000.
6. In the ablation at 4096, raw throughput rises from 39945.8 (1 stage) to 45528.4 GFLOP/s (3 stages), but the clock-adjusted values (44.6%, 41.8%, 42.9%, 46.7%) do not increase with stages; the four runs had median clocks from 1860 to 2205 MHz and minima of 1627 to 1740 MHz.
7. BK = 32 is ahead of BK = 16 at 4096 in both raw and clock-adjusted terms (3 stages: 45528.4 @2205 vs 43520.7 @2340, 42.9% vs 38.6%; 1 stage: 39945.8 @1860 vs 37412.6 @2340, 44.6% vs 33.2%).
8. 128 x 128 x 32 with 4 warps spills (42 B at 4096, 116 B at 4097) and is the slowest configuration at 4097 (21360.7 GFLOP/s).
9. Triton softmax is faster than the CUDA softmax at all 13 row lengths, by 1.7% to 3.9% (95.4% to 97.9% of 1530 GB/s vs 92.1% to 95.7%). At 1000 and 3000 columns Triton uses 32-bit loads and CUDA 128-bit loads, and Triton is still faster (1460.2 vs 1424.6; 1464.3 vs 1440.2 GB/s).
10. The Triton softmax autotuner chose between 1 and 32 warps with no monotonic trend in row length (for example 2 warps and 167 registers at 8192, 32 warps at 12000, 4 warps and 164 registers at 16384).

## Interpretation TODO(Nirak)

- Which mechanism explains Triton's lead over v6 at large sizes (Observation 4): BK and loop unrolling, the cp.async pipeline, the 8x larger shared footprint, or the different register tile layout, given that the ablation does not separate them beyond clock noise.
- Whether SM fill (Observation 3) is the whole explanation for the small-shape results, and why cuBLAS at 512 and 777 x 1111 x 333 is at 12% to 13% of clock peak.
- Why Triton's divisibility specialization costs more at 4097 than v6's scalar fallback costs v6 (Observation 5).
- For softmax: whether single-pass register residency or the CUDA kernel's second read explains the 1.7% to 3.9% gap.
- The answer to the Question in one paragraph: when Triton matches or beats hand-written CUDA, and why.

## Limitations

- Clocks: the card is power-limited at 300 W and the SM clock varied from 1425 to 2347 MHz across runs. Most runs below 8192 have 5 to 24 samples at 100 ms, and the median includes the 0.5 s sustain phase. The Python sustain loop synchronizes every 16 launches and `kfbench` inserts a spin kernel before each rep, while `build/sgemm` runs its reps back to back, so the duty cycle and power differ between Triton, `torch.matmul`, and `build/sgemm` runs. Per-clock values below 8192 carry that uncertainty; within-run min and max clocks are in `clock_summary.csv`.
- The ablation compares configurations one run each; differences under about 15% in clock-adjusted terms are within the min-to-max clock span of the runs.
- SASS counts are static. Executed instruction counts, achieved occupancy, stall reasons, bank conflicts, and DRAM bytes need Nsight Compute counters, which are denied on every Hive node probed.
- One GPU, one driver (580.167.08), one Triton version. The Triton search space is 12 configurations; a larger space could change the small-shape results.
- cuBLAS kernel names per shape were not recorded in this job; Stage 3 identified the SIMT kernel only at 4096 and 8192.

## Open questions

1. Per clock, `torch.matmul` is ahead of `cublasSgemm` from `build/sgemm` at most shapes (for example 57.5% vs 45.8% at 4096 x 4096 x 1000) although both are cuBLAS FP32. Different kernel selection or different clocks within the run; not checked with Nsight Systems.
2. The 64 x 64 x 32 configuration at 4096 has the highest clock-adjusted value in the ablation (48.1%, 3 blocks per SM) but the autotuner timed it slower. Autotuning is on raw time, which includes the clock at that moment.
3. At 4096 x 4096 x 1000 v6 is ahead of Triton per clock (40.8% vs 36.7%) while Triton uses 32-bit loads for the K = 1000 operand and 255 registers with 4 B of spill.
4. Triton softmax at 8192 and 16384 chose 2 and 4 warps with 164 to 167 registers per thread and still ran at 96.8% and 97.9% of 1530 GB/s.
5. Whether a num_stages or BK effect exists at 8192, where the clock is stable at 300 W; the ablation ran only at 4096 and 4097.
