# Stages 4 to 6: roofline and resource analysis

## Question

Where does each Stage 0 to 3 kernel sit relative to the memory and compute ceilings of this GPU, what clock does the GPU actually run at under load, and do occupancy, the reduction tree phase, or problem size explain the Stage 2 and Stage 3 surprises?

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 188 SMs, 300 W board power (datasheet), `high` partition |
| Toolchain / build | CUDA 13.3, Release, sm_120 |
| Clock logging | `scripts/clock_runs_stage4.sh`, job 24039707: `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv,nounits -lms 100` in the background, runs bracketed by timestamp markers; `scripts/clock_summary.py` keeps samples with power at least 50% of the run maximum (the kernel phase) |
| Roofline | `scripts/roofline.py` over the Stage 0 to 3 CSVs; output `results/stage4/roofline.csv`, `roofline.png` |
| Reduction tree phase | `build/reduce --first-pass` (first pass only) plus a load-only baseline kernel (`--version 0`), `scripts/reduce_tree_stage4.sh`, job 24039791 |
| Naive SGEMM sweep | `scripts/naive_sweep_stage4.sh`, v1 and v2 at every M = N = K from 1016 to 1032 and 4088 to 4104, job 24039792 |
| Timing | 10 warm-up, at least 100 timed reps (1000 or 200 for SGEMM and 3000 for streaming in the clock-logged runs so each run spans many 100 ms samples), median |

### FP32 peak

Peak FP32 = SMs x FP32 lanes per SM x 2 (FMA) x SM clock.

- SMs: 188, from `cudaGetDeviceProperties` (`src/device_info.cu`).
- FP32 lanes per SM: 128. Source: the NVIDIA RTX PRO 6000 Blackwell Max-Q datasheet lists 24,064 CUDA cores; 24,064 / 188 = 128. The CUDA Programming Guide pages fetched on 2026-09-26 did not include the per-SM arithmetic throughput table for compute capability 12.0, so the datasheet core count is the source.
- At the maximum SM clock reported by `nvidia-smi` (3090 MHz): 188 x 128 x 2 x 3.090 GHz = 148.7 TFLOPS.
- The datasheet single-precision figure is 110 TFLOPS, which corresponds to 110e12 / (2 x 24064) = 2.29 GHz.
- Under load the SM clock was 1515 to 2332 MHz (median per run, below), so the peak at the observed clock was 72.9 to 112.2 TFLOPS.

## Results

### Clocks, power, temperature (job 24039707)

Median over active samples; FP32 peak at that run's median SM clock. Idle before the runs: 180 MHz SM, 12.8 W.

| Run | Active samples | SM MHz median (min to max) | Memory MHz | Power W median (max) | Temp C median (max) | FP32 peak at median clock (GFLOP/s) |
|---|---|---|---|---|---|---|
| cuBLAS 4096 | 23 | 1537 (1537 to 1567) | 13365 | 300.0 (300.07) | 49 (50) | 73972.7 |
| cuBLAS 8192 | 50 | 1515 (1507 to 1995) | 13365 | 300.0 (300.16) | 52 (54) | 72913.9 |
| v4 4096 | 74 | 1942 (1507 to 2332) | 13365 | 300.0 (301.18) | 56 (58) | 93464.6 |
| v4 8192 | 137 | 2025 (1935 to 2332) | 13365 | 300.0 (300.1) | 61 (64) | 97459.2 |
| v5 4096 | 63 | 2160 (2017 to 2317) | 13365 | 300.0 (305.91) | 65 (66) | 103956.5 |
| v5 8192 | 122 | 2160 (2152 to 2332) | 13365 | 300.0 (300.06) | 66 (68) | 103956.5 |
| v6 4096 | 39 | 1732 (1732 to 2317) | 13365 | 300.0 (307.99) | 68 (69) | 83357.7 |
| v6 8192 | 67 | 1642 (1635 to 1972) | 13365 | 300.0 (300.43) | 68 (69) | 79026.2 |
| vector_add, 2 GiB | 48 | 2190 (1822 to 2332) | 13365 | 300.0 (304.27) | 65 (65) | 105400.3 |
| copy, 2 GiB | 45 | 2332 (2190 to 2332) | 13365 | 299.5 (300.12) | 65 (66) | 112234.5 |
| reduce v5, 2^28 | 23 | 1822 (1815 to 2332) | 13365 | 300.0 (304.01) | 65 (66) | 87689.2 |

SGEMM throughput in the same job, against the peak at the maximum clock and at the run's observed clock:

| Run | Median ms | GFLOP/s | % of 148.7 TFLOPS (3090 MHz) | % of peak at observed clock |
|---|---|---|---|---|
| cuBLAS 4096 | 2.693952 | 51017.6 | 34.3 | 69.0 |
| cuBLAS 8192 | 21.112865 | 52077.8 | 35.0 | 71.4 |
| v4 4096 | 7.186368 | 19125.0 | 12.9 | 20.5 |
| v4 8192 | 62.586800 | 17567.8 | 11.8 | 18.0 |
| v5 4096 | 6.056480 | 22692.9 | 15.3 | 21.8 |
| v5 8192 | 55.417633 | 19840.5 | 13.3 | 19.1 |
| v6 4096 | 3.797248 | 36194.4 | 24.3 | 43.4 |
| v6 8192 | 29.304367 | 37520.4 | 25.2 | 47.5 |

Streaming in the same job (warm, 3000 reps): vector_add 2 GiB 1517.4 GB/s, copy 2 GiB 1494.5 GB/s, reduce v5 2^28 1557.9 GB/s.

### Arithmetic intensity and roofline

FLOPs are FP32 operations the algorithm needs (FMA = 2); bytes are compulsory DRAM bytes (each input read once, each output written once), as in the stage CSVs. Roof = min(148.7 TFLOPS, AI x 1530 GB/s). From `results/stage4/roofline.csv`:

| Stage | Kernel | Configuration | FLOPs | Bytes | AI (FLOP/B) | GFLOP/s | GB/s | Roof (GFLOP/s) | Bound | % of roof |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | vector_add | n = 67108864, flushed | 67108864 | 805306368 | 0.0833 | 127.0 | 1524.1 | 127.5 | memory | 99.6 |
| 0 | saxpy | n = 67108864, flushed | 134217728 | 805306368 | 0.1667 | 256.0 | 1536.0 | 255.0 | memory | 100.4 |
| 0 | copy_kernel | n = 67108864, flushed | 0 | 536870912 | 0 | 0 | 1498.0 | | memory | |
| 0 | memcpy_d2d | n = 67108864, flushed | 0 | 536870912 | 0 | 0 | 1481.0 | | memory | |
| 1 | contiguous | 2^28, flushed | 0 | 2147483648 | 0 | 0 | 1493.7 | | memory | |
| 1 | strided | stride 8, flushed | 0 | 2147483648 | 0 | 0 | 352.2 | | memory | |
| 1 | gather | random, flushed | 0 | 3221225472 | 0 | 0 | 180.5 | | memory | |
| 1 | aos_x | 2^26 structs, flushed | 0 | 536870912 | 0 | 0 | 618.3 | | memory | |
| 1 | aos_sum | 2^26 structs, flushed | 0 | 1342177280 | 0 | 0 | 1549.3 | | memory | |
| 2 | v1_atomic | 2^28, flushed | 268435455 | 1073741824 | 0.25 | 0.7 | 2.8 | 382.5 | memory | 0.2 |
| 2 | v2_interleaved | 2^28, flushed | 268435455 | 1073741824 | 0.25 | 222.9 | 891.6 | 382.5 | memory | 58.3 |
| 2 | v3_sequential | 2^28, flushed | 268435455 | 1073741824 | 0.25 | 224.1 | 896.2 | 382.5 | memory | 58.6 |
| 2 | v4_first_add | 2^28, flushed | 268435455 | 1073741824 | 0.25 | 374.5 | 1498.0 | 382.5 | memory | 97.9 |
| 2 | v5_warp_shuffle | 2^28, flushed | 268435455 | 1073741824 | 0.25 | 382.1 | 1528.5 | 382.5 | memory | 99.9 |
| 2 | v6_grid_stride | 2^28, flushed | 268435455 | 1073741824 | 0.25 | 381.0 | 1524.0 | 382.5 | memory | 99.6 |
| 2 | v7_cub | 2^28, flushed | 268435455 | 1073741824 | 0.25 | 384.4 | 1537.5 | 382.5 | memory | 100.5 |
| 3 | cublas | 8192, warm | 1.0995e12 | 805306368 | 1365.3 | 51188.3 | 37.5 | 148715.5 | compute | 34.4 |
| 3 | v1_naive | 8192, warm | 1.0995e12 | 805306368 | 1365.3 | 832.0 | 0.6 | 148715.5 | compute | 0.6 |
| 3 | v2_coalesced | 8192, warm | 1.0995e12 | 805306368 | 1365.3 | 4253.6 | 3.1 | 148715.5 | compute | 2.9 |
| 3 | v3_smem_tiling | 8192, warm | 1.0995e12 | 805306368 | 1365.3 | 6944.6 | 5.1 | 148715.5 | compute | 4.7 |
| 3 | v4_1d_regblock | 8192, warm | 1.0995e12 | 805306368 | 1365.3 | 17342.8 | 12.7 | 148715.5 | compute | 11.7 |
| 3 | v5_2d_regblock | 8192, warm | 1.0995e12 | 805306368 | 1365.3 | 19573.3 | 14.3 | 148715.5 | compute | 13.2 |
| 3 | v6_vectorized | 8192, warm | 1.0995e12 | 805306368 | 1365.3 | 37066.3 | 27.1 | 148715.5 | compute | 24.9 |

The ridge point (148.7 TFLOPS / 1530 GB/s) is at AI = 97.2 FLOP/B. Plot: `results/stage4/roofline.png` (kernels with zero FP32 FLOPs are in the table only).

### Occupancy versus performance (Stage 3 tile sweep, job 24038802)

Registers per thread and static shared memory per block from `cudaFuncGetAttributes`; blocks per SM and theoretical occupancy from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. GFLOP/s at M = N = K = 4096, sorted by throughput.

| Version | Configuration | Threads/block | Regs/thread | Static smem/block (B) | Blocks/SM | Theo. occupancy | GFLOP/s at 4096 |
|---|---|---|---|---|---|---|---|
| v5 | 128x128x16x8x8 | 256 | 97 | 16384 | 2 | 0.333 | 30351.3 |
| v5 | 64x64x8x4x4 | 256 | 44 | 4096 | 5 | 0.833 | 24368.7 |
| v5 | 64x64x8x8x8 | 64 | 96 | 4096 | 10 | 0.417 | 23350.3 |
| v5 | 128x128x8x8x8 | 256 | 89 | 8192 | 2 | 0.333 | 23269.4 |
| v5 | 128x64x8x8x8 | 128 | 95 | 6144 | 5 | 0.417 | 23111.7 |
| v4 | 64x64x16x8 | 512 | 40 | 8192 | 3 | 1.000 | 20062.7 |
| v4 | 64x64x8x8 | 512 | 48 | 4096 | 2 | 0.667 | 19309.7 |
| v4 | 64x64x8x16 | 256 | 40 | 4096 | 6 | 1.000 | 16897.5 |
| v4 | 128x64x8x8 | 1024 | 64 | 6144 | 1 | 0.667 | 16652.2 |
| v4 | 32x32x8x4 | 256 | 40 | 2048 | 6 | 1.000 | 15048.8 |
| v3 | T = 32 | 1024 | 38 | 8192 | 1 | 0.667 | 8311.1 |
| v3 | T = 16 | 256 | 40 | 2048 | 6 | 1.000 | 8269.2 |
| v3 | T = 8 | 64 | 40 | 512 | 24 | 1.000 | 5777.7 |

Default (main-run) configurations add v6 128x128x8x8x8 (256 threads, 94 regs, 8192 B, 2 blocks/SM, 0.333, 36845.6 GFLOP/s at 4096) and v1/v2 (1024 threads, 31/32 regs, 1 block/SM, 0.667).

### Reduction tree phase (job 24039791)

First pass only (one kernel launch that produces per-block partial sums; checked exactly per block). v0 is the same load into shared memory and barrier without the tree. Median time per launch in us.

| n | Mode | v0 load only | v2 interleaved | v3 sequential | v4 first add | v5 warp shuffle |
|---|---|---|---|---|---|---|
| 2^20 (4 MiB) | warm, 100 launches per rep | 4.108 | 6.173 | 6.173 | 4.125 | 4.106 |
| 2^22 (16 MiB) | warm, 100 launches per rep | 10.274 | 20.500 | 20.501 | 12.304 | 8.208 |
| 2^26 (256 MiB) | L2 flushed | 221.184 | 309.248 | 309.232 | 186.368 | 180.224 |
| 2^28 (1 GiB) | L2 flushed | 724.992 | 1169.408 | 1167.360 | 700.416 | 694.272 |

Plot: `results/stage4/reduce_tree.png`.

### Naive SGEMM around powers of two (job 24039792)

GFLOP/s, square M = N = K (`results/stage4/naive_sweep.csv`, plot `results/stage4/naive_sweep.png`):

| Size | 1016 | 1020 | 1022 | 1023 | 1024 | 1025 | 1026 | 1028 | 1032 |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 2223.6 | 2256.3 | 2478.4 | 2498.7 | 765.0 | 2506.3 | 2479.3 | 2282.2 | 2282.2 |
| v2 | 5613.0 | 5647.4 | 5675.8 | 5713.2 | 5682.1 | 5806.2 | 5687.3 | 5695.2 | 5756.4 |

| Size | 4088 | 4092 | 4094 | 4095 | 4096 | 4097 | 4098 | 4100 | 4104 |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 2496.7 | 2513.0 | 2742.4 | 2767.5 | 839.1 | 2771.7 | 2730.8 | 2514.0 | 2496.9 |
| v2 | 5737.6 | 5643.5 | 5635.6 | 5617.2 | 5722.1 | 5636.8 | 5601.0 | 5589.5 | 5588.4 |

## Observations

- Every clock-logged run held board power at 300 W (median 299.5 to 300.0 W). The median SM clock under load was 1515 to 2332 MHz, 49.0% to 75.5% of the 3090 MHz maximum. The memory clock was 13365 MHz in every run, below the 14001 MHz maximum reported by `nvidia-smi`; at 13365 MHz the nominal bandwidth formula gives 2 x 13365 MHz x 64 B = 1710.7 GB/s.
- cuBLAS ran at the lowest SM clock of all runs (1515 MHz at 8192) and reached 71.4% of the FP32 peak at that clock; v6 ran at 1642 MHz and reached 47.5%; v5 ran at 2160 MHz and reached 19.1%. At 8192, the two highest-throughput SGEMMs (cuBLAS and v6) ran at the two lowest SM clocks (1515 and 1642 MHz); v4 and v5 ran at 2025 and 2160 MHz.
- GPU temperature rose from 49 C to 69 C over the 92 seconds of logged runs (markers 11:50:21 to 11:51:53).
- Against the roofline with 1530 GB/s and 148.7 TFLOPS: vector_add, saxpy, reduction v5 to v7 reach 99.6% to 100.5% of their memory roof; reduction v2 and v3 reach 58.3% and 58.6%. The best hand-written SGEMM (v6) reaches 24.9% of the compute roof at 3090 MHz and cuBLAS 34.4%.
- All Stage 0 to 2 kernels have AI at most 0.25 FLOP/B, below the ridge point of 97.2 FLOP/B; SGEMM at 8192 has AI 1365.3 FLOP/B, above it.
- In the tile sweep, the fastest configuration has the lowest theoretical occupancy (0.333) and the most registers (97 per thread). The five configurations with theoretical occupancy 1.0 are 5777.7 to 20062.7 GFLOP/s.
- Reduction tree phase: v2 and v3 first passes take the same time in every mode, including L2-resident batched launches (20.500 vs 20.501 us at 2^22). The tree roughly doubles the time over the load-only baseline in the L2-resident case (10.274 to 20.500 us at 2^22) and adds 40% to 61% in the DRAM cases (221.184 to 309.248 us at 2^26, 724.992 to 1169.408 us at 2^28). v4 and v5 first passes (half as many blocks, two loads per thread) are faster than the load-only baseline at 2^26 and 2^28; at 2^22, v5 is faster (8.208 us) and v4 slower (12.304 us) than the baseline (10.274 us).
- Naive SGEMM: v1 is 2.9x to 3.3x slower at exactly 1024 and 4096 than at every other size tested from 1016 to 1032 and 4088 to 4104, including other multiples of 8 and 16 (1016, 1032, 4088, 4104). Near the power of two (1022 to 1026 and 4094 to 4098, excluding 1024 and 4096), v1 reaches 2478.4 to 2506.3 and 2730.8 to 2771.7 GFLOP/s, against 2223.6 to 2346.7 at 1016 to 1020 and 1028 to 1032, and 2496.7 to 2569.5 at 4088 to 4092 and 4100 to 4104. v2 varies by at most 3.5% within each range.

## Interpretation

TODO(Nirak)

## Open questions

- Is the 300 W cap the reason SM clocks differ by kernel, with FP32-dense kernels drawing more power per clock and so running slower? `power.draw` may be averaged over a window longer than 100 ms by the driver; `power.draw.instant` and the `clocks_event_reasons` fields would confirm power capping directly.
- With the memory clock at 13365 MHz under load, is 1710.7 GB/s the relevant nominal ceiling rather than 1792.1 GB/s? Against 1710.7 GB/s the Stage 0 achievable 1530 GB/s is 89.4% of nominal.
- Since v2 and v3 cost the same even when the tree phase is half of the kernel time, is the divergent `tid % (2*stride)` branch not a cost on this architecture, or does the compiler produce the same schedule? The SASS listings for both are in `results/stage2/sass_reduce.txt`; a side-by-side diff of the tree loop would answer the second part.
- Why is the load-only baseline slower than v4 and v5 at DRAM sizes? It launches twice as many blocks with one load per thread; do fewer loads in flight per thread limit it?
- Why does only an exact power of two slow v1? A warp in v1 reads A from 32 rows spaced K floats apart; at K = 1024 or 4096 the row spacing is 4 or 16 KiB. A padded-leading-dimension test (K = 4096 data with a row stride of 4097) would separate the stride from the size.
- Can occupancy be traded for registers further in v6 (for example BK = 16, as in the fastest v5 configuration)?
