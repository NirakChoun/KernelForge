# Stage 2: parallel reduction

## Question

How close does each classic reduction optimization bring a float sum to the achievable DRAM bandwidth (about 1530 GB/s), and to CUB `DeviceReduce::Sum`?

## Setup

Seven versions reduce up to 2^28 floats with L2 flushed; each must pass an exact integer test and a float tolerance test before it is timed.

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, hive-dc-7-5-58, `high` |
| Toolchain / build | CUDA 13.3 (CUB namespace `_V_300303`), Release, `-lineinfo`, sm_120 |
| Binary | `build/reduce` (`src/reduce.cu`), `--version 1..7`, run by `scripts/sweep_stage2.sh`, job 24038225 |
| Sizes | 2^20, 2^22, 2^24, 2^26, 2^28 floats; 1000003 and 16777233 (2^24 + 17, not multiples of the block size) |
| Block size | 256 threads |
| Timing | 10 warm-up, 100 reps, median, L2 flushed before each rep; one rep is the whole reduction to one value (all passes) |
| Bandwidth | input bytes (4n) / median time; partial sums of later passes are not counted |

Versions:

| Version | Description | Passes at 2^28 | First-pass grid at 2^28 |
|---|---|---|---|
| v1_atomic | one `atomicAdd` per element into one address; result zeroed with `cudaMemsetAsync` inside each timed rep | 1 | 1048576 |
| v2_interleaved | shared-memory tree, `if (tid % (2*stride) == 0)` | 4 | 1048576 |
| v3_sequential | shared-memory tree, `if (tid < stride)` | 4 | 1048576 |
| v4_first_add | v3, each thread adds two elements during the load (512 elements per block) | 4 | 524288 |
| v5_warp_shuffle | v4, last 64 values reduced by warp 0 with `__shfl_down_sync` | 4 | 524288 |
| v6_grid_stride | 1128 blocks (6 per SM x 188 SMs, from the occupancy API), grid-stride accumulation, shuffle block sum; a second 1-block launch sums the 1128 partials | 2 | 1128 |
| v7_cub | `cub::DeviceReduce::Sum`, temp storage allocated once outside timing | library | library |

Multi-pass versions (v2 to v5) relaunch the same kernel on the partial sums until one value remains, alternating between two scratch buffers.

Correctness, both required before timing:

1. Exact test: values in {-1, 0, +1}, with the nonzero probability set so that fewer than 2^24 elements are nonzero. Every partial sum in any order is an integer of magnitude below 2^24, so it is exact in float, and the result must equal the integer sum exactly.
2. Tolerance test: uniform [-1, 1). Reference is the sum in double. Tolerance is 2 * ceil(log2 n) * 2^-24 * sum|x_i|, which covers tree summation orders of depth log2 n with a factor 2 margin. The v1 order (sequential atomics in arrival order) has a worst-case bound proportional to n, not log2 n; it is held to the same tolerance and passes it at every size tested.

All 49 configurations passed both tests. Largest error as a fraction of the tolerance: v1 0.0322 (n = 1048576); v2 to v7 at most 1.05e-4.

Static analysis (`results/stage2/ptxas_reduce.txt`, `sass_ops_reduce.csv`, and `sass_reduce.txt` for the hand-written kernels): no spills in any kernel.

| Kernel | Registers | Static smem (B) | Theo. occupancy | SASS notes |
|---|---|---|---|---|
| v1 reduce_atomic | 8 | 0 | 1.0 | one `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU` per thread, no warp aggregation |
| v2 reduce_interleaved | 12 | 1024 | 1.0 | tree fully unrolled: 8 `FADD`, 9 `BAR.SYNC` |
| v3 reduce_sequential | 12 | 1024 | 1.0 | 9 `BAR.SYNC` |
| v4 reduce_first_add | 12 | 1024 | 1.0 | 9 `BAR.SYNC` |
| v5 reduce_warp_shuffle | 14 | 1024 | 1.0 | 3 `BAR.SYNC`, 5 `SHFL.DOWN` |
| v6 reduce_grid_stride | 18 | 32 | 1.0 | 1 `BAR.SYNC`, 15 `SHFL.DOWN` (static count) |
| v7 CUB DeviceReduceKernel | 40 | 44 | n/a | |

## Results

At 2^28, v5, v6, and CUB reach 99.6% to 100.5% of achievable bandwidth and v4 97.9%; the shared-memory trees v2 and v3 stop at 58.3% to 58.6%, and single-address atomics (v1) at 2.8 GB/s.

From `results/stage2/reduce.csv`. Median time in ms / effective GB/s.

| Version | 2^20 | 2^22 | 2^24 | 2^26 | 2^28 |
|---|---|---|---|---|---|
| v1_atomic | 1.521664 / 2.8 | 6.074368 / 2.8 | 24.270847 / 2.8 | 97.057793 / 2.8 | 388.167694 / 2.8 |
| v2_interleaved | 0.014336 / 292.6 | 0.030720 / 546.1 | 0.088064 / 762.0 | 0.321568 / 834.8 | 1.204224 / 891.6 |
| v3_sequential | 0.014336 / 292.6 | 0.030720 / 546.1 | 0.088064 / 762.0 | 0.321568 / 834.8 | 1.198080 / 896.2 |
| v4_first_add | 0.012288 / 341.3 | 0.020480 / 819.2 | 0.053248 / 1260.3 | 0.194560 / 1379.7 | 0.716800 / 1498.0 |
| v5_warp_shuffle | 0.012288 / 341.3 | 0.018432 / 910.2 | 0.051200 / 1310.7 | 0.184320 / 1456.4 | 0.702464 / 1528.5 |
| v6_grid_stride | 0.012288 / 341.3 | 0.020480 / 819.2 | 0.055296 / 1213.6 | 0.186400 / 1440.1 | 0.704544 / 1524.0 |
| v7_cub | 0.010240 / 409.6 | 0.020480 / 819.2 | 0.051200 / 1310.7 | 0.184320 / 1456.4 | 0.698368 / 1537.5 |

Sizes that are not multiples of the block size:

| Version | 1000003 | 16777233 |
|---|---|---|
| v1_atomic | 1.452032 / 2.8 | 24.266752 / 2.8 |
| v2_interleaved | 0.014336 / 279.0 | 0.090112 / 744.7 |
| v3_sequential | 0.014336 / 279.0 | 0.090112 / 744.7 |
| v4_first_add | 0.012288 / 325.5 | 0.055296 / 1213.6 |
| v5_warp_shuffle | 0.012288 / 325.5 | 0.053248 / 1260.3 |
| v6_grid_stride | 0.012288 / 325.5 | 0.055296 / 1213.6 |
| v7_cub | 0.010240 / 390.6 | 0.053248 / 1260.3 |

At 2^28, as a percentage of achievable bandwidth and of CUB:

| Version | GB/s | % of 1530 achievable | % of CUB |
|---|---|---|---|
| v1_atomic | 2.8 | 0.2 | 0.2 |
| v2_interleaved | 891.6 | 58.3 | 58.0 |
| v3_sequential | 896.2 | 58.6 | 58.3 |
| v4_first_add | 1498.0 | 97.9 | 97.4 |
| v5_warp_shuffle | 1528.5 | 99.9 | 99.4 |
| v6_grid_stride | 1524.0 | 99.6 | 99.1 |
| v7_cub | 1537.5 | 100.5 | 100.0 |

Every configuration at 2^20 and 1000003 except v1, and v5 at 2^22, has a per-launch median under 20 us (`launch_bound=1`).

Plots: `results/stage2/reduce_bandwidth.png`, `results/stage2/reduce_largest.png`.

## Observations

The large steps are v3 to v4 (1.67x) and v1 to any tree; v2 vs v3, v4 vs v5, and v5 vs v6 differ by a few percent or less at 2^28.

- v1 runs at 2.8 GB/s at every size, about 1.45 ns per element (388.167694 ms for 2^28 elements).
- v2 and v3 have identical medians at 2^20, 2^22, 2^24, 2^26, 1000003, and 16777233. At 2^28, v3 is 0.5% faster (1.198080 vs 1.204224 ms).
- v4 is 1.67x faster than v3 at 2^28 (0.716800 vs 1.198080 ms) and 1.65x at 2^26.
- v5 is 2.0% faster than v4 at 2^28 and 5.6% faster at 2^26.
- v6, with 1128 blocks and 2 launches, is within 0.3% of v5 at 2^28 and 1.1% at 2^26, and 8.0% slower than v5 at 2^24.
- CUB is the fastest at 2^28 (1537.5 GB/s, 0.5% above the 1530 GB/s achievable figure); v5 is within 0.6% of it. At 2^22 v5 has a lower median than CUB (0.018432 vs 0.020480 ms); both are multiples of 1.024 us, the timer step recorded in Stage 0.
- At 2^20 every tiled version and CUB take 10.2 to 14.3 us for 4 MiB of input.

## Interpretation

TODO(Nirak)

## Open questions

The counter-based tests below are pending counter access (commands in `docs/progress.md`).

- v2 (divergent `tid % (2*stride)`) and v3 (contiguous `tid < stride`) take the same time. Is the tree phase too small a share of the kernel to matter once the load is DRAM-bound, or does the compiler generate equivalent code for both? Test: compare the two SASS listings in `results/stage2/sass_reduce.txt`; ncu warp state statistics for both; a warm, L2-resident run where the tree phase is a larger share.
- Why does halving the number of blocks and adding during the load (v4) give 1.67x over v3? Is v3 limited by loads in flight per thread, by 9 barriers per 256 elements, or by the extra partial sums? Test: v3 with 512-thread blocks; nsys per-pass timing.
- v1 at 1.45 ns per atomic: is that the throughput limit of same-address float reductions at one L2 slice? Test: atomics spread over 2, 4, 32 addresses; warp-aggregated atomics.
- v6 is slower than v5 at 2^24 but equal at 2^28. Is one wave of 1128 blocks with a grid-stride loop less able to keep enough loads in flight at 64 MiB, or is the second launch a fixed cost? Test: v6 with 2 or 4 waves; nsys timing of each launch.
- How does CUB exceed 1530 GB/s? A reduction only reads, while the copy and vector add used to set 1530 GB/s also write. Test: compare against a pure read kernel.
