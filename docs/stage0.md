# Stage 0: harness and baselines

## Question

What bandwidth can simple streaming kernels reach on this GPU, how does it change with problem size, and where does the 128 MiB L2 stop hiding DRAM?

## Setup

Four streaming kernels were timed with CUDA events across sizes from 1 MiB to 2 GiB, warm and with L2 flushed. Every run passed a correctness check before timing.

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, CC 12.0, 188 SMs, 128 MiB L2 |
| Nominal DRAM bandwidth | 1792.1 GB/s (2 x 14001 MHz x 512 bit / 8) |
| Node / partition | hive-dc-7-5-58, `high`, `--gpus=6000_blackwell:1` |
| Toolchain | CUDA 13.3 (V13.3.33), driver 580.167.08 (reports CUDA 13.0), gcc 13.2.0, CMake 3.28.1 |
| Build | Release, `-lineinfo`, `CMAKE_CUDA_ARCHITECTURES=120` |
| Kernels | `vector_add` (c = a + b), `saxpy` (y = 1.5 x + y, in place), `copy_kernel` (dst = src), `cudaMemcpyAsync` device-to-device |
| Launch | 256 threads per block, one element per thread, grid = ceil(n / 256) |
| Timing | CUDA events, 10 warm-up, 100 timed reps, median reported |
| Modes | warm: no flush; cold: 256 MiB scratch write before each rep, outside the timed region; batched: 100 launches per event pair, time divided by 100 |
| Bytes counted | vector_add and saxpy 12 B per element, copy 8 B per element |
| Sweep | total data moved per launch 1 MiB to 2 GiB, powers of two and 1.5x points (`scripts/sweep_stage0.sh`, job 24037480) |
| Fixed sizes | n = 1048576, 16777216, 67108864, 1000003 (`scripts/fixed_stage0.sh`, job 24037481) |

Correctness checks:

- vector_add and copy: exact comparison against a CPU result (vector_add also allows a 1e-6 relative tolerance, which was not needed; max error 0).
- saxpy: double-precision reference, absolute tolerance 6e-7 (2 ulp of the largest possible result, 2.5); max error 1.19e-7.

The flushed vector_add result at 1 to 2 GiB (1527.2 to 1530.0 GB/s) sets the achievable bandwidth used in later stages: about 1530 GB/s.

## Results

With L2 flushed, the streaming kernels reach 81.9% to 85.3% of nominal at 2 GiB. Warm runs exceed nominal while the data fits in L2 and fall back to the cold values above 128 MiB.

Large sizes, from `results/stage0/sweep.csv` (2 GiB total data):

| Kernel | Warm GB/s | Cold GB/s | Cold % of nominal |
|---|---|---|---|
| vector_add | 1524.3 | 1528.5 | 85.3 |
| copy_kernel | 1494.7 | 1493.7 | 83.3 |
| memcpy_d2d | 1465.6 | 1468.6 | 81.9 |

Around the L2 capacity, warm (no flush), from `results/stage0/sweep.csv`:

| Total data (MiB) | vector_add GB/s | copy_kernel GB/s | memcpy_d2d GB/s |
|---|---|---|---|
| 64 | 4173.4 | 3054.8 | 4750.1 |
| 96 | 4545.8 | 3314.8 | 5528.5 |
| 128 | 3783.8 | 3378.4 | 5573.8 |
| 192 | 1999.5 | 1655.0 | 1594.0 |
| 256 | 1669.0 | 1507.1 | 1495.0 |
| 384 | 1539.7 | 1499.8 | 1482.3 |

Fixed sizes, from `results/stage0/fixed_sizes.csv` (median time in ms / GB/s):

| Kernel | n | Warm | Cold | Warm, 100 launches/rep |
|---|---|---|---|---|
| vector_add | 1048576 | 0.005600 / 2246.9 | 0.010240 / 1228.8 | 0.004121 / 3053.6 |
| vector_add | 1000003 | 0.005600 / 2142.9 | 0.010240 / 1171.9 | 0.004119 / 2913.6 |
| vector_add | 16777216 | 0.101120 / 1991.0 | 0.131072 / 1536.0 | |
| vector_add | 67108864 | 0.524304 / 1536.0 | 0.528384 / 1524.1 | |
| saxpy | 1048576 | 0.005600 / 2246.9 | 0.010240 / 1228.8 | 0.004120 / 3054.1 |
| saxpy | 1000003 | 0.005536 / 2167.6 | 0.010240 / 1171.9 | 0.004116 / 2915.1 |
| saxpy | 16777216 | 0.041984 / 4795.3 | 0.129024 / 1560.4 | |
| saxpy | 67108864 | 0.522384 / 1541.6 | 0.524288 / 1536.0 | |
| copy_kernel | 1048576 | 0.005616 / 1493.7 | 0.008192 / 1024.0 | 0.004117 / 2037.4 |
| copy_kernel | 1000003 | 0.005568 / 1436.8 | 0.008192 / 976.6 | 0.004103 / 1949.9 |
| copy_kernel | 16777216 | 0.039712 / 3379.8 | 0.088064 / 1524.1 | |
| copy_kernel | 67108864 | 0.357488 / 1501.8 | 0.358400 / 1498.0 | |
| memcpy_d2d | 1048576 | 0.003872 / 2166.5 | 0.008192 / 1024.0 | 0.004106 / 2043.1 |
| memcpy_d2d | 1000003 | 0.006944 / 1152.1 | 0.010240 / 781.3 | 0.006149 / 1301.0 |
| memcpy_d2d | 16777216 | 0.024800 / 5412.0 | 0.088064 / 1524.1 | |
| memcpy_d2d | 67108864 | 0.362816 / 1479.7 | 0.362496 / 1481.0 | |

Rows with a per-launch median under 20 us are marked `launch_bound=1` in the CSVs. In the sweep that is every cold point up to 24 MiB, and every warm point up to 48 MiB (copy_kernel), 64 MiB (vector_add), and 96 MiB (memcpy_d2d).

Nsight Systems kernel durations (`results/stage0/nsys_vector_add_*.csv`, job 24037643, warm, 111 launches):

| n | nsys median kernel duration (ns) | Event median in the same run (ms) |
|---|---|---|
| 1048576 | 3136 | 0.007280 |
| 67108864 | 522114 | 0.526656 |

Static analysis (`results/stage0/ptxas_*.txt`, `sass_ops_*.csv`, `sass_*.txt`):

| Kernel | Registers | Spills / local memory | Theoretical occupancy |
|---|---|---|---|
| vector_add | 12 | none | 1.0 (6 blocks of 256 threads per SM) |
| saxpy | 10 | none | 1.0 |
| copy_kernel | 8 | none | 1.0 |

Loads and stores are 32-bit (`LDG.E`, `LDG.E.CONSTANT`, `STG.E`); saxpy uses one `FFMA`.

Plots: `results/stage0/sweep_bandwidth.png`, `results/stage0/sweep_time.png`.

## Observations

Cold bandwidth plateaus from about 24 MiB; warm bandwidth sits above nominal up to 128 MiB and converges with cold from 384 MiB on.

- Cold bandwidth rises with size and levels off from about 24 MiB of total data. From 256 MiB to 2 GiB, cold results are 1486.6 to 1498.0 GB/s for copy_kernel, 1468.6 to 1481.0 GB/s for memcpy_d2d, and 1500.8 to 1530.0 GB/s for vector_add.
- Warm results exceed the 1792.1 GB/s nominal at every sweep point from 12 MiB to 128 MiB for all three kernels. The warm peak is 4545.8 GB/s (vector_add, 96 MiB), 3378.4 GB/s (copy_kernel, 128 MiB), and 5573.8 GB/s (memcpy_d2d, 128 MiB).
- Warm bandwidth drops between 128 MiB and 192 MiB of total data: vector_add 3783.8 to 1999.5 GB/s, copy_kernel 3378.4 to 1655.0, memcpy_d2d 5573.8 to 1594.0. From 384 MiB on, warm and cold differ by less than 3%.
- At 16777216 elements (192 MiB for vector_add), warm is 1991.0 GB/s and cold is 1536.0 GB/s. At 67108864 elements (768 MiB), warm and cold are 1536.0 and 1524.1 GB/s.
- memcpy_d2d is 1.7% below copy_kernel at 2 GiB cold (1468.6 vs 1493.7 GB/s).
- Of the 69 cold medians in the sweep, 61 are exact multiples of 1.024 us (for example 4.096, 6.144, 8.192, 131.072 us); the other 8 (4.128, 6.176, 10.224, 18.464 us) are within 0.032 us of a multiple. Warm medians do not show this pattern.
- From 8 MiB to 32 MiB, batched per-launch medians cluster at 4.101 to 4.122, 6.149 to 6.170, and 8.207 to 8.218 us, with one point at 10.373 us (copy_kernel, 32 MiB). Below 8 MiB they range from 1.772 to 4.102 us.
- At n = 1048576, nsys reports a 3136 ns median kernel duration; the event timer reports 5.600 us warm (7.280 us under nsys) and 4.121 us batched.

## Interpretation

TODO(Nirak)

## Open questions

The main unknowns are L2 residency in warm runs, timer resolution at small sizes, and why the three kernels plateau at different levels.

- Does L2 residency between reps explain warm results above nominal up to 128 MiB, and the drop between 128 and 192 MiB? Test: DRAM bytes per launch with ncu `--cache-control none` (pending counter access; commands in `docs/progress.md`).
- Why is cold timing close to multiples of 1.024 us? Is the event timestamp resolution different after an idle gap, or does the flush kernel change the clock state? This limits cold precision below about 20 us (one step is 5% at 20 us).
- Why are batched per-launch times on 2.05 us steps, and why is batched vector_add at 1 MiB (4.1 us) slower than the nsys kernel duration (3.1 us)? Is the batched number measuring kernel launch rate rather than kernel time? Not pursued (small-size launch-rate question, deferred by Nirak).
- Why does memcpy_d2d reach 5573.8 GB/s warm at 128 MiB, above vector_add and copy_kernel? Does the driver use a different copy path (kernel with wider accesses or copy engine)? An nsys trace of the memcpy would show which.
- Would 128-bit loads (`LDG.E.128`) change the cold plateau of copy_kernel (1494 GB/s) relative to vector_add (1528 GB/s)?
- Why does the plateau for copy (2 streams) sit below vector_add (3 streams)? Read/write ratio differs (1:1 vs 2:1).
