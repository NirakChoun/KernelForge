# Stage 1: memory access patterns

## Question

How much effective bandwidth does each global memory read pattern reach, relative to the Stage 0 achievable bandwidth (about 1530 GB/s, L2 flushed)?

## Setup

One binary times each read pattern at 1 GiB of input with L2 flushed; all patterns write their output contiguously and are checked exactly before timing.

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, hive-dc-7-5-58, `high` |
| Toolchain / build | CUDA 13.3, Release, `-lineinfo`, sm_120 |
| Binary | `build/mem_patterns` (`src/mem_patterns.cu`), run by `scripts/sweep_stage1.sh`, job 24037839 |
| Data | uint32 elements; outputs checked exactly against CPU-computed indices before timing |
| Size | 2^28 elements (1 GiB in, 1 GiB out) for contiguous, strided, gather, offset; 2^26 structs (1 GiB AoS) for AoS/SoA |
| Launch | 256 threads per block, one output element per thread |
| Timing | 10 warm-up, 100 reps, median, L2 flushed before each rep (contiguous also warm) |
| Effective bandwidth | useful bytes / median time; useful bytes listed per pattern below |

Patterns (all write `out[i]` contiguously):

| Pattern | Read | Useful bytes per element |
|---|---|---|
| contiguous | `in[i]` | 8 |
| strided, stride S | `in[(i mod (n/S)) * S + i / (n/S)]`: consecutive threads read S elements apart; every input element is read exactly once | 8 |
| gather | `in[idx[i]]`, `idx` a random permutation (mt19937_64 Fisher-Yates) | 12 (index, data, output) |
| offset O | `in[i + O]`, O = 0 to 32 | 8 |
| aos_x | field x of a 16-byte, 4-byte-aligned struct | 8 |
| soa_x | `x[i]` from a separate array | 8 |
| aos_sum | all four fields of the struct | 20 |
| soa_sum | four separate arrays | 20 |

Static analysis (`results/stage1/ptxas_mem_patterns.txt`, `sass_ops_mem_patterns.csv`, `sass_mem_patterns.txt`): 8 to 16 registers per kernel (soa_sum 16), no spills, theoretical occupancy 1.0 for every kernel. Every load is a 32-bit `LDG.E.CONSTANT`; aos_sum issues four of them per struct, aos_x one.

## Results

Contiguous, offset, and SoA reads stay at 97.1% to 97.7% of achievable bandwidth. Strided reads fall to 23.0% at stride 8, random gather to 11.8%, and reading one field of an AoS struct to 40.4%.

From `results/stage1/patterns.csv`. Percentages are of 1530 GB/s (achievable) and 1792.1 GB/s (nominal).

| Pattern | Median (ms) | Effective GB/s | % achievable | % nominal |
|---|---|---|---|---|
| contiguous (warm) | 1.437264 | 1494.1 | 97.7 | 83.4 |
| contiguous | 1.437696 | 1493.7 | 97.6 | 83.3 |
| strided, stride 1 | 1.439728 | 1491.6 | 97.5 | 83.2 |
| strided, stride 2 | 2.107392 | 1019.0 | 66.6 | 56.9 |
| strided, stride 4 | 3.458048 | 621.0 | 40.6 | 34.7 |
| strided, stride 8 | 6.096896 | 352.2 | 23.0 | 19.7 |
| strided, stride 16 | 6.033408 | 355.9 | 23.3 | 19.9 |
| strided, stride 32 | 6.072848 | 353.6 | 23.1 | 19.7 |
| strided, stride 64 | 8.587264 | 250.1 | 16.3 | 14.0 |
| gather, random permutation | 17.847296 | 180.5 | 11.8 | 10.1 |
| offset 0 | 1.439744 | 1491.6 | 97.5 | 83.2 |
| offset 1 to 31 (each) | 1.445888 | 1485.2 | 97.1 | 82.9 |
| offset 32 | 1.437696 | 1493.7 | 97.6 | 83.3 |
| aos_x | 0.868352 | 618.3 | 40.4 | 34.5 |
| soa_x | 0.360448 | 1489.5 | 97.4 | 83.1 |
| aos_sum | 0.866304 | 1549.3 | 101.3 | 86.5 |
| soa_sum | 0.866304 | 1549.3 | 101.3 | 86.5 |

### How effective bandwidth is counted

Effective bandwidth counts only the bytes the algorithm must move; a 32-byte sector model estimates the DRAM traffic each pattern actually causes.

Useful bytes are every 4-byte element the kernel reads and uses, plus every 4-byte element it writes. Bytes the hardware moves but the kernel does not use (the rest of a 32-byte sector, re-fetches) are not counted, so a pattern that wastes DRAM traffic shows a lower effective bandwidth. Per-pattern useful bytes are in the `bytes` column of `results/stage1/patterns.csv`.

The sector model (`scripts/sector_model.py`, output `results/stage1/sector_model.csv`) assumes DRAM transfers whole 32-byte sectors (8 elements); a sector is fetched once each time the pattern touches it, with no L2 reuse between strided passes (each pass spans the whole 1 GiB input, 8x the 128 MiB L2) and full reuse between neighbouring warps within a pass. Writes are contiguous in every pattern and count at their useful size. The model is an estimate from the access pattern, not a measurement.

| Pattern | Useful bytes | Model DRAM bytes | Model / useful | Effective GB/s (useful) | Model DRAM GB/s |
|---|---|---|---|---|---|
| contiguous | 2147483648 | 2147483648 | 1.000 | 1493.7 | 1493.7 |
| stride 1 | 2147483648 | 2147483648 | 1.000 | 1491.6 | 1491.6 |
| stride 2 | 2147483648 | 3221225472 | 1.500 | 1019.0 | 1528.5 |
| stride 4 | 2147483648 | 5368709120 | 2.500 | 621.0 | 1552.5 |
| stride 8 | 2147483648 | 9663676416 | 4.500 | 352.2 | 1585.0 |
| stride 16 | 2147483648 | 9663676416 | 4.500 | 355.9 | 1601.7 |
| stride 32 | 2147483648 | 9663676416 | 4.500 | 353.6 | 1591.3 |
| stride 64 | 2147483648 | 9663676416 | 4.500 | 250.1 | 1125.3 |
| gather | 3221225472 | 10737418240 | 3.333 | 180.5 | 601.6 |
| offset 1 to 31 | 2147483648 | 2147483680 | 1.000 | 1485.2 | 1485.2 |
| aos_x | 536870912 | 1342177280 | 2.500 | 618.3 | 1545.7 |
| soa_x | 536870912 | 536870912 | 1.000 | 1489.5 | 1489.5 |
| aos_sum / soa_sum | 1342177280 | 1342177280 | 1.000 | 1549.3 | 1549.3 |

Model details: strided with stride S reads the input in S passes; for S <= 8 each pass touches every sector (read bytes 4 S n), for S >= 8 every element is in its own sector (read bytes 32 n). Gather reads the index stream (4 n) plus one sector per random element (32 n). aos_x touches every sector of the 16-byte struct array (16 n).

Plots: `results/stage1/stride.png`, `results/stage1/offset.png`, `results/stage1/patterns.png`.

## Observations

Effective bandwidth falls with stride up to stride 8 and again at 64. Under the sector model, strides 2 to 32 and aos_x imply DRAM rates of 1528.5 to 1601.7 GB/s; stride 64 and gather imply much lower rates.

- Strided reads: effective bandwidth drops by 31.7% from stride 1 to 2 (1491.6 to 1019.0 GB/s), by 39.1% from 2 to 4, and by 43.3% from 4 to 8. Strides 8, 16, and 32 are within 1.1% of each other (352.2 to 355.9 GB/s). Stride 64 is 29.3% below stride 32 (250.1 GB/s) and has the largest spread (stddev 0.138 ms, min 8.180 ms vs median 8.587 ms).
- Median time at stride 2 is 1.46x stride 1, at stride 4 2.40x, and at strides 8 to 32 4.19x to 4.23x.
- Counting read over-fetch alone would predict 2x (stride 2) and 8x (stride 8) the stride-1 time; measured ratios are 1.46x and 4.23x. The write stream, half of the stride-1 traffic, does not grow with stride: with writes included, the sector model gives total DRAM traffic of 1.5x (stride 2) and 4.5x (stride 8). Measured times are 2.4% below the 1.5x model at stride 2 and 5.9% below the 4.5x model at stride 8, so the model DRAM rates are 1528.5 GB/s (stride 2) and 1585.0 GB/s (stride 8), against 1491.6 GB/s at stride 1.
- Under the sector model, strides 4 to 32 and aos_x imply DRAM rates of 1545.7 to 1601.7 GB/s, above the 1530 GB/s achievable figure; stride 64 implies 1125.3 GB/s and gather 601.6 GB/s.
- Random gather reaches 180.5 GB/s of useful traffic, 12.1% of contiguous.
- Every offset from 1 to 31 gives the same median, 1.445888 ms, 0.43% slower than offset 0 (1.439744 ms) and 0.57% slower than offset 32 (1.437696 ms). Both values are integer multiples of 1.024 us (1412 and 1404 steps), the timing pattern recorded in Stage 0.
- aos_x takes the same time as aos_sum (0.868352 vs 0.866304 ms) and 2.41x the time of soa_x. soa_sum and aos_sum have identical medians.
- aos_sum and soa_sum exceed the 1530 GB/s achievable figure by 1.3% (1549.3 GB/s); their read-to-write ratio is 4:1, compared with 1:1 for contiguous.
- Warm and cold contiguous results differ by 0.03% at 2 GiB.

## Interpretation

TODO(Nirak)

## Open questions

Each question below needs DRAM or sector counters to settle; those are pending counter access (commands in `docs/progress.md`).

- Does DRAM traffic per useful byte explain the stride curve? A 32-byte sector holds 8 elements. If every read fetched whole sectors that were not reused, reads at stride 2, 4, and 8+ would move 2x, 4x, and 8x the useful read bytes, and total traffic (reads plus the unchanged writes) would be 1.5x, 2.5x, and 4.5x contiguous. Measured time ratios are 1.46x, 2.40x, and 4.19x to 4.23x. Test: sectors per request and `dram__bytes_read.sum` for stride 1, 2, 4, 8, 64.
- Why is stride 64 slower than stride 32 when both should fetch one sector per element? Candidates to test: DRAM page or bank conflicts, TLB reach for a 256-byte stride across 1 GiB.
- Is gather limited by DRAM sector efficiency (one 4-byte element per 32-byte sector, 1/8) or by latency with 1 load in flight per thread? Test: gather with several independent loads per thread; ncu sectors per request.
- Why do all offsets 1 to 31 cost the same, including offsets that are multiples of 8 (32-byte sector aligned) and 16? Is the extra cost only the one additional 128-byte line per warp, independent of alignment within it?
- Why do the 4:1 read-to-write kernels (aos_sum, soa_sum) exceed the 1:1 copy bandwidth? Test with a read-only kernel and a write-only kernel to separate read and write DRAM efficiency.
- aos_x reads one field but takes the same time as reading all four. Test: DRAM bytes read for aos_x (expected about 4x the useful read bytes if whole sectors are fetched).
