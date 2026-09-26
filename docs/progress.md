# KernelForge progress

Running log of stage reports, performance counter access, and profiling that could not run. Stage details are in `docs/stageN.md`.

## Performance counter access

`ncu` needs access to GPU performance counters. On the primary node (hive-dc-7-5-58, `high`) it fails with `ERR_NVGPUCTRPERM` (job 24037044, ncu 2026.2.0.0). Probe: `scripts/ncu_probe.sh`, one job per node or GPU type on `low`, `ncu --metrics sm__cycles_elapsed.avg ./build-probe/vector_add 4096` with a multi-arch build (sm_80, 86, 89, 120). Type names from `sinfo -p low -o "%G|%D"` on 2026-09-26.

| GRES type | Node | Job | GPU reported | Counters |
|---|---|---|---|---|
| 6000_blackwell | hive-dc-7-4-58 | 24037338 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| 6000_blackwell | hive-dc-7-5-58 | 24037339 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| 6000_blackwell | hive-dc-7-5-62 | 24037340 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| 6000_blackwell | hive-dc-7-6-54 | 24037341 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| 6000_blackwell | hive-dc-7-7-58 | 24037342 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition | hive-as-11-2-34 | 24037343 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| nvidia_rtx_5000_ada_generation | (hive-as-11-2-54) | 24037344 | | cancelled before it ran |
| nvidia_l40s | hive-as-11-2-58 | 24037345 | NVIDIA L40S | DENIED |
| nvidia_a100-sxm4-80gb | hive-as-11-4-34 | 24037346 | NVIDIA A100-SXM4-80GB | DENIED |
| nvidia_a100-pcie-40gb | hive-as-11-4-54 | 24037347 | NVIDIA A100-PCIE-40GB | DENIED |
| nvidia_a100_80gb_pcie | hive-as-11-4-46 | 24037348 | NVIDIA A100 80GB PCIe | DENIED |
| a6000 | hive-dc-7-5-30 | 24037349 | NVIDIA RTX A6000 | DENIED |
| a100 | hive-dc-7-6-14 | 24037350 | NVIDIA A100 80GB PCIe | DENIED |

No probed node allows counter access so far. Jobs 24037338 and 24037343 ran later and were also denied. Probe 24037342 later ran on hive-dc-7-7-58 and was denied. Probe 24037344 was cancelled while pending on 2026-09-26 at the project owner's request; no further probes are submitted. Nsight Systems CUDA tracing works on hive-dc-7-5-58 (job 24037643) and is used for kernel durations.

## Pending profiling

Commands to run once a node with counter access is available. Replace the partition and GPU type with that node's. Each writes to the Slurm log only (no `.ncu-rep` kept).

Stage 0:

1. vector_add DRAM traffic with L2 kept warm (tests the L2 residency question):
   `sbatch -p low --gpus=6000_blackwell:1 --time=00:05:00 scripts/run_gpu.sh ncu -k regex:^vector_add$ --launch-skip 10 --launch-count 1 --cache-control none --metrics dram__bytes_read.sum,dram__bytes_write.sum ./build/vector_add 16777216 --no-csv`
2. Same launch with ncu's default cache flush, for comparison: the command above with `--cache-control all`.

Expected if there were no L2 reuse: 128 MiB read (134217728 B), 64 MiB write (67108864 B).

Stage 1 (sectors per request and DRAM bytes; `--section MemoryWorkloadAnalysis` as the brief asks, plus explicit metrics):

3. Contiguous: `sbatch -p low --gpus=6000_blackwell:1 --mem=16G --time=00:10:00 scripts/run_gpu.sh ncu -k regex:^contiguous$ --launch-skip 10 --launch-count 1 --section MemoryWorkloadAnalysis --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,dram__bytes_read.sum,dram__bytes_write.sum ./build/mem_patterns 268435456 --pattern contiguous --no-csv`
4. Strided, one run per stride S in 2, 4, 8, 64: the command above with `-k regex:^strided$` and `--pattern strided --stride S`.
5. Gather and aos_x: the command above with `-k regex:^gather$ ... --pattern gather` and `-k regex:^aos_x$ ... 67108864 --pattern aos_x`.

Stage 2 (warp state statistics and memory throughput for v2, v3, v5). Each run of a multi-pass version launches the kernel 4 times at 2^28 (passes), and the program runs it twice for correctness before the 10 warm-ups, so `--launch-skip 12` selects the first (full-size) pass of the second warm-up:

6. v2: `sbatch -p low --gpus=6000_blackwell:1 --mem=16G --time=00:10:00 scripts/run_gpu.sh ncu -k regex:^reduce_interleaved$ --launch-skip 12 --launch-count 1 --section SpeedOfLight --section WarpStateStats --section MemoryWorkloadAnalysis --section Occupancy ./build/reduce 268435456 --version 2 --no-csv`
7. v3: the command above with `-k regex:^reduce_sequential$` and `--version 3`.
8. v5: the command above with `-k regex:^reduce_warp_shuffle$` and `--version 5`.

Stage 3 (memory throughput, achieved occupancy, registers, shared memory, warp stall reasons for every version). The program runs two cuBLAS reference GEMMs, then one correctness launch and 10 warm-ups of the selected kernel, so with a name filter `--launch-skip 5` selects the fifth warm-up:

9. Versions 1 to 6, one run each with V = 1..6: `sbatch -p low --gpus=6000_blackwell:1 --mem=16G --time=00:20:00 scripts/run_gpu.sh ncu -k regex:^sgemm_ --launch-skip 5 --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats ./build/sgemm 4096 --version V --no-csv`
10. Tile sweep configurations (versions 3 to 5): the command above with `--version V --cfg C` for each configuration in `scripts/tile_sweep_stage3.sh`.
11. cuBLAS for comparison: `... ncu --launch-skip 5 --launch-count 1 --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats ./build/sgemm 4096 --version 0 --no-csv` (no name filter; every launch in this run is a cuBLAS kernel).

## Stage 0 report

Commits (oldest first):

| Hash | Message |
|---|---|
| 457b314 | stage0: initial repo, README, and environment report script |
| 3568f51 | stage0: CMake build, CUDA error checking, event timer, device info smoke test |
| da61d52 | stage0: add slurm batch script for gpu runs |
| 91b301e | stage0: add vector add with cpu check and event timing |
| e44a522 | stage0: add csv result writer and write vector add results to results/stage0 |
| 08ac004 | stage0: add optional l2 flush before each timed rep |
| 76c0a13 | stage0: add batched launches per timed rep and flag launch-bound sizes |
| bf3910c | stage0: generalize csv rows and add shared bench harness with occupancy info |
| 94a5eab | stage0: add ncu counter access probe script |
| d23afd2 | stage0: add saxpy with double-precision cpu reference |
| 2102290 | stage0: add device-to-device copy kernel and cudaMemcpy measurement |
| d7228a4 | stage0: add size sweep script for copy and vector add |
| 3404d22 | stage0: add fixed-size baseline script |
| 2eb576c | stage0: add plot script and csv-to-markdown table tool |
| 3bb09bc | stage0: add size sweep and fixed-size baseline results with plots |
| 85c0a67 | stage0: add counter-free kernel report script (ptxas, sass opcode counts) |
| 50f54c3 | stage0: add ptxas and sass reports for stage 0 kernels |
| 212944f | stage0: add nsight systems kernel summary script and vector add traces |

Headline results (2 GiB total data, L2 flushed, jobs 24037480 and 24037481):

| Kernel | GB/s | % of 1792.1 nominal |
|---|---|---|
| vector_add | 1528.5 | 85.3 |
| copy_kernel | 1493.7 | 83.3 |
| memcpy_d2d | 1468.6 | 81.9 |

Achievable bandwidth for later stages: about 1530 GB/s (flushed vector_add, 1 to 2 GiB). Warm runs exceed nominal from 12 MiB to 128 MiB of total data and fall to within 3% of cold from 384 MiB on.

Failures: ncu counter access denied (above). No correctness failures.

Open questions (details in `docs/stage0.md`): L2 residency as the cause of warm results above nominal; cold timings near multiples of 1.024 us; batched per-launch times on about 2.05 us steps and above the nsys kernel duration at small sizes (launch-rate question, not pursued); memcpy_d2d warm peak of 5573.8 GB/s; copy plateau below vector_add.

## Stage 1 report

Commits:

| Hash | Message |
|---|---|
| 021e2ef | stage1: add memory access pattern kernels with exact checks |
| 4d8f802 | stage1: add access pattern run script |
| 364efe8 | stage1: add stage 1 plots to plot script |
| 207f0a8 | stage1: add access pattern results, plots, and ptxas/sass reports |
| b9577c6 | stage1: add stage 1 doc |

Headline results (2^28 uint32 elements, L2 flushed, job 24037839, % of 1530 GB/s achievable):

| Pattern | Effective GB/s | % achievable |
|---|---|---|
| contiguous | 1493.7 | 97.6 |
| strided, stride 2 | 1019.0 | 66.6 |
| strided, stride 4 | 621.0 | 40.6 |
| strided, stride 8 / 16 / 32 | 352.2 / 355.9 / 353.6 | 23.0 / 23.3 / 23.1 |
| strided, stride 64 | 250.1 | 16.3 |
| gather, random permutation | 180.5 | 11.8 |
| offset 1 to 31 | 1485.2 | 97.1 |
| aos_x / soa_x | 618.3 / 1489.5 | 40.4 / 97.4 |
| aos_sum / soa_sum | 1549.3 / 1549.3 | 101.3 / 101.3 |

Failures: none. ncu profiling of contiguous and strided cases not possible (counter access); commands under Pending profiling.

Open questions (details in `docs/stage1.md`): stride curve vs sector over-fetch; stride 64 below stride 32; gather limited by sector efficiency or latency; identical cost for all offsets 1 to 31; 4:1 read-to-write kernels above copy bandwidth; aos_x DRAM bytes.

## Stage 2 report

Commits:

| Hash | Message |
|---|---|
| 153059c | stage2: add reduction harness and v1 global atomicAdd reduction |
| 70f5755 | stage2: add v2 shared-memory tree reduction with interleaved addressing |
| 63424fc | stage2: add v3 reduction with sequential addressing |
| 65de076 | stage2: add v4 reduction with first add during global load |
| 4cb46ca | stage2: add v5 reduction with warp-shuffle last warp |
| f6b01fd | stage2: add v6 grid-stride reduction with multiple elements per thread |
| f005b9d | stage2: add v7 cub DeviceReduce::Sum reference |
| 22384f0 | stage2: add reduction run script |
| f249a4b | stage2: add stage 2 plots to plot script |
| 2261908 | stage2: add reduction results, plots, and ptxas/sass reports |
| e45b8df | stage2: add stage 2 doc |
Headline results (2^28 floats, L2 flushed, job 24038225):

| Version | Median (ms) | GB/s | % of 1530 achievable | % of CUB |
|---|---|---|---|---|
| v1_atomic | 388.167694 | 2.8 | 0.2 | 0.2 |
| v2_interleaved | 1.204224 | 891.6 | 58.3 | 58.0 |
| v3_sequential | 1.198080 | 896.2 | 58.6 | 58.3 |
| v4_first_add | 0.716800 | 1498.0 | 97.9 | 97.4 |
| v5_warp_shuffle | 0.702464 | 1528.5 | 99.9 | 99.4 |
| v6_grid_stride | 0.704544 | 1524.0 | 99.6 | 99.1 |
| v7_cub | 0.698368 | 1537.5 | 100.5 | 100.0 |

Correctness: exact integer test and float tolerance test (2 * ceil(log2 n) * 2^-24 * sum|x|) passed for all 49 configurations.

Flagged (contradicts the usual expectation, recorded as measured): v2 and v3 take the same time at every size up to 2^26 and differ by 0.5% at 2^28.

Failures: none. ncu profiling of v2, v3, v5 pending (counter access).

Open questions (details in `docs/stage2.md`): v2 equal to v3; source of the v4 gain; same-address atomic throughput; v6 at 2^24; CUB above 1530 GB/s.

## Stage 3 report

Commits:

| Hash | Message |
|---|---|
| 1908f4e | stage3: add sgemm harness with cublas and cpu checks, and v1 naive kernel |
| 4f5aac8 | stage3: add v2 sgemm with coalesced global access |
| 205527b | stage3: add v3 sgemm with shared-memory tiling |
| fbbb26a | stage3: add v4 sgemm with 1d register blocking |
| 3d96368 | stage3: add v5 sgemm with 2d register blocking |
| ac28db7 | stage3: add v6 sgemm with float4 vectorized loads and stores |
| 89ebc87 | stage3: add sgemm run script |
| c15b31d | stage3: add tile and block size sweep script |
| d8cbda5 | stage3: add stage 3 plots to plot script |
| 6bb25b3 | stage3: add sgemm and tile sweep results, plots, and ptxas/sass reports |
| 89600ee | stage3: add stage 3 doc |
Headline results (square M = N = K, warm, job 24038801; GFLOP/s and % of cuBLAS):

| Version | 4096 | 8192 |
|---|---|---|
| cublas | 51082.8 | 51188.3 |
| v1_naive | 824.6 (1.6%) | 832.0 (1.6%) |
| v2_coalesced | 5768.3 (11.3%) | 4253.6 (8.3%) |
| v3_smem_tiling | 8197.5 (16.0%) | 6944.6 (13.6%) |
| v4_1d_regblock | 19141.1 (37.5%) | 17342.8 (33.9%) |
| v5_2d_regblock | 23039.8 (45.1%) | 19573.3 (38.2%) |
| v6_vectorized | 36845.6 (72.1%) | 37066.3 (72.4%) |

Best tile sweep configuration at 4096 (job 24038802): v5 128x128x16x8x8, 30351.3 GFLOP/s, theoretical occupancy 0.333.

Correctness: all 70 main-run configurations passed kernel-vs-cuBLAS; the 42 with M N K <= 2^30 also passed kernel-vs-CPU and cuBLAS-vs-CPU (double). Tolerance 16 sqrt(K) 2^-24 (|A||B|)_ij per element.

Flagged (recorded as measured): v1 is 3.3x faster at 1023 and 4097 than at 1024 and 4096; the v5 default is slower than v4 at 1024 and 2048; the fastest tile configuration has the lowest theoretical occupancy.

Failures: none. ncu profiling of all versions pending (counter access); commands under Pending profiling.

Open questions (details in `docs/stage3.md`): v1 at power-of-two K; v2 to v5 losing throughput from 4096 to 8192; v5 block count at small sizes; what limits high-occupancy configurations; which v6 change gives its gain; v6 with BK = 16.

Stage 3 is the last stage in this run. Stages 4 to 6 (roofline and consolidation) have not been started.

## Stages 4 to 6 report

Also in this block: the cuBLAS FP32 verification (recorded in `docs/stage3.md`) and the Stage 1 sector-model column (in `docs/stage1.md`).

Commits:

| Hash | Message |
|---|---|
| abfcfc4 | stage3: add cublas fp32 verification tool |
| 48dfcfe | stage3: record cublas fp32 verification results and kernel names |
| 7de9841 | stage1: add 32-byte sector model estimate of dram bytes |
| 5202a95 | stage1: document effective bandwidth counting and sector model comparison |
| 71f272f | stage4: add --warmup and --reps flags with project minimums |
| 9a0df71 | stage4: add clock and power logged sgemm and streaming run script |
| 221fc4a | stage4: add clock log summary script |
| c6cad16 | stage4: add first-pass mode and load-only baseline to reduction for tree-phase timing |
| 44c894a | stage4: add reduction tree-phase run script |
| 7d84d7b | stage4: add naive sgemm size sweep script |
| d1a4e23 | stage4: add clock, power, and temperature logs for sgemm and streaming runs |
| 36d668e | stage4: add reduction tree-phase and naive sgemm sweep results |
| afbed32 | stage4: add roofline script with arithmetic intensity table |
| 2c9b6db | stage4: add stage 4 plots to plot script |
| 1266ef8 | stage4: add roofline table and plots |
| 0006548 | stage4: add roofline and resource analysis doc for stages 4 to 6 |
Headline results (details in `docs/roofline.md`):

| Item | Result |
|---|---|
| FP32 peak | 188 SMs x 128 lanes x 2 x clock: 148.7 TFLOPS at 3090 MHz; 72.9 to 112.2 TFLOPS at the SM clocks observed under load |
| Clocks under load | 300 W board power in every run; SM 1515 to 2332 MHz; memory 13365 MHz (max 14001); 49 to 69 C |
| cuBLAS 8192 | 52077.8 GFLOP/s, 35.0% of peak at 3090 MHz, 71.4% of peak at its observed 1515 MHz |
| v6 8192 | 37520.4 GFLOP/s, 25.2% at 3090 MHz, 47.5% at its observed 1642 MHz |
| Memory-bound kernels | vector_add, saxpy, reduction v5 to v7 at 99.6% to 100.5% of the 1530 GB/s roof |
| Occupancy | fastest tile configuration (30351.3 GFLOP/s) has the lowest theoretical occupancy (0.333) |
| Reduction v2 vs v3 | equal even with the tree phase isolated on L2-resident data (20.500 vs 20.501 us per first pass at 2^22) |
| Naive SGEMM | 2.9x to 3.3x slower only at exactly 1024 and 4096, not at neighbouring multiples of 8 |

Failures: none. Counter probes: 6000_blackwell hive-dc-7-4-58 (job 24037338) and the Max-Q GRES type on hive-as-11-2-34 (job 24037343) also DENIED; two probes still queued.

Open questions (details in `docs/roofline.md`): power cap as the cause of kernel-dependent clocks; 1710.7 GB/s as the nominal ceiling at the loaded memory clock; v2 equal to v3; load-only baseline slower than v4 and v5; power-of-two stride in v1.

## Stage 7 report

Details in `docs/stage7.md`.

Commits:

| Hash | Message |
|---|---|
| 991c2e3 | stage7: add python benchmark harness matching the c++ harness |
| ce01d93 | stage7: add autotuned triton kernels for vector add, softmax, rmsnorm, matmul |
| 8c25580 | stage7: add triton driver with correctness checks, ir dumps, and autotune records |
| 31fe0cc | stage7: add stage 7 run script |
| 388dd41 | stage7: ignore cubin build outputs |
| 395bdc0 | stage7: add triton vector add, softmax, matmul results, autotune records, and matmul ir dumps (job 24047985) |
| e3ceb1a | stage7: add stage 7 doc |

Headline results (job 24047985, matmul warm, GFLOP/s and % of `cublasSgemm` in the same job):

| Shape | cuBLAS | Triton | CUDA v6 |
|---|---|---|---|
| 4096 | 50557.9 | 47127.3 (93.2%) | 37209.4 (73.6%) |
| 8192 | 50029.8 | 46334.9 (92.6%) | 38032.9 (76.0%) |
| 4097 | 45829.2 | 36174.2 (78.9%) | 34265.9 (74.8%) |
| 777 x 1111 x 333 | 13191.1 | 18714.9 (141.9%) | 8283.2 (62.8%) |

Triton matmul at 4096: BM=128, BN=128, BK=32, 8 warps, 3 stages; 220 registers, 65536 B shared memory; FP32 FFMA only (`input_precision="ieee"`). Triton vector add at 2^26, L2 flushed: 1512.4 GB/s (98.8% of 1530).

Flagged (recorded as measured): Triton beats `cublasSgemm` at 1024 (104.7%) and 777 x 1111 x 333 (141.9%); the 4097 and 1000 compiles use 255 registers with 6 B spill.

Failures: none in job 24047985. The previous session's Mac working folder held no uncopied KernelForge files.

Open questions (details in `docs/stage7.md`): two cuBLAS paths differ by 4.8% at 1024; register growth at non-divisible shapes; softmax at 3000 columns; autotune CSV lacks run-identification columns.

## Stage 8 report

Details in `docs/stage8.md`.

Commits:

| Hash | Message |
|---|---|
| 8d8eeb3 | stage8: add cuda softmax and rmsnorm kernels as a shared library |
| 6b611a7 | stage8: add ml kernel driver with fp64 reference checks and run script |
| fec3388 | stage8: add softmax and rmsnorm results for cuda, triton, pytorch (job 24048749) |
| 4587723 | stage8: add row lengths 1001 and 4099 to cover the cuda scalar path |
| 3e97e66 | stage8: add results for row lengths 1001 and 4099 (job 24049119) |
| a4eb6e8 | stage8: add stage 8 doc |

Headline results (2^26 FP32 elements per shape, L2 flushed, jobs 24048749 and 24049119, GB/s and % of 1530):

| Kernel | Range over 13 row lengths (128 to 16384) |
|---|---|
| cuda_softmax | 1417.0 to 1472.6 (92.6 to 96.2) |
| triton_softmax | 1464.5 to 1498.0 (95.7 to 97.9) |
| torch_softmax | 1448.3 to 1489.5 (94.7 to 97.3) |
| cuda_rmsnorm | 1417.0 to 1481.2 (92.6 to 96.8) |
| triton_rmsnorm | 1424.7 to 1498.0 (93.1 to 97.9) |
| torch_rmsnorm | 766.5 to 1464.5 (50.1 to 95.7) |

Flagged (recorded as measured): `F.rms_norm` at 50.1% of 1530 GB/s at 1001 columns and 64.4% at 16384; Triton RMSNorm at 512 columns below CUDA.

Failures: job 24048053 (uncommitted earlier code) failed CUDA correctness at every row length below 16384; the code was fixed before commit and the committed version passed all 78 checks. Job 24048144 (previous session test run) failed writing to an output directory outside the repository; not a kernel failure.

Open questions (details in `docs/stage8.md`): unrecorded fix of the earlier CUDA reduction bug; 1.024 to 2.048 us timer quantization; `F.rms_norm` dispatch; clocks not logged; DRAM bytes pending counter access.

## Stage 9

Skipped by choice (project owner's decision, 2026-09-26). No code or results.
