# KernelForge progress

Running log of stage reports, performance counter access, and profiling that could not run. Stage details are in `docs/stageN.md`.

## Performance counter access

`ncu` needs access to GPU performance counters. On the primary node (hive-dc-7-5-58, `high`) it fails with `ERR_NVGPUCTRPERM` (job 24037044, ncu 2026.2.0.0). Probe: `scripts/ncu_probe.sh`, one job per node or GPU type on `low`, `ncu --metrics sm__cycles_elapsed.avg ./build-probe/vector_add 4096` with a multi-arch build (sm_80, 86, 89, 120). Type names from `sinfo -p low -o "%G|%D"` on 2026-09-26.

| GRES type | Node | Job | GPU reported | Counters |
|---|---|---|---|---|
| 6000_blackwell | hive-dc-7-4-58 | 24037338 | | pending (Resources) |
| 6000_blackwell | hive-dc-7-5-58 | 24037339 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| 6000_blackwell | hive-dc-7-5-62 | 24037340 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| 6000_blackwell | hive-dc-7-6-54 | 24037341 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | DENIED |
| 6000_blackwell | hive-dc-7-7-58 | 24037342 | | pending (Resources) |
| nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition | (hive-as-11-2-34) | 24037343 | | pending (Resources) |
| nvidia_rtx_5000_ada_generation | (hive-as-11-2-54) | 24037344 | | pending (Priority) |
| nvidia_l40s | hive-as-11-2-58 | 24037345 | NVIDIA L40S | DENIED |
| nvidia_a100-sxm4-80gb | hive-as-11-4-34 | 24037346 | NVIDIA A100-SXM4-80GB | DENIED |
| nvidia_a100-pcie-40gb | hive-as-11-4-54 | 24037347 | NVIDIA A100-PCIE-40GB | DENIED |
| nvidia_a100_80gb_pcie | hive-as-11-4-46 | 24037348 | NVIDIA A100 80GB PCIe | DENIED |
| a6000 | hive-dc-7-5-30 | 24037349 | NVIDIA RTX A6000 | DENIED |
| a100 | hive-dc-7-6-14 | 24037350 | NVIDIA A100 80GB PCIe | DENIED |

No probed node allows counter access so far. Nsight Systems CUDA tracing works on hive-dc-7-5-58 (job 24037643) and is used for kernel durations.

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
