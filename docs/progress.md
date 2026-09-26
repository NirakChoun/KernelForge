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
