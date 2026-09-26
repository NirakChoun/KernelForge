"""Python benchmark harness for Triton and PyTorch kernels, matching the C++ harness.

- Timing: CUDA events around each rep, 10 warm-up and at least 100 timed reps, median.
- Before each start event the stream runs a short spin kernel (torch.cuda._sleep), so the
  CPU has enqueued the start event and the measured launch before the GPU reaches them.
  Without it, Python launch latency (tens of microseconds for a Triton launch) would be
  counted as GPU time for short kernels.
- Optional L2 flush: a 256 MiB write before each rep, outside the timed region.
- CSV rows carry the same run-identification columns as include/kf/csv.hpp.
"""
import csv
import ctypes
import datetime
import os
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SPIN_CYCLES = 200_000  # about 0.1 ms at 2 GHz; covers Python launch latency
LAUNCH_BOUND_MS = 0.020


def cuda_driver_version():
    v = ctypes.c_int()
    ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(v))
    return f"{v.value // 1000}.{(v.value % 1000) // 10}"


def run_info(build_type):
    return {
        "gpu": torch.cuda.get_device_name(0),
        "job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "date": datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": cuda_driver_version(),
        "build_type": build_type,
    }


def bench(fn, warmup=10, reps=100, flush=False):
    assert warmup >= 10 and reps >= 100
    scratch = torch.empty(64 * 2**20, dtype=torch.int32, device="cuda") if flush else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(reps):
        if scratch is not None:
            scratch.fill_(i)
        torch.cuda._sleep(SPIN_CYCLES)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    times.sort()
    return {"median_ms": statistics.median(times), "min_ms": times[0],
            "stddev_ms": statistics.stdev(times), "warmup": warmup, "reps": reps,
            "l2_flush": int(flush)}


def append_csv(path, info, row):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(info.keys()) + list(row.keys())
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path) as f:
            header = f.readline().strip()
        if header != ",".join(cols):
            raise SystemExit(f"csv header mismatch in {path}\n  file: {header}\n  row:  {','.join(cols)}")
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(cols)
        w.writerow([info[c] for c in info] + [row[c] for c in row])


def fmt(x, digits=6):
    return f"{x:.{digits}f}"


def result_row(kernel, variant, n, nbytes, flops, grid, block, regs, local, smem, stats, unit,
               extra=None):
    t = stats["median_ms"] * 1e-3
    metric = (flops if unit == "GFLOP/s" else nbytes) / t / 1e9
    row = {"kernel": kernel, "variant": variant, "n": n, "bytes": f"{nbytes:.0f}",
           "flops": f"{flops:.0f}", "grid": grid, "block": block, "regs": regs,
           "local_bytes": local, "smem_bytes": smem, "l2_flush": stats["l2_flush"],
           "warmup": stats["warmup"], "reps": stats["reps"],
           "median_ms": fmt(stats["median_ms"]), "min_ms": fmt(stats["min_ms"]),
           "stddev_ms": fmt(stats["stddev_ms"]), "metric_value": f"{metric:.1f}",
           "metric_unit": unit, "launch_bound": int(stats["median_ms"] < LAUNCH_BOUND_MS)}
    row.update(extra or {})
    return row
