#!/usr/bin/env python3
"""Summarizes the Stage 4 nvidia-smi log per run.

For each run between its start and end markers, samples with power draw at least 50%
of that run's maximum are treated as the active (kernel) phase; the rest of the run is
host setup and correctness checks. Reports median, min, and max SM clock, memory clock,
power, and temperature over active samples, and the FP32 peak at the median SM clock.

usage: python3 scripts/clock_summary.py [results/stage4]
"""
import csv
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

SMS = 188
FP32_LANES_PER_SM = 128  # 24064 CUDA cores (NVIDIA datasheet) / 188 SMs (device query)

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/stage4")
fmt = "%Y/%m/%d %H:%M:%S.%f"


def parse(ts):
    return datetime.strptime(ts.strip(), fmt)


samples = []
with open(d / "gpu_log.csv") as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) < 5 or "N/A" in row[1:]:
            continue
        samples.append((parse(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))

runs = {}
with open(d / "gpu_log_markers.csv") as f:
    for row in csv.DictReader(f):
        runs.setdefault(row["label"], {})[row["event"]] = parse(row["timestamp"])

idle = [s for s in samples if s[0] < min(v["start"] for v in runs.values())]
out = []
for label, t in runs.items():
    ss = [s for s in samples if t["start"] <= s[0] <= t["end"]]
    if not ss:
        continue
    pmax = max(s[3] for s in ss)
    act = [s for s in ss if s[3] >= 0.5 * pmax]
    sm = [s[1] for s in act]
    med_sm = st.median(sm)
    out.append({
        "label": label, "samples_total": len(ss), "samples_active": len(act),
        "sm_mhz_median": med_sm, "sm_mhz_min": min(sm), "sm_mhz_max": max(sm),
        "mem_mhz_median": st.median(s[2] for s in act),
        "power_w_median": round(st.median(s[3] for s in act), 1), "power_w_max": pmax,
        "temp_c_median": st.median(s[4] for s in act), "temp_c_max": max(s[4] for s in act),
        "fp32_peak_gflops_at_median_clock": round(SMS * FP32_LANES_PER_SM * 2 * med_sm * 1e6 / 1e9, 1),
    })
with open(d / "clock_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
for o in out:
    print(o)
if idle:
    print("idle before runs: sm", st.median(s[1] for s in idle), "MHz, power", st.median(s[3] for s in idle), "W")
