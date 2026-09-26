#!/usr/bin/env python3
"""Stages 4 to 6: arithmetic intensity and roofline for every Stage 0 to 3 kernel.

FLOPs count the FP32 floating-point operations the algorithm needs (FMA = 2). Bytes are
compulsory DRAM traffic, the same bytes the stage CSVs use for bandwidth: every input
read once and every output written once (SGEMM: 4 (MK + KN + MN)). Arithmetic
intensity (AI) = FLOPs / bytes.

Ceilings:
  memory: 1530 GB/s (Stage 0 achievable, L2 flushed) and 1792.1 GB/s (nominal).
  compute: SMs x FP32 lanes per SM x 2 x SM clock; 188 SMs (device query), 128 lanes
  per SM (24064 CUDA cores in the NVIDIA datasheet / 188 SMs). Evaluated at the 3090 MHz
  maximum SM clock (nvidia-smi clocks.max.sm), and shown as a band over the median SM
  clocks observed in Stage 4 (results/stage4/clock_summary.csv).

Writes results/stage4/roofline.csv and results/stage4/roofline.png.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SMS, LANES, MAX_MHZ = 188, 128, 3090
ACHIEVABLE, NOMINAL = 1530.0, 1792.1


def peak_gflops(mhz):
    return SMS * LANES * 2 * mhz * 1e6 / 1e9


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pick(rs, **kw):
    out = [r for r in rs if all(str(r[k]) == str(v) for k, v in kw.items())]
    assert out, kw
    return out[0]


points = []


def add(stage, r, flops, nbytes, config, source):
    points.append(dict(stage=stage, kernel=r["kernel"], config=config, n=r["n"], flops=flops,
                       bytes=nbytes, median_ms=float(r["median_ms"]), job_id=r["job_id"], source=source))


s0 = rows(RES / "stage0" / "fixed_sizes.csv")
for k, fl in [("vector_add", 1), ("saxpy", 2), ("copy_kernel", 0), ("memcpy_d2d", 0)]:
    r = pick(s0, kernel=k, n=67108864, l2_flush=1, launches_per_rep=1)
    add(0, r, fl * int(r["n"]), float(r["bytes"]), "n=67108864, L2 flushed", "stage0/fixed_sizes.csv")
s1 = rows(RES / "stage1" / "patterns.csv")
for k, v in [("contiguous", "-"), ("strided", "stride=8"), ("gather", "random_permutation"),
             ("aos_x", "-"), ("aos_sum", "-")]:
    r = pick(s1, kernel=k, variant=v, l2_flush=1)
    add(1, r, 0, float(r["bytes"]), f"{v}, L2 flushed", "stage1/patterns.csv")
for r in [x for x in rows(RES / "stage2" / "reduce.csv") if x["n"] == "268435456"]:
    add(2, r, int(r["n"]) - 1, float(r["bytes"]), "n=2^28, L2 flushed", "stage2/reduce.csv")
for r in [x for x in rows(RES / "stage3" / "sgemm.csv") if x["M"] == "8192"]:
    m = int(r["M"])
    add(3, r, 2 * m ** 3, 12 * m * m, "M=N=K=8192, warm", "stage3/sgemm.csv")

clock = {r["label"]: float(r["sm_mhz_median"]) for r in rows(RES / "stage4" / "clock_summary.csv")}
obs_lo, obs_hi = min(clock.values()), max(clock.values())

out = []
for p in points:
    t = p["median_ms"] * 1e-3
    ai = p["flops"] / p["bytes"]
    gflops = p["flops"] / t / 1e9
    gbs = p["bytes"] / t / 1e9
    roof = min(peak_gflops(MAX_MHZ), ai * ACHIEVABLE) if ai else 0.0
    out.append({**p, "ai_flop_per_byte": round(ai, 4), "gflops": round(gflops, 1), "gbs": round(gbs, 1),
                "roof_gflops": round(roof, 1),
                "bound": ("memory" if ai * ACHIEVABLE < peak_gflops(MAX_MHZ) else "compute") if ai else "memory (no FLOPs)",
                "pct_of_roof": round(100 * gflops / roof, 1) if roof else "",
                "pct_of_1530": round(100 * gbs / ACHIEVABLE, 1)})
with open(RES / "stage4" / "roofline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
x = [10 ** (e / 10) for e in range(-20, 41)]
ax.plot(x, [min(peak_gflops(MAX_MHZ), a * ACHIEVABLE) for a in x], "k-", lw=1.2,
        label=f"1530 GB/s and FP32 peak at {MAX_MHZ} MHz ({peak_gflops(MAX_MHZ) / 1e3:.1f} TFLOPS)")
ax.plot(x, [a * NOMINAL for a in x], "k:", lw=0.8, label="1792.1 GB/s nominal")
ax.axhline(110000, color="tab:gray", ls="--", lw=0.8, label="110 TFLOPS (datasheet)")
ax.axhspan(peak_gflops(obs_lo), peak_gflops(obs_hi), color="tab:orange", alpha=0.15,
           label=f"FP32 peak at observed clocks {obs_lo:.0f} to {obs_hi:.0f} MHz")
colors = {0: "tab:blue", 2: "tab:green", 3: "tab:red"}
for o in out:
    if o["flops"] == 0:
        continue
    ax.scatter(o["ai_flop_per_byte"], o["gflops"], color=colors[o["stage"]], s=18, zorder=3)
    ax.annotate(o["kernel"], (o["ai_flop_per_byte"], o["gflops"]), fontsize=6,
                xytext=(3, 2), textcoords="offset points")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity (FLOP / compulsory DRAM byte)")
ax.set_ylabel("GFLOP/s")
ax.set_ylim(0.3, 3e5)
ax.set_title("Roofline, RTX PRO 6000 Blackwell Max-Q, Stages 0 to 3", fontsize=10)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(RES / "stage4" / "roofline.png", dpi=120)
for o in out:
    print(o["stage"], o["kernel"], o["config"], o["ai_flop_per_byte"], o["gflops"], o["gbs"],
          o["roof_gflops"], o["bound"], o["pct_of_roof"], o["pct_of_1530"])
