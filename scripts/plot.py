#!/usr/bin/env python3
"""Plots KernelForge results CSVs into PNGs under results/.

usage: .venv/bin/python scripts/plot.py <stage> [<stage> ...]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
NOMINAL_GBS = 1792.1     # 2 * 14001 MHz * 512 bit / 8
ACHIEVABLE_GBS = 1530.0  # measured, L2 flushed (Stage 0)
L2_MIB = 128
MIB = 1 << 20


def ceilings(ax, achievable=True):
    ax.axhline(NOMINAL_GBS, color="0.3", ls="--", lw=1, label=f"nominal {NOMINAL_GBS} GB/s")
    if achievable:
        ax.axhline(ACHIEVABLE_GBS, color="0.6", ls=":", lw=1,
                   label=f"achievable ~{ACHIEVABLE_GBS:.0f} GB/s (flushed)")


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def stage0():
    df = pd.read_csv(RES / "stage0" / "sweep.csv")
    df["total_mib"] = df["bytes"] / MIB
    styles = {"vector_add": "tab:blue", "copy_kernel": "tab:orange", "memcpy_d2d": "tab:green"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel, color in styles.items():
        k = df[df.kernel == kernel]
        for (flush, launches), ls, lab in [((0, 1), "-", "warm"), ((1, 1), "--", "cold (L2 flushed)"),
                                           ((0, 100), ":", "warm, 100 launches/rep")]:
            s = k[(k.l2_flush == flush) & (k.launches_per_rep == launches)].sort_values("total_mib")
            if len(s):
                ax.plot(s.total_mib, s.metric_value, ls=ls, marker="o", ms=3, color=color,
                        label=f"{kernel} {lab}")
    ceilings(ax)
    ax.axvline(L2_MIB, color="k", lw=0.8, alpha=0.5)
    ax.text(L2_MIB * 1.05, ax.get_ylim()[0] + 50, "L2 = 128 MiB", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("total data moved per launch (MiB)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_title("Stage 0: bandwidth vs size, RTX PRO 6000 Blackwell Max-Q", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    save(fig, RES / "stage0" / "sweep_bandwidth.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel, color in styles.items():
        for (flush, launches), ls in [((0, 1), "-"), ((1, 1), "--"), ((0, 100), ":")]:
            s = df[(df.kernel == kernel) & (df.l2_flush == flush) &
                   (df.launches_per_rep == launches)].sort_values("total_mib")
            if len(s):
                ax.plot(s.total_mib, s.median_ms * 1e3, ls=ls, marker="o", ms=3, color=color,
                        label=f"{kernel} flush={flush} K={launches}")
    ax.axhline(20, color="r", lw=0.8, label="20 us launch-bound threshold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("total data moved per launch (MiB)")
    ax.set_ylabel("median time per launch (us)")
    ax.set_title("Stage 0: time per launch", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    save(fig, RES / "stage0" / "sweep_time.png")


def stage1():
    df = pd.read_csv(RES / "stage1" / "patterns.csv")
    df["pct"] = 100 * df.metric_value / ACHIEVABLE_GBS
    cold = df[df.l2_flush == 1]

    st = cold[cold.kernel == "strided"].copy()
    st["stride"] = st.variant.str.split("=").str[1].astype(int)
    st = st.sort_values("stride")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(st.stride, st.metric_value, marker="o", label="strided read, contiguous write")
    ceilings(ax)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("read stride (elements of 4 B)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_ylim(0, NOMINAL_GBS * 1.1)
    ax.set_title("Stage 1: stride sweep, 2^28 elements, L2 flushed", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, RES / "stage1" / "stride.png")

    off = cold[cold.kernel == "offset"].copy()
    off["offset"] = off.variant.str.split("=").str[1].astype(int)
    off = off.sort_values("offset")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(off.offset, off.metric_value, marker="o", label="read in[i + offset]")
    ax.axhline(ACHIEVABLE_GBS, color="0.6", ls=":", lw=1, label="achievable ~1530 GB/s")
    ax.set_xlabel("start offset (elements of 4 B)")
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.set_xticks(range(0, 33, 4))
    ax.set_title("Stage 1: misaligned start offset, 2^28 elements, L2 flushed", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, RES / "stage1" / "offset.png")

    summary = [("contiguous", "-"), ("strided", "stride=2"), ("strided", "stride=8"),
               ("strided", "stride=32"), ("gather", "random_permutation"), ("offset", "offset=1"),
               ("aos_x", "-"), ("soa_x", "-"), ("aos_sum", "-"), ("soa_sum", "-")]
    rows = [cold[(cold.kernel == k) & (cold.variant == v)].iloc[0] for k, v in summary
            if len(cold[(cold.kernel == k) & (cold.variant == v)])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{r.kernel}\n{r.variant.replace('random_permutation', 'random')}" if r.variant != "-" else r.kernel for r in rows]
    ax.bar(range(len(rows)), [r.pct for r in rows], color="tab:blue")
    for i, r in enumerate(rows):
        ax.text(i, r.pct + 1, f"{r.pct:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("% of achievable 1530 GB/s")
    ax.set_title("Stage 1: effective bandwidth by pattern, L2 flushed", fontsize=10)
    save(fig, RES / "stage1" / "patterns.png")


def stage2():
    df = pd.read_csv(RES / "stage2" / "reduce.csv")
    pow2 = df[(df.n & (df.n - 1)) == 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel, g in pow2.groupby("kernel"):
        g = g.sort_values("n")
        ax.plot(g.n, g.metric_value, marker="o", ms=4, label=kernel)
    ceilings(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("elements (float)")
    ax.set_ylabel("effective bandwidth, input bytes / time (GB/s)")
    ax.set_title("Stage 2: reduction versions, L2 flushed", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, RES / "stage2" / "reduce_bandwidth.png")

    big = df[df.n == df.n.max()].sort_values("kernel")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pct = 100 * big.metric_value / ACHIEVABLE_GBS
    ax.bar(range(len(big)), pct, color="tab:blue")
    for i, (v, p) in enumerate(zip(big.metric_value, pct)):
        ax.text(i, p * 1.1, f"{v:.1f} GB/s", ha="center", fontsize=7)
    ax.set_xticks(range(len(big)))
    ax.set_xticklabels(big.kernel, fontsize=7, rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("% of achievable 1530 GB/s")
    ax.set_title(f"Stage 2: reduction at n = {int(big.n.iloc[0])}, L2 flushed", fontsize=10)
    save(fig, RES / "stage2" / "reduce_largest.png")


def stage3():
    df = pd.read_csv(RES / "stage3" / "sgemm.csv")
    sq = df[(df.M == df.N) & (df.N == df.K) & ((df.M & (df.M - 1)) == 0)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for kernel, g in sq.groupby("kernel"):
        g = g.sort_values("M")
        ax.plot(g.M, g.metric_value, marker="o", ms=4, label=kernel,
                ls="--" if kernel == "cublas" else "-")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("M = N = K")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Stage 3: SGEMM versions vs cuBLAS, square sizes", fontsize=10)
    ax.legend(fontsize=7)
    save(fig, RES / "stage3" / "sgemm_gflops.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sizes = sorted(sq.M.unique())
    kernels = [k for k in sorted(sq.kernel.unique()) if k != "cublas"]
    width = 0.8 / len(kernels)
    for j, kernel in enumerate(kernels):
        pct = []
        for m in sizes:
            ref = sq[(sq.kernel == "cublas") & (sq.M == m)].metric_value
            val = sq[(sq.kernel == kernel) & (sq.M == m)].metric_value
            pct.append(100 * val.iloc[0] / ref.iloc[0] if len(ref) and len(val) else float("nan"))
        ax.bar([i + j * width for i in range(len(sizes))], pct, width, label=kernel)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(sizes))])
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("M = N = K")
    ax.set_ylabel("% of cuBLAS GFLOP/s")
    ax.set_title("Stage 3: SGEMM as a percentage of cuBLAS", fontsize=10)
    ax.legend(fontsize=7)
    save(fig, RES / "stage3" / "sgemm_pct_cublas.png")

    ts = pd.read_csv(RES / "stage3" / "tile_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (kernel, g) in zip(axes, ts.groupby("kernel")):
        cfgs = list(dict.fromkeys(g.variant))
        sizes = sorted(g.M.unique())
        width = 0.8 / len(sizes)
        for j, m in enumerate(sizes):
            vals = [g[(g.variant == c) & (g.M == m)].metric_value.iloc[0] for c in cfgs]
            ax.bar([i + j * width for i in range(len(cfgs))], vals, width, label=f"{m}")
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(cfgs))])
        ax.set_xticklabels([c.split("=")[1] for c in cfgs], fontsize=7, rotation=30)
        ax.set_title(kernel, fontsize=9)
    axes[0].set_ylabel("GFLOP/s")
    axes[-1].legend(title="M = N = K", fontsize=7)
    fig.suptitle("Stage 3: tile and block size sweep", fontsize=10)
    save(fig, RES / "stage3" / "tile_sweep.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    big = ts[ts.M == ts.M.max()]
    for kernel, g in big.groupby("kernel"):
        ax.scatter(g.theo_occupancy, g.metric_value, label=kernel)
        for _, r in g.iterrows():
            ax.annotate(r.variant.split("=")[1], (r.theo_occupancy, r.metric_value), fontsize=6)
    ax.set_xlabel("theoretical occupancy (occupancy API)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Stage 3: occupancy vs GFLOP/s at M = N = K = {int(big.M.max())}", fontsize=10)
    ax.legend(fontsize=7)
    save(fig, RES / "stage3" / "occupancy_vs_gflops.png")


STAGES = {"0": stage0, "1": stage1, "2": stage2, "3": stage3}

if __name__ == "__main__":
    if len(sys.argv) < 2 or any(a not in STAGES for a in sys.argv[1:]):
        sys.exit(f"usage: {sys.argv[0]} <stage>...  stages: {', '.join(STAGES)}")
    for a in sys.argv[1:]:
        STAGES[a]()
