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


STAGES = {"0": stage0}

if __name__ == "__main__":
    if len(sys.argv) < 2 or any(a not in STAGES for a in sys.argv[1:]):
        sys.exit(f"usage: {sys.argv[0]} <stage>...  stages: {', '.join(STAGES)}")
    for a in sys.argv[1:]:
        STAGES[a]()
