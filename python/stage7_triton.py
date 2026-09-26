#!/usr/bin/env python3
"""Stage 7: Triton vector add, softmax, and matmul (FP32), autotuned.

For each kernel and shape: correctness against a PyTorch reference, then timing with
python/kfbench.py. Records the autotuned config, registers, spills, and shared memory
of the compiled kernel. For matmul at one shape, dumps the lowering stages (TTIR,
TTGIR, LLIR, PTX) and the cubin, and writes the autotuner timings of every config.

usage: python python/stage7_triton.py [vadd] [softmax] [matmul] [--outdir results/stage7]
"""
import argparse
import csv
import math
import os
import subprocess
from pathlib import Path

# Keep the Triton kernel cache out of the repository; must be set before importing triton.
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/kf_scratch/triton_cache"))

import torch  # noqa: E402
import triton

import kfbench
from triton_kernels import MATMUL_SPACE, add_kernel, matmul_kernel, softmax_kernel

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
BUILD = f"torch-{torch.__version__}+triton-{triton.__version__}"


def compiled_info(ck):
    """Registers, spills, shared memory, warps, stages of a CompiledKernel."""
    if getattr(ck, "n_regs", None) is None and hasattr(ck, "_init_handles"):
        ck._init_handles()
    md = ck.metadata
    return {"regs": ck.n_regs, "spills": ck.n_spills, "smem": md.shared,
            "num_warps": md.num_warps, "num_stages": getattr(md, "num_stages", "")}


def cfg_str(cfg):
    kw = ",".join(f"{k}={v}" for k, v in cfg.kwargs.items())
    return f"{kw};num_warps={cfg.num_warps};num_stages={cfg.num_stages}".lstrip(",;")


def report(out_csv, info, row):
    kfbench.append_csv(out_csv, info, row)
    print("RESULT", " ".join(f"{k}={v}" for k, v in row.items() if k not in ("bytes", "flops")), flush=True)


def run_vadd(outdir, info):
    for n in (1 << 20, 1 << 24, 1 << 26, 1000003):
        x = torch.rand(n, device="cuda") * 2 - 1
        y = torch.rand(n, device="cuda") * 2 - 1
        out = torch.full_like(x, float("nan"))
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
        ck = add_kernel[grid](x, y, out, n)
        torch.cuda.synchronize()
        ok = torch.equal(out, x + y)
        print(f"correctness vadd n={n}: {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            continue
        ci = compiled_info(ck)
        cfg = add_kernel.best_config
        for flush in (False, True):
            st = kfbench.bench(lambda: add_kernel[grid](x, y, out, n), flush=flush)
            row = kfbench.result_row("triton_vector_add", cfg_str(cfg), n, 12.0 * n, n,
                                     str(triton.cdiv(n, cfg.kwargs["BLOCK"])), str(32 * ci["num_warps"]),
                                     ci["regs"], ci["spills"], ci["smem"], st, "GB/s")
            report(outdir / "vector_add.csv", info, row)


def run_softmax(outdir, info):
    for cols in (1024, 4096, 16384, 3000):
        rows = (1 << 24) // cols
        x = torch.randn(rows, cols, device="cuda")
        out = torch.full_like(x, float("nan"))
        block = triton.next_power_of_2(cols)
        ck = softmax_kernel[(rows,)](out, x, cols, BLOCK=block)
        torch.cuda.synchronize()
        ref = torch.softmax(x.double(), dim=-1)
        err = ((out.double() - ref).abs() / ref.abs().clamp_min(1e-30)).max().item()
        ok = err <= 1e-5
        print(f"correctness softmax rows={rows} cols={cols}: {'PASS' if ok else 'FAIL'} (max rel err {err:.3g})",
              flush=True)
        if not ok:
            continue
        ci = compiled_info(ck)
        st = kfbench.bench(lambda: softmax_kernel[(rows,)](out, x, cols, BLOCK=block))
        row = kfbench.result_row("triton_softmax", cfg_str(softmax_kernel.best_config) + f";BLOCK={block}",
                                 rows * cols, 8.0 * rows * cols, 0, str(rows), str(32 * ci["num_warps"]),
                                 ci["regs"], ci["spills"], ci["smem"], st, "GB/s", {"rows": rows, "cols": cols})
        report(outdir / "softmax.csv", info, row)


def matmul_check(a, b, c):
    """Per-element |c - ref| <= 16 sqrt(K) 2^-24 (|A||B|)_ij, as in src/sgemm.cu."""
    k = a.shape[1]
    ref = a.double() @ b.double()
    absprod = a.double().abs() @ b.double().abs()
    ratio = ((c.double() - ref).abs() / absprod).max().item()
    tol = 16 * math.sqrt(k) * 2.0**-24
    return ratio <= tol, ratio, tol


def run_matmul(outdir, info, dump_shape=(4096, 4096, 4096)):
    shapes = [(1024,) * 3, (2048,) * 3, (4096,) * 3, (8192,) * 3, (1000,) * 3, (4097,) * 3,
              (777, 1111, 333)]
    tune_rows = []
    for m, n, k in shapes:
        a = torch.rand(m, k, device="cuda") * 2 - 1
        b = torch.rand(k, n, device="cuda") * 2 - 1
        c = torch.full((m, n), float("nan"), device="cuda")
        grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)  # noqa: E731
        args = (a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
        ck = matmul_kernel[grid](*args)
        torch.cuda.synchronize()
        ok, ratio, tol = matmul_check(a, b, c)
        print(f"correctness matmul {m}x{n}x{k}: {'PASS' if ok else 'FAIL'} (max err/(|A||B|) {ratio:.3g}, tol {tol:.3g})",
              flush=True)
        for cfg, t in matmul_kernel.configs_timings.items():
            tt = t if isinstance(t, (list, tuple)) else [t]
            tune_rows.append({"M": m, "N": n, "K": k, "config": cfg_str(cfg),
                              "autotune_ms_median": tt[0], "chosen": int(cfg == matmul_kernel.best_config)})
        if not ok:
            continue
        ci = compiled_info(ck)
        cfg = matmul_kernel.best_config
        flops = 2.0 * m * n * k
        nbytes = 4.0 * (m * k + k * n + m * n)
        st = kfbench.bench(lambda: matmul_kernel[grid](*args))
        nprog = triton.cdiv(m, cfg.kwargs["BM"]) * triton.cdiv(n, cfg.kwargs["BN"])
        row = kfbench.result_row("triton_matmul", cfg_str(cfg), m, nbytes, flops, str(nprog),
                                 str(32 * ci["num_warps"]), ci["regs"], ci["spills"], ci["smem"], st, "GFLOP/s",
                                 {"M": m, "N": n, "K": k})
        report(outdir / "matmul.csv", info, row)
        # cuBLAS through PyTorch, FP32 (allow_tf32 off), same job, same inputs.
        st = kfbench.bench(lambda: torch.matmul(a, b, out=c))
        row = kfbench.result_row("torch_matmul_cublas", "allow_tf32=False", m, nbytes, flops, "-", "-",
                                 -1, -1, -1, st, "GFLOP/s", {"M": m, "N": n, "K": k})
        report(outdir / "matmul.csv", info, row)
        if (m, n, k) == dump_shape:
            dump_ir(ck, outdir / "matmul_ir", ci, cfg)
    with open(outdir / "matmul_autotune.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tune_rows[0].keys()))
        w.writeheader()
        w.writerows(tune_rows)


def dump_ir(ck, d, ci, cfg):
    d.mkdir(parents=True, exist_ok=True)
    for key, ext in (("ttir", "ttir"), ("ttgir", "ttgir"), ("llir", "ll"), ("ptx", "ptx")):
        (d / f"matmul_4096.{ext}").write_text(ck.asm[key])
    cubin = d / "matmul_4096.cubin"
    cubin.write_bytes(ck.asm["cubin"])
    sass = subprocess.run(["cuobjdump", "-sass", str(cubin)], capture_output=True, text=True).stdout
    ops = subprocess.run(["python3", str(kfbench.ROOT / "scripts" / "sass_stats.py")], input=sass,
                         capture_output=True, text=True).stdout
    (d / "sass_ops_matmul_4096.csv").write_text(ops)
    (d / "compiled_kernel.txt").write_text(
        f"config: {cfg_str(cfg)}\nregs: {ci['regs']}\nspills: {ci['spills']}\nshared_bytes: {ci['smem']}\n"
        f"num_warps: {ci['num_warps']}\nnum_stages: {ci['num_stages']}\n")
    print(f"dumped IR to {d}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parts", nargs="*", default=["vadd", "softmax", "matmul"])
    p.add_argument("--outdir", default="results/stage7")
    a = p.parse_args()
    outdir = kfbench.ROOT / a.outdir
    info = kfbench.run_info(BUILD)
    print("run info:", info, flush=True)
    print("matmul config space:", MATMUL_SPACE, flush=True)
    if "vadd" in a.parts:
        run_vadd(outdir, info)
    if "softmax" in a.parts:
        run_softmax(outdir, info)
    if "matmul" in a.parts:
        run_matmul(outdir, info)


if __name__ == "__main__":
    main()
