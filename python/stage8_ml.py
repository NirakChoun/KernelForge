#!/usr/bin/env python3
"""Stage 8: row-wise softmax and RMSNorm in CUDA (build/libkf_ml.so), Triton, and PyTorch.

Each shape has 2^26 FP32 elements (256 MiB in, 256 MiB out); rows = 2^26 // cols.
Correctness against a float64 reference, then timing with python/kfbench.py (L2 flushed
before each rep). Bandwidth counts compulsory bytes: read x and write y once (8 B per
element), plus the RMSNorm weight row (4 cols B).

usage: python python/stage8_ml.py [--outdir results/stage8] [--cols 128,256,...]
"""
import argparse
import ctypes
import os

os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/kf_scratch/triton_cache"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402

import kfbench  # noqa: E402
from stage7_triton import compiled_info, cfg_str  # noqa: E402
from triton_kernels import rmsnorm_kernel, softmax_kernel  # noqa: E402

EPS = 1e-6
TOTAL = 1 << 26
DEFAULT_COLS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 1000, 3000, 5000]
BUILD = f"torch-{torch.__version__}+triton-{triton.__version__}"

lib = ctypes.CDLL(str(kfbench.ROOT / "build" / "libkf_ml.so"))
for f in (lib.kf_softmax, lib.kf_rmsnorm):
    f.restype = ctypes.c_int
lib.kf_softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.kf_rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                           ctypes.c_float, ctypes.c_void_p]


def cuda_attrs(which):
    r, l, s = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    assert lib.kf_ml_attrs(which, ctypes.byref(r), ctypes.byref(l), ctypes.byref(s)) == 0
    return r.value, l.value, s.value


def stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def check(y, ref, rtol, atol):
    err = (y.double() - ref).abs()
    bad = (err > atol + rtol * ref.abs()).sum().item()
    return bad == 0, (err / (ref.abs() + atol)).max().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results/stage8")
    p.add_argument("--cols", default=",".join(map(str, DEFAULT_COLS)))
    a = p.parse_args()
    out = kfbench.ROOT / a.outdir / "ml_kernels.csv"
    info = kfbench.run_info(BUILD)
    print("run info:", info, flush=True)
    for cols in map(int, a.cols.split(",")):
        rows = TOTAL // cols
        n = rows * cols
        torch.manual_seed(cols)
        x = torch.randn(rows, cols, device="cuda")
        w = torch.rand(cols, device="cuda") + 0.5
        y = torch.empty_like(x)
        block = triton.next_power_of_2(cols)
        vec = cols % 4 == 0
        ref_sm = torch.softmax(x.double(), dim=-1)
        xd = x.double()
        ref_rms = xd * torch.rsqrt((xd * xd).mean(dim=-1, keepdim=True) + EPS) * w.double()
        threads = lib.kf_ml_threads(cols)
        impls = {
            "softmax": [
                ("cuda_softmax", lambda: lib.kf_softmax(x.data_ptr(), y.data_ptr(), rows, cols, stream()),
                 0 if vec else 1),
                ("triton_softmax", lambda: softmax_kernel[(rows,)](y, x, cols, BLOCK=block), None),
                ("torch_softmax", lambda: y.copy_(torch.softmax(x, dim=-1)), None),
            ],
            "rmsnorm": [
                ("cuda_rmsnorm", lambda: lib.kf_rmsnorm(x.data_ptr(), w.data_ptr(), y.data_ptr(), rows, cols, EPS,
                                                        stream()), 2 if vec else 3),
                ("triton_rmsnorm", lambda: rmsnorm_kernel[(rows,)](y, x, w, cols, EPS, BLOCK=block), None),
                ("torch_rmsnorm", lambda: y.copy_(F.rms_norm(x, (cols,), w, EPS)), None),
            ],
        }
        for op, lst in impls.items():
            ref = ref_sm if op == "softmax" else ref_rms
            nbytes = 8.0 * n + (4.0 * cols if op == "rmsnorm" else 0.0)
            for name, fn, which in lst:
                y.fill_(float("nan"))
                ret = fn()
                torch.cuda.synchronize()
                if name.startswith("cuda") and ret != 0:
                    raise SystemExit(f"{name} launch error {ret}")
                ok, worst = check(y, ref, rtol=1e-5, atol=1e-7)
                print(f"correctness {name} rows={rows} cols={cols}: {'PASS' if ok else 'FAIL'} "
                      f"(max err/(|ref|+1e-7) {worst:.3g})", flush=True)
                if not ok:
                    continue
                grid, blk, regs, local, smem, variant = str(rows), "-", -1, -1, -1, "-"
                if name.startswith("cuda"):
                    regs, local, smem = cuda_attrs(which)
                    blk = str(threads)
                    variant = "float4" if vec else "scalar"
                elif name.startswith("triton"):
                    ck = fn()  # returns the compiled kernel after tuning
                    ci = compiled_info(ck)
                    kern = softmax_kernel if op == "softmax" else rmsnorm_kernel
                    variant = cfg_str(kern.best_config) + f";BLOCK={block}"
                    blk, regs, local, smem = str(32 * ci["num_warps"]), ci["regs"], ci["spills"], ci["smem"]
                else:
                    grid = "-"
                    variant = "torch.softmax" if op == "softmax" else "F.rms_norm"
                    if op == "softmax":
                        fn = lambda: torch.softmax(x, dim=-1)  # noqa: E731
                    else:
                        fn = lambda: F.rms_norm(x, (cols,), w, EPS)  # noqa: E731
                st = kfbench.bench(fn, flush=True)
                row = kfbench.result_row(name, variant, n, nbytes, 0, grid, blk, regs, local, smem, st, "GB/s",
                                         {"rows": rows, "cols": cols,
                                          "pct_of_1530": f"{100 * nbytes / (st['median_ms'] * 1e-3) / 1e9 / 1530:.1f}"})
                kfbench.append_csv(out, info, row)
                print("RESULT", name, variant, f"rows={rows} cols={cols}", f"median_ms={row['median_ms']}",
                      f"GB/s={row['metric_value']}", f"regs={regs}", flush=True)


if __name__ == "__main__":
    main()
