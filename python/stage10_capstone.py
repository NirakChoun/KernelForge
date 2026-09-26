#!/usr/bin/env python3
"""Stage 10 capstone: when does Triton match or beat hand-written CUDA, and why?

Matmul: Triton (autotuned over MATMUL_SPACE) and cuBLAS through PyTorch, then
build/sgemm versions 0 (cuBLAS), 6, and 5 (128x128x16x8x8) at the same shapes in the
same job. Fixed-config Triton runs at 4096 and 4097 separate tile shape, BK, warps,
and pipeline depth (num_stages). Softmax: Triton vs build/libkf_ml.so, L2 flushed.

Before every timed run the kernel runs back to back for at least SUSTAIN_S seconds,
so the SM clock settles under the 300 W limit and the nvidia-smi log (100 ms samples,
started by scripts/stage10_capstone.sh) has samples inside the run. Each run is
bracketed by start/end markers in <outdir>/gpu_log_markers.csv; per-run clocks are
computed by scripts/clock_summary.py and joined on the `label` column.

Static analysis per compiled kernel: SASS opcode counts (cuobjdump), registers,
spills, shared memory, and resident blocks per SM from the CUDA occupancy API.

usage: python stage10_capstone.py [--outdir results/stage10] [--quick] [--parts matmul ablation softmax]
"""
import argparse
import collections
import csv
import ctypes
import datetime
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/kf_scratch/triton_cache"))

import torch  # noqa: E402
import triton  # noqa: E402

import kfbench  # noqa: E402
from stage7_triton import cfg_str, compiled_info, matmul_check  # noqa: E402
from triton_kernels import matmul_kernel, softmax_kernel  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
BUILD = f"torch-{torch.__version__}+triton-{triton.__version__}"
ROOT = kfbench.ROOT
SUSTAIN_S = 0.5

MATMUL_SHAPES = [(s, s, s) for s in (512, 1000, 1024, 1536, 2048, 3000, 4096, 4097, 6144, 8192)] + [
    (777, 1111, 333), (1000, 3000, 2000), (4096, 4096, 1000)]
ABLATION_SHAPES = [(4096,) * 3, (4097,) * 3]
# (BM, BN, BK, num_warps, num_stages). BK = 16 is the smallest K tile tl.dot accepts;
# v6 uses 128 x 128 x 8 with 8 warps.
ABLATION = [(128, 128, 32, 8, 1), (128, 128, 32, 8, 2), (128, 128, 32, 8, 3), (128, 128, 32, 8, 4),
            (128, 128, 16, 8, 1), (128, 128, 16, 8, 3), (128, 128, 32, 4, 3), (64, 64, 32, 4, 3),
            (128, 64, 32, 4, 4)]
SOFTMAX_COLS = [128, 512, 1000, 1001, 1024, 2048, 3000, 4096, 4099, 6000, 8192, 12000, 16384]
SOFTMAX_TOTAL = 1 << 26

libcuda = ctypes.CDLL("libcuda.so.1")
libml = ctypes.CDLL(str(ROOT / "build" / "libkf_ml.so"))
libml.kf_softmax.restype = ctypes.c_int
libml.kf_softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


class Run:
    def __init__(self, outdir):
        self.out = outdir
        self.out.mkdir(parents=True, exist_ok=True)
        self.info = kfbench.run_info(BUILD)
        self.markers = self.out / "gpu_log_markers.csv"
        if not self.markers.exists():
            self.markers.write_text("timestamp,event,label\n")

    def mark(self, event, label):
        ts = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        with open(self.markers, "a") as f:
            f.write(f"{ts},{event},{label}\n")

    def timed(self, label, fn, flush):
        """Sustain phase, then kfbench.bench (10 warm-up, 100 reps), inside markers."""
        self.mark("start", label)
        t0 = time.perf_counter()
        i = 0
        while time.perf_counter() - t0 < SUSTAIN_S:
            fn()
            i += 1
            if i % 16 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        st = kfbench.bench(fn, flush=flush)
        self.mark("end", label)
        return st

    def csv(self, name, row):
        kfbench.append_csv(self.out / name, self.info, row)


def occupancy(ck, threads, smem):
    """Resident blocks per SM for a Triton CompiledKernel (dynamic smem = its shared bytes)."""
    n = ctypes.c_int()
    rc = libcuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(n), ctypes.c_void_p(ck.function),
                                                             ctypes.c_int(threads), ctypes.c_size_t(smem))
    return n.value if rc == 0 else -1


def sass_ops(binary, match=None):
    """{function: Counter(opcode)} from cuobjdump -sass | c++filt | scripts/sass_stats.py."""
    sass = subprocess.run(f"cuobjdump -sass {binary} | c++filt", shell=True, capture_output=True, text=True).stdout
    txt = subprocess.run(["python3", str(ROOT / "scripts" / "sass_stats.py")], input=sass, capture_output=True,
                         text=True).stdout
    out = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(txt.splitlines()):
        if match is None or match in r["function"]:
            out[r["function"]][r["opcode"]] += int(r["count"])
    return out


def cubin_ops(ck):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(ck.asm["cubin"])
        f.flush()
        ops = sass_ops(f.name)
    total = collections.Counter()
    for c in ops.values():
        total.update(c)
    return total


def width(op):
    for w in ("128", "64"):
        if f".{w}" in op:
            return w
    return "32"


def summarize(ops):
    """Static SASS counts grouped by instruction class and access width."""
    s = collections.OrderedDict(total=sum(ops.values()), FFMA=ops.get("FFMA", 0))
    for cls in ("LDGSTS", "LDG", "STG", "LDS", "STS"):
        for w in ("32", "64", "128"):
            s[f"{cls}_{w}"] = 0
    for op, n in ops.items():
        base = op.split(".")[0]
        if base in ("LDGSTS", "LDG", "STG", "LDS", "STS"):
            s[f"{base}_{width(op)}"] += n
    s["BAR"] = sum(n for op, n in ops.items() if op.startswith("BAR"))
    s["MMA"] = sum(n for op, n in ops.items() if "MMA" in op.split(".")[0])
    return s


def write_sass(run, kernel, label, cfg, regs, spills, smem, threads, blocks, ops):
    row = collections.OrderedDict(kernel=kernel, label=label, config=cfg, regs=regs, spills=spills,
                                  smem_bytes=smem, threads=threads, blocks_per_sm=blocks)
    row.update(summarize(ops))
    run.csv("sass_summary.csv", row)
    with open(run.out / "sass_ops.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["label", "opcode", "count"])
        for op, n in sorted(ops.items(), key=lambda kv: -kv[1]):
            w.writerow([label, op, n])


def run_matmul(run, shapes):
    for m, n, k in shapes:
        shp = f"{m}x{n}x{k}"
        a = torch.rand(m, k, device="cuda") * 2 - 1
        b = torch.rand(k, n, device="cuda") * 2 - 1
        c = torch.full((m, n), float("nan"), device="cuda")
        grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)  # noqa: E731
        args = (a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
        ck = matmul_kernel[grid](*args)
        torch.cuda.synchronize()
        ok, ratio, tol = matmul_check(a, b, c)
        print(f"correctness triton_matmul {shp}: {'PASS' if ok else 'FAIL'} (max err/(|A||B|) {ratio:.3g}, "
              f"tol {tol:.3g})", flush=True)
        for cfg, t in matmul_kernel.configs_timings.items():
            tt = t if isinstance(t, (list, tuple)) else [t]
            run.csv("matmul_autotune.csv", {"M": m, "N": n, "K": k, "config": cfg_str(cfg),
                                            "autotune_ms_median": f"{tt[0]:.6f}",
                                            "chosen": int(cfg == matmul_kernel.best_config)})
        if not ok:
            continue
        cfg = matmul_kernel.best_config
        ci = compiled_info(ck)
        threads = 32 * ci["num_warps"]
        blocks = occupancy(ck, threads, ci["smem"])
        label = f"triton_matmul_{shp}"
        write_sass(run, "triton_matmul", label, cfg_str(cfg), ci["regs"], ci["spills"], ci["smem"], threads, blocks,
                   cubin_ops(ck))
        flops = 2.0 * m * n * k
        nbytes = 4.0 * (m * k + k * n + m * n)
        fn = lambda: matmul_kernel[grid](*args)  # noqa: E731
        st = run.timed(label, fn, flush=False)
        nprog = triton.cdiv(m, cfg.kwargs["BM"]) * triton.cdiv(n, cfg.kwargs["BN"])
        run.csv("matmul.csv", kfbench.result_row("triton_matmul", cfg_str(cfg), m, nbytes, flops, str(nprog),
                                                 str(threads), ci["regs"], ci["spills"], ci["smem"], st, "GFLOP/s",
                                                 {"M": m, "N": n, "K": k, "blocks_per_sm": blocks, "label": label}))
        print("RESULT", label, f"median_ms={st['median_ms']:.6f}", f"GFLOP/s={flops / st['median_ms'] / 1e6:.1f}",
              flush=True)
        t_est = st["median_ms"]
        label = f"torch_matmul_{shp}"
        st = run.timed(label, lambda: torch.matmul(a, b, out=c), flush=False)
        run.csv("matmul.csv", kfbench.result_row("torch_matmul_cublas", "allow_tf32=False", m, nbytes, flops, "-",
                                                 "-", -1, -1, -1, st, "GFLOP/s",
                                                 {"M": m, "N": n, "K": k, "blocks_per_sm": -1, "label": label}))
        print("RESULT", label, f"median_ms={st['median_ms']:.6f}", f"GFLOP/s={flops / st['median_ms'] / 1e6:.1f}",
              flush=True)
        del a, b, c
        torch.cuda.empty_cache()
        # Hand-written CUDA and cuBLAS through build/sgemm; warm-up sized to the sustain time.
        warm = max(10, math.ceil(SUSTAIN_S / (t_est * 1e-3)))
        for v in (0, 6, 5):
            label = f"sgemm_v{v}_{shp}"
            cmd = [str(ROOT / "build" / "sgemm"), str(m), "--ncols", str(n), "--k", str(k), "--version", str(v),
                   "--warmup", str(warm), "--reps", "100", "--csv", str(run.out / "cuda_sgemm.csv")]
            if v == 5:
                cmd += ["--cfg", "128x128x16x8x8"]
            run.mark("start", label)
            rc = subprocess.run(cmd).returncode
            run.mark("end", label)
            if rc != 0:
                print(f"RUN_FAIL rc={rc} label={label}", flush=True)


def run_ablation(run, shapes, configs):
    for m, n, k in shapes:
        shp = f"{m}x{n}x{k}"
        a = torch.rand(m, k, device="cuda") * 2 - 1
        b = torch.rand(k, n, device="cuda") * 2 - 1
        c = torch.empty((m, n), device="cuda")
        args = (a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
        for bm, bn, bk, w, s in configs:
            cfg = f"BM={bm},BN={bn},BK={bk},GROUP_M=8;num_warps={w};num_stages={s}"
            grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
            c.fill_(float("nan"))
            try:
                ck = matmul_kernel.fn[grid](*args, BM=bm, BN=bn, BK=bk, GROUP_M=8, num_warps=w, num_stages=s)
                torch.cuda.synchronize()
            except Exception as e:  # compile or launch failure (for example shared memory limit)
                print(f"ABLATION_FAIL {shp} {cfg}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
                continue
            ok, ratio, tol = matmul_check(a, b, c)
            print(f"correctness triton_matmul_fixed {shp} {cfg}: {'PASS' if ok else 'FAIL'} "
                  f"(max err/(|A||B|) {ratio:.3g}, tol {tol:.3g})", flush=True)
            if not ok:
                continue
            ci = compiled_info(ck)
            threads = 32 * w
            blocks = occupancy(ck, threads, ci["smem"])
            label = f"triton_fixed_{shp}_{bm}x{bn}x{bk}_w{w}_s{s}"
            write_sass(run, "triton_matmul_fixed", label, cfg, ci["regs"], ci["spills"], ci["smem"], threads, blocks,
                       cubin_ops(ck))
            fn = lambda: matmul_kernel.fn[grid](*args, BM=bm, BN=bn, BK=bk, GROUP_M=8,  # noqa: E731
                                                num_warps=w, num_stages=s)
            st = run.timed(label, fn, flush=False)
            flops = 2.0 * m * n * k
            run.csv("matmul_ablation.csv", kfbench.result_row(
                "triton_matmul_fixed", cfg, m, 4.0 * (m * k + k * n + m * n), flops, str(grid[0]), str(threads),
                ci["regs"], ci["spills"], ci["smem"], st, "GFLOP/s",
                {"M": m, "N": n, "K": k, "blocks_per_sm": blocks, "label": label}))
            print("RESULT", label, f"median_ms={st['median_ms']:.6f}", f"GFLOP/s={flops / st['median_ms'] / 1e6:.1f}",
                  f"regs={ci['regs']} spills={ci['spills']} smem={ci['smem']} blocks_per_sm={blocks}", flush=True)
        del a, b, c
        torch.cuda.empty_cache()


def softmax_check(y, ref):
    err = (y.double() - ref).abs()
    bad = (err > 1e-7 + 1e-5 * ref.abs()).sum().item()
    return bad == 0, (err / (ref.abs() + 1e-7)).max().item()


def run_softmax(run, cols_list):
    # CUDA SASS and resources, once per variant.
    cuda_ops = sass_ops(ROOT / "build" / "libkf_ml.so", match="softmax_rows")
    for cols in cols_list:
        rows = SOFTMAX_TOTAL // cols
        n = rows * cols
        torch.manual_seed(cols)
        x = torch.randn(rows, cols, device="cuda")
        y = torch.empty_like(x)
        ref = torch.softmax(x.double(), dim=-1)
        nbytes = 8.0 * n
        vec = cols % 4 == 0
        which = 0 if vec else 1
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        block = triton.next_power_of_2(cols)
        impls = [("cuda_softmax", lambda: libml.kf_softmax(x.data_ptr(), y.data_ptr(), rows, cols, stream)),
                 ("triton_softmax", lambda: softmax_kernel[(rows,)](y, x, cols, BLOCK=block))]
        for name, fn in impls:
            y.fill_(float("nan"))
            ret = fn()
            torch.cuda.synchronize()
            if name == "cuda_softmax" and ret != 0:
                raise SystemExit(f"cuda_softmax launch error {ret}")
            ok, worst = softmax_check(y, ref)
            print(f"correctness {name} rows={rows} cols={cols}: {'PASS' if ok else 'FAIL'} "
                  f"(max err/(|ref|+1e-7) {worst:.3g})", flush=True)
            if not ok:
                continue
            label = f"{name}_{cols}"
            if name == "cuda_softmax":
                r, lb, sm, bps = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
                assert libml.kf_ml_attrs(which, ctypes.byref(r), ctypes.byref(lb), ctypes.byref(sm)) == 0
                assert libml.kf_ml_occupancy(which, cols, ctypes.byref(bps)) == 0
                regs, spills, smem, blocks = r.value, lb.value, sm.value, bps.value
                threads = libml.kf_ml_threads(cols)
                variant = "float4" if vec else "scalar"
                fname = [f for f in cuda_ops if f"softmax_rows<{'true' if vec else 'false'}>" in f][0]
                ops = cuda_ops[fname]
            else:
                ck = fn()  # returns the compiled kernel after tuning
                ci = compiled_info(ck)
                regs, spills, smem = ci["regs"], ci["spills"], ci["smem"]
                threads = 32 * ci["num_warps"]
                blocks = occupancy(ck, threads, smem)
                variant = cfg_str(softmax_kernel.best_config) + f";BLOCK={block}"
                ops = cubin_ops(ck)
            write_sass(run, name, label, variant, regs, spills, smem, threads, blocks, ops)
            st = run.timed(label, fn, flush=True)
            gbs = nbytes / (st["median_ms"] * 1e-3) / 1e9
            run.csv("softmax.csv", kfbench.result_row(
                name, variant, n, nbytes, 0, str(rows), str(threads), regs, spills, smem, st, "GB/s",
                {"rows": rows, "cols": cols, "blocks_per_sm": blocks, "pct_of_1530": f"{100 * gbs / 1530:.1f}",
                 "label": label}))
            print("RESULT", label, f"median_ms={st['median_ms']:.6f}", f"GB/s={gbs:.1f}", f"regs={regs}",
                  f"blocks_per_sm={blocks}", flush=True)
        del x, y, ref
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results/stage10")
    p.add_argument("--quick", action="store_true", help="small test set, for checking the driver")
    p.add_argument("--parts", nargs="*", default=["matmul", "ablation", "softmax"])
    a = p.parse_args()
    outdir = Path(a.outdir)
    run = Run(outdir if outdir.is_absolute() else ROOT / outdir)
    print("run info:", run.info, flush=True)
    shapes, ab_shapes, ab_cfgs, cols = MATMUL_SHAPES, ABLATION_SHAPES, ABLATION, SOFTMAX_COLS
    if a.quick:
        shapes, ab_shapes, ab_cfgs, cols = [(1000,) * 3, (777, 1111, 333)], [(1024,) * 3], ABLATION[:2], [1001, 4096]
    if "matmul" in a.parts:
        run_matmul(run, shapes)
    if "ablation" in a.parts:
        run_ablation(run, ab_shapes, ab_cfgs)
    if "softmax" in a.parts:
        run_softmax(run, cols)


if __name__ == "__main__":
    main()
