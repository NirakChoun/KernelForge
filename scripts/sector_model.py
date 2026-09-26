"""Stage 1: useful bytes vs estimated DRAM bytes under a 32-byte sector model.

Model: DRAM moves whole 32-byte sectors (8 uint32 elements). A sector is fetched once per
time it is touched, unless the pattern touches it again while it is still in L2; the
model assumes no L2 reuse between strided passes (each pass spans the whole 1 GiB input,
8x the 128 MiB L2) and full reuse between neighbouring warps of one pass.
"""
import csv, os
os.chdir(os.path.expanduser("~/KernelForge"))
rows = list(csv.DictReader(open("results/stage1/patterns.csv")))
out = []
for r in rows:
    n = int(r["n"]); k = r["kernel"]; v = r["variant"]
    write = 4 * n
    if k == "contiguous":
        read = 4 * n
    elif k == "strided":
        s = int(v.split("=")[1])
        # stride s: s passes; each pass touches every sector while s <= 8
        # (n/8 sectors per pass), one sector per element once s >= 8.
        read = 32 * (s * n // 8) if s <= 8 else 32 * n
    elif k == "gather":
        read = 4 * n + 32 * n  # index stream + one sector per random element
    elif k == "offset":
        read = 4 * n + 32  # at most one extra sector at the end
    elif k == "aos_x":
        read = 16 * n  # every 32-byte sector holds 2 structs, all fetched
    elif k == "soa_x":
        read = 4 * n
    elif k in ("aos_sum", "soa_sum"):
        read = 16 * n
    model = read + write
    t = float(r["median_ms"]) * 1e-3
    out.append(dict(kernel=k, variant=v, flush=r["l2_flush"], useful=int(float(r["bytes"])), model=model,
                    ratio=model / float(r["bytes"]), useful_gbs=float(r["metric_value"]),
                    model_gbs=model / t / 1e9, median_ms=r["median_ms"]))
with open("results/stage1/sector_model.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    for o in out:
        w.writerow({**o, "ratio": f"{o['ratio']:.3f}", "model_gbs": f"{o['model_gbs']:.1f}"})
for o in out:
    if o["kernel"] != "offset" or o["variant"] in ("offset=0", "offset=1", "offset=32"):
        print(o["kernel"], o["variant"], o["flush"], o["useful"], o["model"], f"{o['ratio']:.3f}",
              o["useful_gbs"], f"{o['model_gbs']:.1f}")
