#!/bin/bash
# Traces one command with Nsight Systems (CUDA API and kernels only) and writes the
# per-kernel duration summary as CSV. The .nsys-rep stays in $TMPDIR and is not kept.
# Run inside a GPU job: sbatch scripts/run_gpu.sh scripts/nsys_kern_sum.sh <out.csv> <cmd...>
set -euo pipefail
out=$1
shift
tmp=$(mktemp -d)
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o "$tmp/rep" "$@"
mkdir -p "$(dirname "$out")"
nsys stats --quiet --report cuda_gpu_kern_sum --format csv --output "$tmp/sum" "$tmp/rep.nsys-rep"
cp "$tmp"/sum*cuda_gpu_kern_sum*.csv "$out"
cat "$out"
rm -rf "$tmp"
