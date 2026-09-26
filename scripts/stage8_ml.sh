#!/bin/bash
# Stage 8: softmax and RMSNorm in CUDA (build/libkf_ml.so), Triton, and PyTorch.
# Submit: sbatch --mem=32G --time=01:00:00 scripts/run_gpu.sh scripts/stage8_ml.sh
set -euo pipefail
cd python && ../.venv/bin/python stage8_ml.py --outdir results/stage8
