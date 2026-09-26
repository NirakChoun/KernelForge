#!/bin/bash
#SBATCH --job-name=kf
#SBATCH --account=publicgrp
#SBATCH --partition=high
#SBATCH --gpus=6000_blackwell:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd ~/KernelForge
source scripts/modules.sh
scripts/env_report.sh
"$@"
