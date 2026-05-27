#!/bin/bash
# Build the TorchMD-Net neighbor CUDA kernel on a GPU node, then run the
# env smoketest. nvcc + matching CUDA libs are only available on compute,
# which is why this can't be a `pixi install` step.
#
# Usage:  sbatch scripts/setup/build_kernel_and_smoketest.sbatch
#
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --gpu-bind=none
#SBATCH -t 00:30:00
#SBATCH -J scd_setup
#SBATCH -o /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/setup_%j.out
#SBATCH -e /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/setup_%j.err

set -uo pipefail
cd /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms
export PATH=/global/homes/l/luisc440/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1

echo "[host] $(hostname)"
echo "[date] $(date)"
nvidia-smi --query-gpu=index,name --format=csv

echo ""
echo "=== build TorchMD-Net neighbor kernel ==="
( cd models/ET_models && pixi run python -u setup.py build_ext --inplace ) 2>&1 | tail -40

echo ""
echo "=== smoketest ==="
pixi run smoketest 2>&1 | tail -60

echo ""
echo "[done] $(date)"
