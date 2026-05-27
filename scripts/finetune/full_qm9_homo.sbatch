#!/bin/bash
# Phase 1.5 full reproduction: the paper's CT-SCD QM9 HOMO finetune.
# Default: 300k steps on 1 A100 (shared-QoS billing per GPU-hour, not whole-node).
# Probe extrapolation puts this at ~10 GPU-h on Perlmutter A100s; -t 12:00:00 has
# margin without crossing the 24h shared cap.
#
# Usage:
#   sbatch scripts/finetune/full_qm9_homo.sbatch
#   sbatch --export=NOISE_IN_LOADER=True scripts/finetune/full_qm9_homo.sbatch  # kernel-off variant
#
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH -t 12:00:00
#SBATCH -J scd_qm9_full
#SBATCH -o /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/full_%j.out
#SBATCH -e /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/full_%j.err

set -uo pipefail
cd /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms
export PATH=/global/homes/l/luisc440/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1

NOISE_IN_LOADER=${NOISE_IN_LOADER:-False}
EXTRA=${EXTRA:-}
JOB_TAG=${JOB:-full_kernel_$( [ "$NOISE_IN_LOADER" = "True" ] && echo off || echo on )}

LOG=$SCRATCH/SCD_data/finetune_runs/${SLURM_JOB_ID}_${JOB_TAG}
mkdir -p "$LOG"
echo "[host]   $(hostname)"   | tee    "$LOG/banner.txt"
echo "[date]   $(date)"       | tee -a "$LOG/banner.txt"
echo "[jobid]  $SLURM_JOB_ID" | tee -a "$LOG/banner.txt"
echo "[noise_in_loader] $NOISE_IN_LOADER" | tee -a "$LOG/banner.txt"
echo "[job_tag] $JOB_TAG"     | tee -a "$LOG/banner.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a "$LOG/banner.txt"

srun --cpu-bind=cores pixi run python -u train.py \
    --conf configs/finetune_qm9.yaml \
    --load-hf ct-scd-pcq \
    --job-id "$JOB_TAG" \
    --log-dir "$LOG/experiments" \
    --noise_in_loader "$NOISE_IN_LOADER" \
    --no-wandb-resume \
    --wandb-notes "full run (noise_in_loader=$NOISE_IN_LOADER extra=$EXTRA)" \
    $EXTRA 2>&1 | tee "$LOG/train.log"

echo "[done]   $(date)" | tee -a "$LOG/banner.txt"
