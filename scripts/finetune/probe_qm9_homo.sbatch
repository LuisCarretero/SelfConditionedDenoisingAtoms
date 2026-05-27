#!/bin/bash
# Phase 1.4 timing probe: short finetune (default 3000 steps) to extrapolate
# GPU-h/task for the full 300k. We deliberately do NOT touch lr_cosine_length —
# the schedule remains the paper's 300k cosine, just truncated. MAE from this
# run is meaningless (too few steps); the only signal we want is step_time
# stability post-warmup.
#
# Usage:
#   sbatch scripts/finetune/probe_qm9_homo.sbatch              # kernel-on probe
#   sbatch --export=NOISE_IN_LOADER=True scripts/finetune/probe_qm9_homo.sbatch  # kernel-off A/B
#
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH -t 00:45:00
#SBATCH -J scd_probe
#SBATCH -o /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/probe_%j.out
#SBATCH -e /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/probe_%j.err

set -uo pipefail
cd /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms
export PATH=/global/homes/l/luisc440/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1

PROBE_STEPS=${PROBE_STEPS:-3000}
NOISE_IN_LOADER=${NOISE_IN_LOADER:-False}
JOB_TAG=${JOB:-probe_kernel_$( [ "$NOISE_IN_LOADER" = "True" ] && echo off || echo on )}

LOG=$SCRATCH/SCD_data/finetune_runs/${SLURM_JOB_ID}_${JOB_TAG}
mkdir -p "$LOG"
echo "[host]   $(hostname)"   | tee    "$LOG/banner.txt"
echo "[date]   $(date)"       | tee -a "$LOG/banner.txt"
echo "[jobid]  $SLURM_JOB_ID" | tee -a "$LOG/banner.txt"
echo "[steps]  $PROBE_STEPS"  | tee -a "$LOG/banner.txt"
echo "[noise_in_loader] $NOISE_IN_LOADER" | tee -a "$LOG/banner.txt"
echo "[job_tag] $JOB_TAG"     | tee -a "$LOG/banner.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a "$LOG/banner.txt"

srun --cpu-bind=cores pixi run python -u train.py \
    --conf configs/finetune_qm9.yaml \
    --load-hf ct-scd-pcq \
    --job-id "$JOB_TAG" \
    --log-dir "$LOG/experiments" \
    --num-steps "$PROBE_STEPS" \
    --noise_in_loader "$NOISE_IN_LOADER" \
    --no-wandb-resume \
    --wandb-notes "phase1 timing probe (steps=$PROBE_STEPS noise_in_loader=$NOISE_IN_LOADER)" \
    2>&1 | tee "$LOG/train.log"

echo "[done]   $(date)" | tee -a "$LOG/banner.txt"
