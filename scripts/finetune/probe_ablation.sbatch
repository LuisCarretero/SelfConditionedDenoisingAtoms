#!/bin/bash
# Phase 2 ablation probe: short (default 3000-step) finetune for A/B-ing one
# Phase 2.2 lever at a time vs the Phase 1 baseline. Same wiring as
# probe_qm9_homo.sbatch; the diff is `EXTRA` is the lever string and `JOB`
# carries a short tag so W&B runs are distinguishable.
#
# Usage examples:
#   sbatch --export=ALL,JOB=p2_tf32,EXTRA="--tf32 True" scripts/finetune/probe_ablation.sbatch
#   sbatch --export=ALL,JOB=p2_bf16,EXTRA="--precision 16" scripts/finetune/probe_ablation.sbatch
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
#SBATCH -J scd_probe_ab
#SBATCH -o /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/probeab_%j.out
#SBATCH -e /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/probeab_%j.err

set -uo pipefail
cd /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms
export PATH=/global/homes/l/luisc440/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1

PROBE_STEPS=${PROBE_STEPS:-3000}
JOB_TAG=${JOB:?JOB tag required (e.g. p2_tf32, p2_bf16)}
EXTRA=${EXTRA:-}

LOG=$SCRATCH/SCD_data/finetune_runs/${SLURM_JOB_ID}_${JOB_TAG}
mkdir -p "$LOG"
echo "[host]   $(hostname)"   | tee    "$LOG/banner.txt"
echo "[date]   $(date)"       | tee -a "$LOG/banner.txt"
echo "[jobid]  $SLURM_JOB_ID" | tee -a "$LOG/banner.txt"
echo "[steps]  $PROBE_STEPS"  | tee -a "$LOG/banner.txt"
echo "[job_tag] $JOB_TAG"     | tee -a "$LOG/banner.txt"
echo "[extra]  $EXTRA"        | tee -a "$LOG/banner.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a "$LOG/banner.txt"

srun --cpu-bind=cores pixi run python -u train.py \
    --conf configs/finetune_qm9.yaml \
    --load-hf ct-scd-pcq \
    --job-id "$JOB_TAG" \
    --log-dir "$LOG/experiments" \
    --num-steps "$PROBE_STEPS" \
    --no-wandb-resume \
    --wandb-notes "phase2 probe (steps=$PROBE_STEPS extra=$EXTRA)" \
    $EXTRA 2>&1 | tee "$LOG/train.log"

echo "[done]   $(date)" | tee -a "$LOG/banner.txt"
