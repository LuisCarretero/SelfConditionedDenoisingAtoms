#!/bin/bash
# DDP throughput probe: 4-GPU DDP at batch_size=128 per GPU (effective batch
# 512), matching the paper's pretrain hparam convention (Hyperparameters.tex
# tab:backbone_hparams: CT-SCD-PCQ = 4× A100, batch/GPU=128, effective=512).
# The QM9 finetune hparam table omits the # GPUs row but the repo's config
# defaults (distributed_backend=ddp, ngpus=-1) would auto-enable DDP on any
# multi-GPU node — so this probe answers: did the paper's 46 GPU-h come from
# 4-GPU DDP at effective batch 512, rather than single-GPU at 128 as we
# assumed? Compare step_time_ms × 4 GPUs vs our 1×A100 numbers.
#
# Stays on `interactive` QoS (full 4-GPU node, ~15s queue, ≤4h cap; probe
# wall ≈ 10–15 min). EXTRA defaults to none — bare baseline matches what
# 2023-era torch/PL would have produced. Set EXTRA="--tf32 True --torch-compile True"
# for a Phase-2-winner DDP variant if needed later.
#
# Usage:
#   sbatch scripts/finetune/probe_ddp4.sbatch
#   sbatch --export=ALL,EXTRA="--tf32 True --torch-compile True",JOB=probe_ddp4_tf32_compile \
#          scripts/finetune/probe_ddp4.sbatch
#
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -q interactive
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH -t 00:30:00
#SBATCH -J scd_probe_ddp4
#SBATCH -o /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/probeddp4_%j.out
#SBATCH -e /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/probeddp4_%j.err

set -uo pipefail
cd /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms
export PATH=/global/homes/l/luisc440/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1

PROBE_STEPS=${PROBE_STEPS:-3000}
JOB_TAG=${JOB:-probe_ddp4_bs128}
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

# All 4 GPUs visible → PL sees ngpus=-1 → world_size=4. With 4 srun tasks,
# each PL DDP rank pins to one GPU via SLURM_LOCALID.
srun --cpu-bind=cores pixi run python -u train.py \
    --conf configs/finetune_qm9.yaml \
    --load-hf ct-scd-pcq \
    --job-id "$JOB_TAG" \
    --log-dir "$LOG/experiments" \
    --num-steps "$PROBE_STEPS" \
    --no-wandb-resume \
    --wandb-notes "ddp4 throughput probe (steps=$PROBE_STEPS extra=$EXTRA effective_bs=512)" \
    $EXTRA 2>&1 | tee "$LOG/train.log"

echo "[done]   $(date)" | tee -a "$LOG/banner.txt"
