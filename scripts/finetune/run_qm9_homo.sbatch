#!/bin/bash
# QM9 HOMO finetune from a HF-loaded CT-SCD-PCQ checkpoint. Used both for the
# timing-probe (LimitRun/--num-steps override) and the full 300k-step
# reproduction. Targets Perlmutter A100 nodes.
#
# Usage (timing probe, 30 min on 1 GPU, premium QoS):
#   sbatch --export=LOAD_HF=ct-scd-pcq,EXTRA="--max_time_minutes 30",JOB=probe \
#          --gpus=1 --ntasks-per-node=1 -t 00:45:00 -q premium \
#          scripts/finetune/run_qm9_homo.sbatch
#
# Usage (full 300k-step finetune, 1 A100, regular QoS — picks up via LimitRun
# chain across submissions if you cap wall-time per submission):
#   sbatch --export=LOAD_HF=ct-scd-pcq,JOB=qm9_homo_full \
#          --gpus=1 --ntasks-per-node=1 -t 24:00:00 -q regular \
#          scripts/finetune/run_qm9_homo.sbatch
#
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH -J scd_qm9_homo
#SBATCH -o /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/qm9_homo_%j.out
#SBATCH -e /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/qm9_homo_%j.err

set -uo pipefail
cd /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms
export PATH=/global/homes/l/luisc440/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1

LOG=$SCRATCH/SCD_data/finetune_runs/${SLURM_JOB_ID}
mkdir -p "$LOG"
echo "[host]   $(hostname)"   | tee    "$LOG/banner.txt"
echo "[date]   $(date)"       | tee -a "$LOG/banner.txt"
echo "[jobid]  $SLURM_JOB_ID" | tee -a "$LOG/banner.txt"
echo "[ntasks] $SLURM_NTASKS" | tee -a "$LOG/banner.txt"
echo "[load_hf] ${LOAD_HF:-(none)}" | tee -a "$LOG/banner.txt"
echo "[extra]  ${EXTRA:-}"    | tee -a "$LOG/banner.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a "$LOG/banner.txt"

JOB_ID_TAG=${JOB:-qm9_homo}

srun --cpu-bind=cores pixi run python -u train.py \
    --conf configs/finetune_qm9.yaml \
    --load-hf "${LOAD_HF:-ct-scd-pcq}" \
    --job-id "$JOB_ID_TAG" \
    --log-dir "$LOG/experiments" \
    ${EXTRA:-} 2>&1 | tee "$LOG/train.log"

echo "[done]   $(date)" | tee -a "$LOG/banner.txt"
