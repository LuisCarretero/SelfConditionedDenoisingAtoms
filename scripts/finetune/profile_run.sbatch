#!/bin/bash
# Phase 2.5: PyTorch-profile the canonical finetune setup.
# Two lanes on one full-node allocation (premium QoS bills 4 GPUs anyway):
#   lane 0: baseline (kernel-on, no tf32, no compile)
#   lane 1: tf32 + torch-compile (Phase 2 winner candidate)
# Each lane runs ~3 epochs (limit) so the profiler captures: 200 train steps
# post-warmup, one full val epoch (val_interval=1), and one epoch boundary
# (reload + checkpoint write). Compare lanes to see how val's share of wall
# time shifts under the speedup — informs whether further work should target
# train or val.
#
# Usage:
#   sbatch scripts/finetune/profile_run.sbatch
#
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH -t 01:00:00
#SBATCH -J scd_profile
#SBATCH -o /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/profile_%j.out
#SBATCH -e /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/profile_%j.err

set -uo pipefail
cd /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms
export PATH=/global/homes/l/luisc440/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1

ROOT=$SCRATCH/SCD_data/finetune_runs/${SLURM_JOB_ID}_profile
mkdir -p "$ROOT"
echo "[host]  $(hostname)"   | tee    "$ROOT/banner.txt"
echo "[date]  $(date)"       | tee -a "$ROOT/banner.txt"
echo "[jobid] $SLURM_JOB_ID" | tee -a "$ROOT/banner.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a "$ROOT/banner.txt"

# Lane args: (GPU index, tag, extra flags).
LANES=(
  "0|prof_baseline|"
  "1|prof_tf32_compile|--tf32 True --torch-compile True"
)

for entry in "${LANES[@]}"; do
  IFS='|' read -r gpu tag extra <<< "$entry"
  LANE_DIR="$ROOT/$tag"
  mkdir -p "$LANE_DIR/traces" "$LANE_DIR/experiments"
  echo "[launch] GPU $gpu tag=$tag extra='$extra'" | tee -a "$ROOT/banner.txt"

  CUDA_VISIBLE_DEVICES=$gpu pixi run python -u train.py \
      --conf configs/finetune_qm9.yaml \
      --load-hf ct-scd-pcq \
      --job-id "$tag" \
      --log-dir "$LANE_DIR/experiments" \
      --num-epochs 3 \
      --num-steps -1 \
      --profile-trace-dir "$LANE_DIR/traces" \
      --no-wandb-resume \
      --wandb-notes "phase2.5 profile lane=$tag" \
      $extra > "$LANE_DIR/train.log" 2>&1 &
done

wait
echo "[done]  $(date)" | tee -a "$ROOT/banner.txt"

# Quick summary: PL's PyTorchProfiler writes the average-table .txt next to
# the trace files. Print the top ops for each lane so the slurm .out is
# self-contained.
for entry in "${LANES[@]}"; do
  IFS='|' read -r gpu tag extra <<< "$entry"
  echo "============ $tag (GPU $gpu) ============"
  head -50 "$ROOT/$tag"/traces/*.txt 2>/dev/null || echo "(no profile txt)"
done | tee "$ROOT/profile_summary.txt"
