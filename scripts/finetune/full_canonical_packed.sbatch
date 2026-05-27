#!/bin/bash
# Phase 2.5 → Phase 2 verdict: canonical full QM9 HOMO finetune for the three
# arms (baseline, tf32, tf32+compile) packed onto ONE premium 4-GPU node.
# Premium bills the whole node either way; CUDA_VISIBLE_DEVICES lanes give us
# three independent single-GPU runs in parallel. Lane 3 idle.
#
# Each lane runs 349 whole epochs (paper's 300k steps ≈ 348.84 epochs;
# whole-epochs config keeps the train/val ratio stable across arms). With
# probe-extrapolated step times ~80–112 ms × 860 batches/epoch × 349 epochs,
# the slowest arm (baseline) lands at ~9.5h wall — comfortably under the 11h
# allocation.
#
# Uses --load-model (paper-faithful path): the pretrained ct-scd-pcq
# checkpoint's mean/std are 0/1 (pretrain has no y-target). Going through
# --load-hf would *overwrite* with QM9 HOMO's data-mean/std and silently
# shrink the final-layer effective LR by ~25× via the `x * std` rescaling in
# scd_model.py — different training dynamics.
#
# Usage:
#   sbatch scripts/finetune/full_canonical_packed.sbatch
#
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --gpu-bind=none
#SBATCH -t 11:00:00
#SBATCH -J scd_full_packed
#SBATCH -o /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/full_packed_%j.out
#SBATCH -e /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms/slurm_logs/full_packed_%j.err

set -uo pipefail
cd /global/u1/l/luisc440/workspace/SelfConditionedDenoisingAtoms
export PATH=/global/homes/l/luisc440/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1

CKPT=experiments/models--Ty-Perez--ct-scd-pcq/snapshots/fcd4353d3b9f90eb38e08ccc0050cea02623d85f/last.ckpt
if [ ! -f "$CKPT" ]; then
  echo "ERROR: pretrained checkpoint not found at $CKPT — run the hf_hub_download cell in examples.ipynb first." >&2
  exit 1
fi

ROOT=$SCRATCH/SCD_data/finetune_runs/${SLURM_JOB_ID}_canonical
mkdir -p "$ROOT"
echo "[host]  $(hostname)"   | tee    "$ROOT/banner.txt"
echo "[date]  $(date)"       | tee -a "$ROOT/banner.txt"
echo "[jobid] $SLURM_JOB_ID" | tee -a "$ROOT/banner.txt"
echo "[ckpt]  $CKPT"         | tee -a "$ROOT/banner.txt"
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a "$ROOT/banner.txt"

# Lane args: (GPU index | tag | extra flags). Lane 3 left idle.
LANES=(
  "0|full_kernel_on|"
  "1|full_tf32|--tf32 True"
  "2|full_tf32_compile|--tf32 True --torch-compile True"
)

for entry in "${LANES[@]}"; do
  IFS='|' read -r gpu tag extra <<< "$entry"
  LANE_DIR="$ROOT/$tag"
  mkdir -p "$LANE_DIR/experiments"
  echo "[launch] GPU $gpu tag=$tag extra='$extra'" | tee -a "$ROOT/banner.txt"

  # Distinct MASTER_PORT per lane — PL still spins up a DDP master with one
  # GPU under distributed_backend=ddp, and three processes on the same host
  # otherwise race for the same default port (EADDRINUSE in 2/3 lanes).
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((29500 + gpu)) pixi run python -u train.py \
      --conf configs/finetune_qm9.yaml \
      --load-model "$CKPT" \
      --job-id "$tag" \
      --log-dir "$LANE_DIR/experiments" \
      --num-epochs 349 \
      --num-steps -1 \
      --no-wandb-resume \
      --wandb-notes "canonical full run (load-model paper-faithful) extra='$extra'" \
      $extra > "$LANE_DIR/train.log" 2>&1 &
done

wait
echo "[done]  $(date)" | tee -a "$ROOT/banner.txt"

# Slurm .out gets a quick summary so the run is self-describing.
for entry in "${LANES[@]}"; do
  IFS='|' read -r gpu tag extra <<< "$entry"
  echo "============ $tag (GPU $gpu) ============"
  grep -E "test_loss|TrainTiming" "$ROOT/$tag"/train.log | tail -5 || echo "(no summary lines yet)"
done | tee "$ROOT/canonical_summary.txt"
