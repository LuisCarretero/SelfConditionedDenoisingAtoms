# Restart note for the next agent

Snapshot: 2026-05-26 ~22:30 PDT. The previous agent landed Phase 2.5 (apples-to-apples GPU-h machinery + whole-epochs config + profile sbatch), resolved the `--load-model` vs `--load-hf` open question (data-side / loss-affecting → use `--load-model`), and **submitted the three canonical full runs packed onto one premium 4-GPU node**.

## TL;DR

- **Slurm job 53465532** (one sbatch, three lanes): `full_kernel_on`, `full_tf32`, `full_tf32_compile`. Live in `$SCRATCH/SCD_data/finetune_runs/53465532_canonical/<tag>/train.log` once running. Each lane is single-GPU (`CUDA_VISIBLE_DEVICES=0/1/2`), `--load-model <local-ckpt>`, 349 whole epochs. Walltime allocation 11h; the slowest arm (baseline) extrapolates ~9.5h.
- Phase 2.5 commits: `c0789a1` (TrainTiming wall-h + tests), `e474f9e` (refactor for num_steps=-1), `e0effc0` (whole-epochs config), `e01aa6c` (profile sbatch).
- Resubmission commits: `153455e` (new packed sbatch), `e64f8c5` (worklog).
- `--load-model` decision rationale + per-commit summary now live in `docs/finetune-speedup-worklog.md` under "Phase 2.5" + "load-model vs load-hf".

## What's next (in order)

### Step 1 — Wait for 53465532, then harvest

Poll: `squeue -j 53465532` and `sacct -j 53465532 -o JobID,State,Elapsed,ExitCode -X -n`. Each lane completes independently inside the same sbatch; tail `$SCRATCH/SCD_data/finetune_runs/53465532_canonical/<tag>/train.log` per lane.

When all three are done (sbatch returns COMPLETED), for each lane:
1. `test_loss` from the bottom of `train.log` (Lightning prints it at the end of `trainer.test()`). Multiply ×1000 → meV.
2. Steady-state step time and **both** GPU-h numbers from the **last** `[TrainTiming] epoch …` line (NOT the first — compile/cuDNN warmup spike inflates epochs 0–1 for the compile lane).
3. Update the `_pending_` rows in `docs/finetune-speedup-worklog.md` under "Pending full runs (packed …)". Report `X GPU-h (compute) / Y GPU-h (wall) vs paper's 46`. **Y** is the apples-to-apples number.
4. Gate per acceptance criteria in the worklog: baseline lane MAE ≤ 14 meV; Phase 2 candidates within ±0.5 meV of the baseline lane. Anything outside → reject + flag.
5. Commit each worklog update (`docs/finetune-speedup-worklog.md` only).

### Step 2 — (Optional) profile run if val/IO breakdown is wanted

`scripts/finetune/profile_run.sbatch` packs two lanes (baseline, tf32+compile) on one premium 4-GPU node with `--profile-trace-dir` enabling PL's PyTorchProfiler. Captures 200 train steps post-warmup + one val epoch + one epoch boundary. Outputs trace files + a summary `.txt` next to the traces. Only worth running if the canonical-run numbers raise a question that needs op-level granularity (e.g. val share looks unexpectedly large under compile).

### Step 3 — Report

Update `docs/finetune-speedup.md` ONLY if the acceptance verdict changes — e.g. if a lever is rejected. Otherwise the worklog row is the final record. The Phase 2 winner candidate going into the report is `--tf32 True --torch-compile True` (probe data: −28.3% step time, predicted (1−0.108)(1−0.189)=0.723 ⇒ −27.7% matches).

## What changed since the previous restart note

1. **`TrainTiming` now reports wall-clock-honest GPU-h.** Second metric `train/extrapolated_wall_h_total` = compute + per-epoch-overhead × planned-epochs + post-fit test. Per-epoch overhead is measured as `full_epoch_wall − step_s × steps_per_epoch` (captures val + reload + ckpt). Tests in `tests/test_traintiming.py` guard the math (`_extrapolate`, `_planned_train_epochs`, `_effective_total_steps`).
2. **Config switched to whole epochs.** `configs/finetune_qm9.yaml`: `num_epochs: 349, num_steps: -1` (= 300,140 steps via 860 batches/epoch). The TrainTiming code now derives effective step count from `min(step_gate, epoch_gate × steps_per_epoch)`, so `num_steps=-1` no longer produces a negative compute-h.
3. **`--load-model` is now the paper-faithful path.** `models/ET_models/scd_model.py:312-326` uses `model.mean/std` inside the forward pass (atom outputs × std, sum, + mean), so they're data-side / loss-affecting. The `ct-scd-pcq` pretrain has no y-target → checkpoint stores `mean=0, std=1`. `--load-model` keeps those; `--load-hf` overwrites with QM9 HOMO data stats (≈−0.4/0.04), shrinking the final-layer effective LR ~25×. Paper command uses `--load-model`; canonical sbatch now matches.

## Useful runtime facts

- **Cancelled jobs** 53461802 / 53462549 / 53462886 (the previous gpu_regular submissions) are CANCELLED in sacct. The packed resubmission 53465532 supersedes all three.
- Local checkpoint path: `experiments/models--Ty-Perez--ct-scd-pcq/snapshots/fcd4353d3b9f90eb38e08ccc0050cea02623d85f/last.ckpt`. The sbatch checks this exists and exits cleanly if not.
- `gpu_premium`: full-node, 2× factor, 48h max — fast queue. The packed pattern (3 lanes × 1 GPU each) makes premium roughly cost-equivalent to three sequential `gpu_shared` allocations while finishing all three in parallel.
- Pixi at `/global/homes/l/luisc440/.pixi/bin/pixi`. `.pixi/` is a symlink to `$SCRATCH/SCD_data/.pixi`.
- W&B project: `SCD-finetune-speedup`.

## Things still deferred

- **bf16-mixed**: needs targeted `torch.amp.autocast(enabled=False)` around `noise_normalizer` (an `AccumulatedNormalization` with side-effectful `update_statistics()` that mixes fp32 buffers with bf16 autocast tensors). Lane 3 of 53465532 is idle — a cheap retry slot if the autocast fix lands while the run is in flight; otherwise leave it alone.
- **Phase 2.3** (fused attention, DDP scaling sweep) — only worth it if 2.2 winners at full scale leave headroom AND the profile shows where to look.
- **DDP×4 with per-GPU batch=128** violates the "same effective batch size" invariant (would go from 128 → 512). Per-GPU batch=32 + DDP×4 would preserve effective=128 but lose arithmetic intensity. Don't enable without explicit scope change.
