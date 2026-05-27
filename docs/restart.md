# Restart note for the next agent

You're picking up the SCD finetune-speedup work mid-flight. Snapshot taken 2026-05-26 ~22:00 PDT, after the previous agent reviewed setup vs paper, identified an apples-vs-oranges issue in GPU-h reporting, and cancelled the three pending full runs so the next iteration can include the fixes.

## TL;DR

- Phase 0 (wiring) + Phase 1 probes + Phase 2 probes → done. **Winner so far: `--tf32 True --torch-compile True`, −28.3% step time** at the 3000-step probe, MAE preserved.
- Three full runs (`full_kernel_on`, `full_tf32`, `full_tf32_compile`) were sitting in `gpu_regular` PENDING. **Cancelled** (jobids 53461802 / 53462549 / 53462886). They were going to ship before Phase 2.5 fixes landed; would have produced strictly-undercounted GPU-h numbers and a partial-epoch tail. Better to resubmit cleanly with the fixes.
- **Phase 2.5 added to plan** (see `docs/finetune-speedup.md`): wall-clock-honest GPU-h, val/train profiler, whole-epochs config. These should land *before* the canonical full runs.
- **CLAUDE.md updated** with the "use the whole node" rule (full-node QoS bills 4 GPUs regardless of `--gpus=N`; use `CUDA_VISIBLE_DEVICES` lanes to pack ablations onto one node).

Full numbers + per-run W&B IDs: [`finetune-speedup-worklog.md`](finetune-speedup-worklog.md). Plan / scope / acceptance: [`finetune-speedup.md`](finetune-speedup.md).

## What changed since the previous restart note

1. **Setup vs paper audit (passed).** Every dynamics-affecting hyperparameter in `configs/finetune_qm9.yaml` matches `docs/paper_ref-SCD/Hyperparameters.tex` `tab:qm9_finetune_hparams`. Backbone, batch, LR, warmup, β1/β2, weight decay, DropPath schedule, noise σ, denoise weight, head aggregation, atomref off, EMA off — all match. Pretrained checkpoint (`ct-scd-pcq`) is the right one. Only deliberate deviations are `--tf32 True` and `--torch-compile True` (Phase 2 levers).
2. **Apples-to-apples GPU-h gap identified.** Our `train/extrapolated_gpu_h_total` = `step_time × num_steps × world_size / 3600`. The paper's 46 GPU-h is end-to-end wall-clock — includes ~349 val epochs (val_interval=1 default), the post-fit `trainer.test()`, sanity-val, and `reload_dataloaders_every_n_epochs=1` overhead. Rough estimate: ~10–25 min of val across a full run. So our current 9.4 GPU-h extrapolation undercounts true wall-clock by ~5–10%, and any "X GPU-h vs 46" claim is currently apples-vs-oranges. Phase 2.5 fixes this.
3. **`--load-model` vs `--load-hf` divergence.** The paper's reference command uses `--load-model <path>`, our sbatch uses `--load-hf ct-scd-pcq`. Both load the identical pretrained weights file from disk. But the code paths in `models/trainer.py:76-89` differ in how `mean`/`std` are forwarded to `load_model`: the `--load-hf` (pretrained_model) branch passes `data.mean`/`data.std` from the QM9 DataModule; the `--load-model` branch silently drops them, so the model is left with whatever `create_model` initialized. Impact depends on whether `standardize: true` baking uses model.mean/std on the **data side** (loss-affecting) or the **prediction side** (only de-standardizes outputs). Worth confirming before the full-run resubmission — see "Open question for the next agent" below.

## What's next (in order)

The user's directive was: build the apples-to-apples machinery first, *then* run the canonical full set with it. So Phase 2.5 work goes before queue resubmission.

### Step 1 — Implement Phase 2.5 changes

See `docs/finetune-speedup.md` Phase 2.5 for the full spec. Concretely:

- **Extend `TrainTiming` (`models/callbacks.py`) with val + test + reload overhead accounting.** Time one val epoch once post-warmup, time the post-fit test pass, estimate dataloader-reload overhead. Add a second metric `train/extrapolated_wall_h_total` (= compute + val + test + reload) and report both. Keep `train/extrapolated_gpu_h_total` as the compute-only number so probe-to-probe comparison stays valid. Print both in the epoch-end summary line.
- **Switch the canonical full-run config to whole epochs.** `num_steps: 300000` → 349.05 epochs (partial tail). Use `num_epochs: 349, num_steps: null` (or 350 — 300k is the spec but 349 lands at 299,891 steps, 350 at 300,750; either is closer to integer epochs than the current setup).
- **Profiler trace (extends Phase 2.1).** A PyTorch profiler run that covers (a) 200 train steps post-warmup, (b) one full val epoch, (c) one epoch boundary including reload + checkpoint write. Output: a table of train/val/IO time fractions. Re-run with `--tf32 True --torch-compile True` to see how val's share of wall time shifts under the speedup (val uses the same backbone forward as train, so compile should help val too — verify).

Each of these is a separate commit per CLAUDE.md commits policy.

### Step 2 — Resubmit the canonical full runs

Once Phase 2.5 is in: relaunch the three runs (`full_kernel_on`, `full_tf32`, `full_tf32_compile`). Recommended pattern, per the new "use the whole node" rule in CLAUDE.md:

- **One sbatch on `gpu_premium` with `--gpus=4`**, launching three lanes in parallel via `CUDA_VISIBLE_DEVICES=0/1/2`, each with `--num-epochs 349 --num-steps -1`. Lane 3 idle (or use it for a fourth ablation — e.g. a `bf16-mixed` fix attempt if you implement the autocast fix described below).
- Premium = fast queue start, 2× cost factor but the cost is per-node-hour and the node is allocated whole anyway — so packing three lanes onto one node makes premium roughly cost-equivalent to three separate `gpu_regular` allocations (which is what we had before, but those were queueing for hours).
- Premium MaxWall is 48h per sacctmgr (the previous restart note said 6h — that was wrong). No `LimitRun` chunking needed.
- Alternative: `gpu_shared_interactive` (salloc-only, 4h wall, ~15 s start, per-GPU billing — cheapest per-GPU but one allocation at a time, would need to be done sequentially in a tmux session). Slower wall-clock to finish all three, lowest cost.

Queue poll: `sqs | grep luisc440` or `sacct -j <ids> -o JobID,State,Elapsed,ExitCode -n`. Once `RUNNING`, log is at `$SCRATCH/SCD_data/finetune_runs/<jobid>_<tag>/train.log`.

### Step 3 — Harvest results and gate

When each run finishes:
1. Pull `test_loss` from the bottom of its slurm log (eV → meV ×1000).
2. Pull steady-state step time from the last `[TrainTiming] epoch …` line (not the first — recompile spikes inflate epoch 0/1 for compile runs).
3. Pull both the compute-only and wall-clock-honest GPU-h from the same line.
4. Update the `_pending_` rows in `finetune-speedup-worklog.md`.
5. **Acceptance**: HOMO MAE ≤ 14 meV for the Phase 1 baseline (paper is 12.7 meV; ~10% slop). Phase 2 candidates additionally need MAE within ~±0.5 meV of the Phase 1 baseline (training-dynamics invariant). Anything outside → reject the lever and flag.
6. Commit each update (`finetune-speedup-worklog.md` only; the plan file shouldn't move unless scope changes).
7. Report `X GPU-h (compute) / Y GPU-h (wall-clock) vs paper's 46`. The wall-clock number is the apples-to-apples one.

## Open question for the next agent

**Should the canonical full runs use `--load-model <local checkpoint path>` (paper-faithful) or keep `--load-hf ct-scd-pcq` (current sbatch)?** The two paths load the same weights but differ in mean/std propagation (see "What changed" #3). Recommend: spend 10 minutes inspecting `data/loaders.py` standardization logic to determine if model.mean/std affects the loss (data-side use) or only output rescaling (prediction-side use). If data-side: switch to `--load-model` and re-anchor to the paper. If prediction-side only: keep `--load-hf` and note the divergence in the worklog as "doesn't affect MAE". Either way, document the choice.

The local hf-cache path after one download is deterministic:
```
experiments/models--Ty-Perez--ct-scd-pcq/snapshots/<hash>/last.ckpt
```

## Useful runtime facts (unchanged from prev note)

- Env: pixi default env. `.pixi/` is a symlink to `$SCRATCH/SCD_data/.pixi`. Pixi binary at `/global/homes/l/luisc440/.pixi/bin/pixi`; sbatch scripts hardcode this (Slurm `--export=` can strip `$HOME`).
- TorchMD-Net neighbor kernel is already built (`models/ET_models/extensions/torchmdnet_extensions.so`). Rebuild via `scripts/setup/build_kernel_and_smoketest.sbatch` if `.pixi/` is wiped.
- `--precision` accepts strings (`"bf16-mixed"`, `"16-mixed"`) as well as ints. Default 32.
- `torch.compile` requires the `@torch.compiler.disable()` patch on `get_neighbor_pairs_kernel` in `models/ET_models/extensions/__init__.py`. Committed; redundant only if the C++ extension's `set_python_module` declaration gets fixed upstream.
- W&B project: `SCD-finetune-speedup` (user `luis-carretero-eth-zurich`).
- **QoS reference** (from `sacctmgr show qos`, not the slightly stale `../misc/setup-info.md`):
  - `gpu_premium`: full-node, 2× factor, 48h max — fast queue
  - `gpu_regular`: full-node, 1× factor, 48h max — slow queue for long jobs
  - `gpu_shared` / `gpu_shared_interactive`: per-GPU billing, 1× factor
  - `gpu_interactive` / `interactive`: full-node, 1× factor, 4h max, salloc-only
  - All `UsageFactor=1.0` in Slurm DB; iris applies premium 2× separately.

## Things deliberately not done

- **bf16-mixed** — needs targeted `torch.amp.autocast(enabled=False)` blocks around `noise_normalizer` (an `AccumulatedNormalization` with side-effectful `update_statistics()` that mixes its fp32 buffers with bf16 autocast tensors). Not in scope for the "trivial wins" pass; revisit only if the TF32+compile stack hits the acceptance gate and we want more. Lane 3 of the resubmission would be a cheap place to retry if the autocast fix is implemented.
- **Phase 2.3 deeper optimizations** (fused attention, effective-batch increase, DDP scaling sweep) — defer until Phase 2.2 winners are confirmed at full scale and Phase 2.5 reveals where time actually goes.
- **DDP×4 with per-GPU batch=128** would shrink wall-clock 4× at the same effective GPU-h, but **violates the plan's "same effective batch size" training-dynamics invariant** (would go from effective 128 → 512). Per-GPU batch=32 keeps effective=128 but loses arithmetic intensity. Don't enable without explicit scope change.
- **Tests for `TrainTiming`** — callback is GPU-only and small; per CLAUDE.md test policy, not worth the round-trip until something breaks. The Phase 2.5 extension is a good moment to add one if the val-overhead accounting has any non-trivial math (e.g. checking that wall-clock = compute + measured-overheads exactly when run on a real epoch).
