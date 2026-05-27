# SCD Finetune Speedup — Worklog

Chronological log of the work executed against the plan in [`finetune-speedup.md`](finetune-speedup.md). Numbers here are the live results; cite the plan doc for goals, acceptance criteria, and scope.

Hardware reference: NERSC Perlmutter, NVIDIA A100-SXM4-40GB (4× per node), torch 2.8.0+cu126, PL 2.6.4, batch 128.
Paper reference: 46 GPU-h/task, HOMO MAE 12.7 meV (Table 5, A100, single-task QM9 HOMO).
W&B project: `SCD-finetune-speedup` (luis-carretero-eth-zurich).

## Chronological board

Every measurement we've taken, in submission order. Step ms / throughput / compute-h come from the `[TrainTiming]` summary; wall-h is only available for runs that included the Phase 2.5 callback extension (commit `c0789a1`, 2026-05-26 ~22:00 PDT — everything before is `n/a`). Compute GPU-h is extrapolated to the canonical 300k-step run for probes, and is the actual run cost for full runs. `samp/s = 128 / (ms/1000)`.

| Time (PDT) | Tag | Step (ms) | Thru (samp/s) | Compute GPU-h | Wall GPU-h | Notes |
|---|---|---|---|---|---|---|
| 2026-05-26 19:46 | `probe_kernel_on` | 112.5 ± 4.3 | 1138 | 9.4 | n/a | Phase 1 baseline, kernel on, 3000 steps, `--load-hf` |
| 2026-05-26 19:53 | `probe_kernel_off` | 111.6 ± 4.0 | 1147 | 9.3 | n/a | Phase 1 kernel A/B (~0% speedup) |
| 2026-05-26 20:07 | `p2_tf32` | 100.3 ± 2.0 | 1276 | 8.4 | n/a | Phase 2 TF32, −10.8% |
| 2026-05-26 20:07 | `p2_bf16` | crash | — | — | — | `BFloat16 vs Float` in noise_normalizer (deferred) |
| 2026-05-26 20:12 | `p2_compile_v2` | 91.2 ± 1.4 | 1404 | 7.6 | n/a | Phase 2 compile, −18.9% (after `@torch.compiler.disable()` patch) |
| 2026-05-26 20:18 | `p2_tf32_compile` | 80.7 ± 1.8 | 1586 | 6.7 | n/a | Phase 2 stack, −28.3% — winner candidate |
| 2026-05-26 21:27 | `full_*` (53465532) | cancelled | — | — | — | DDP `EADDRINUSE` on 2/3 lanes, fix `a56441a` |
| 2026-05-26 21:35 | `full_kernel_on` (53465652) | _pending_ | _pending_ | _pending_ | _pending_ | canonical baseline, `--load-model`, 349 ep |
| 2026-05-26 21:35 | `full_tf32` (53465652) | _pending_ | _pending_ | _pending_ | _pending_ | canonical TF32 |
| 2026-05-26 21:35 | `full_tf32_compile` (53465652) | _pending_ | _pending_ | _pending_ | _pending_ | canonical TF32+compile |

Update the `_pending_` rows in place as each lane lands `test_loss` + final `[TrainTiming]` line. Append a new row for any follow-up probe (bf16-with-autocast-fix, dataloader sweep, etc).

## Phase 1 — Baseline

Step times measured by the `TrainTiming` callback (`models/callbacks.py`): `cuda.synchronize`-anchored, mean of the last 200 batches at each epoch end, first 50 batches dropped for warmup.

| Run | Kernel | Step time (ms) | Throughput (samp/s) | Extrapolated GPU-h (300k × 1) | HOMO MAE | W&B |
|-----|--------|----------------|---------------------|------------------------------|----------|-----|
| Probe (3000 steps) | on | 112.5 ± 4.3 (epoch 2; std/mean 3.8%) | 1138 | 9.4 | 0.061 eV ≈ 61 meV (undertrained) | `probe_kernel_on_20260526-194641` |
| Probe (3000 steps) | off (`noise_in_loader=True`) | 111.6 ± 4.0 (epoch 2; std/mean 3.6%) | 1147 | 9.3 | 0.057 eV ≈ 57 meV (undertrained) | `probe_kernel_off_20260526-195340` |
| Full reproduction (300k steps) | on | _pending_ | — | — | _pending_ (target ≤14 meV) | _pending_ (sbatch 53461802) |

**Kernel A/B (Phase 1.2).** The TorchMD-Net neighbor kernel (`noise_in_loader=False`) gives **~0% speedup** vs the loader-side path on this workload — both arms land at 112 ms/step within run-to-run noise. The plan predicted "the kernel is the single biggest expected win for non-periodic data"; that expectation does not hold for QM9 finetune at batch 128. Probable cause: QM9 molecules are small (≈18 atoms each), so graph construction is cheap (~1 ms/batch), and 6 dataloader workers + `pin_memory=True` fully hide the CPU cost behind the GPU compute. The MAE delta at 3000 steps (61 vs 57 meV) is undertrained noise, not a real signal. **Implication for Phase 2: do not budget for kernel speedup; the headroom must come from compute (TF32 / bf16 / compile) or dataloader (which is already hidden).**

**Sanity-check vs paper.** Paper Table 5 reports **46 GPU-h/task** for CT-SCD QM9 finetuning; our extrapolation lands at ~9.4 GPU-h — ~5× *faster*, comfortably outside the plan's "30–80h ballpark" lower bound. Two plausible reasons: (a) Perlmutter A100-SXM4 + torch 2.8 / cu126 is materially faster than whatever A100 + torch the paper used (the paper does not specify); (b) the paper's 46h figure may include overheads we don't pay (multi-task sweep, eval cadence). Either way, the plan explicitly allows this divergence ("Goal is 'roughly reproduce', not match exactly"); we keep the extrapolation as the observational reference and gate the baseline on MAE alone.

## Phase 2 — Speedups (A/B vs Phase 1 baseline)

Method: 3000-step probes (`scripts/finetune/probe_ablation.sbatch`) for each lever. Step times = late-epoch TrainTiming (warmup excluded). MAE column is the `trainer.test()` L1 loss in eV from the probe — **not** the paper-comparable number (that needs 300k steps); included only to flag obviously broken arms.

| Change | Step time (ms) | vs baseline (112.5) | HOMO MAE (eV) @ 3000 steps | W&B run | Notes |
|--------|----------------|---------------------|----------------------------|---------|-------|
| `--tf32 True` | 100.3 ± 2.0 (epoch 3) | **−10.8%** ✅ | 0.0645 ≈ 64.5 meV | `p2_tf32_20260526-200756` | `set_float32_matmul_precision('medium')` for A100/H100. Trivial win, no MAE risk. MAE delta vs baseline (+3.5 meV) is within probe noise (undertrained). |
| `--precision bf16-mixed` | — (crash) | — | — | `p2_bf16_20260526-200756` | **Rejected**: RuntimeError "Index put requires source/destination dtypes match (BFloat16 vs Float)" during sanity check. The denoising/normalizer path mixes autocast bf16 with fp32 buffers (the plan warned about exactly this); fixing it needs targeted `autocast(enabled=False)` blocks around `noise_normalizer`. Deferred. |
| `--torch-compile True` (mode=default, dynamic=True) | 91.2 ± 1.4 (epoch 3) | **−18.9%** ✅ | 0.0601 ≈ 60.1 meV | `p2_compile_v2_20260526-201259` | First attempt blew up on the TorchMD-Net `get_neighbor_pairs` op — upstream's C++ registration declares the python fake impl in `torchmdnet.extensions` but we host it at `models.ET_models.extensions`, so Dynamo's `set_python_module` check refused to trace. Fix: `@torch.compiler.disable()` on `get_neighbor_pairs_kernel` (kept opaque to Dynamo, still runs as the compiled CUDA op). Recompile spikes at epoch boundaries inflate epoch-1 std; mean is stable by epoch 2. |
| **Stack: `--tf32 True --torch-compile True`** | **80.7 ± 1.8 (epoch 2)** | **−28.3%** ✅✅ | 0.0521 ≈ 52.1 meV | `p2_tf32_compile_20260526-201828` | Speedups multiply cleanly: (1−0.108)(1−0.189) = 0.723 ⇒ predicts −27.7%; measured −28.3%. MAE at 3000 steps is actually lower than baseline (probe-noise territory). This is the Phase 2 winner candidate; submitted as `full_tf32_compile` for the 300k-step MAE confirmation. |

## Phase 2.5 — Apples-to-apples GPU-h + val profiling

Three changes landed before the canonical resubmission to make the GPU-h figure comparable to the paper's 46:

- **`TrainTiming` now reports both compute-only and wall-clock-honest GPU-h.** `train/extrapolated_gpu_h_total` is unchanged (compute only — kept for probe-to-probe stability). New `train/extrapolated_wall_h_total` adds per-epoch overhead (full-epoch wall − step×steps_per_epoch, captures val + reload + ckpt) × planned-epochs + post-fit `trainer.test()`. Epoch-end summary prints both. Regression tests in `tests/test_traintiming.py` guard the `_extrapolate` math (commits `c0789a1`, `e474f9e`).
- **Canonical config switched to whole epochs.** `num_epochs: 349, num_steps: -1` (= 300,140 steps via 860 batches/epoch — closest integer-epoch match to the paper's 300k). Removes the partial-tail epoch and keeps train/val ratio stable across arms (commit `e0effc0`).
- **Profile sbatch** (`scripts/finetune/profile_run.sbatch`) packs two lanes (baseline, tf32+compile) on one premium node with `--profile-trace-dir` enabling PL's PyTorchProfiler. 200-step train window post-warmup + one val epoch + one epoch boundary. Run when needed to break down train vs val vs IO share (commit `e01aa6c`).

## `--load-model` vs `--load-hf` — paper-faithful decision

The paper's reference command (`python train.py --conf … --load-model …`) and our sbatches diverged on this. Investigation:

- `models/ET_models/scd_model.py:312-326` uses `model.mean`/`model.std` **inside the forward pass**: atom outputs are scaled by `std` before reduce, then `mean` is added to the molecular total. This is data-side/loss-affecting, not cosmetic output rescaling.
- The `ct-scd-pcq` pretrain has no y-target → `data.mean`/`std` were `None` during pretraining → checkpoint stores `mean=0`, `std=1`.
- `--load-model` keeps those (mean=0, std=1). Model emits HOMO in physical units; head learns the full ~−0.4 eV range.
- `--load-hf` (current sbatch path) overwrites with QM9 HOMO data mean/std (≈−0.4 eV / ≈0.04 eV). Model emits normalized internally; gradient on the atom-output projection is multiplied by `std`, giving ~25× smaller effective LR on that layer.

**Decision: canonical full runs use `--load-model`** to match the paper. All Phase 1/2 probes ran with `--load-hf`, so the speedup % deltas (TF32 −10.8%, compile −18.9%, stacked −28.3%) stand (orthogonal to the load path), but the full-run MAE column establishes a new paper-faithful baseline rather than comparing to the Phase 1 probe MAE (which is undertrained anyway).

## Pending full runs (packed on one `gpu_premium` 4-GPU node)

| Slurm JobID | Lane | W&B job_id | Lever | Status |
|---|---|---|---|---|
| _pending_ | 0 | `full_kernel_on` | none (paper-faithful baseline) | not yet submitted |
| _pending_ | 1 | `full_tf32` | `--tf32 True` | not yet submitted |
| _pending_ | 2 | `full_tf32_compile` | `--tf32 True --torch-compile True` | not yet submitted |
| _idle_ | 3 | — | — | reserved (bf16-mixed once autocast fix lands) |

Submission script: `scripts/finetune/full_canonical_packed.sbatch`. All lanes use `--load-model` against the local `ct-scd-pcq` checkpoint. Each runs 349 epochs ≈ 9.5h wall on the slowest arm. Logs at `$SCRATCH/SCD_data/finetune_runs/<jobid>_canonical/<tag>/train.log`.

**53465532 cancelled after 2 min.** First submission lost 2/3 lanes to a DDP `EADDRINUSE` on the default master port — PL spins up a DDP master per process even under `distributed_backend=ddp` with one GPU, and three lanes on the same host raced for port 20532. Fixed by setting `MASTER_PORT=29500+gpu_index` per lane (commit `a56441a`). Resubmitted as **53465652**.

Acceptance: HOMO MAE ≤ 14 meV for the baseline lane (paper 12.7 meV ± 10%). Phase 2 candidates additionally within ±0.5 meV of the baseline lane (training-dynamics invariant). Report both `compute GPU-h` and `wall GPU-h` (the latter is the apples-to-apples comparison to the paper's 46).
