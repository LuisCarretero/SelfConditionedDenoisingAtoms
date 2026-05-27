# SCD Finetune Speedup — Worklog

Chronological log of the work executed against the plan in [`finetune-speedup.md`](finetune-speedup.md). Numbers here are the live results; cite the plan doc for goals, acceptance criteria, and scope.

Hardware reference: NERSC Perlmutter, NVIDIA A100-SXM4-40GB (4× per node), torch 2.8.0+cu126, PL 2.6.4, batch 128.
Paper reference: 46 GPU-h/task, HOMO MAE 12.7 meV (Table 5, A100, single-task QM9 HOMO).
W&B project: `SCD-finetune-speedup` (luis-carretero-eth-zurich).

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

## Pending full runs (queued in NERSC `regular`)

| Slurm JobID | W&B job_id | Lever | Status |
|---|---|---|---|
| 53461802 | `full_kernel_on` (auto-suffixed) | none (Phase 1 baseline) | PENDING |
| 53462549 | `full_tf32` | `--tf32 True` | PENDING |
| 53462886 | `full_tf32_compile` | `--tf32 True --torch-compile True` | PENDING |

All three live in `$SCRATCH/SCD_data/finetune_runs/<jobid>_<tag>/` once they start. The next agent's first job is to surface their MAE numbers, fill in the `_pending_` cells above, and update the plan's acceptance verdict.
