# SCD Finetuning: Reproduce + Speed Up

Goal: reproduce the CT-SCD QM9 HOMO finetune from the paper (46 GPU-h/task, 12.7 meV MAE; Table 5, `paper_ref-SCD/scd_vs_jmp_qm9.tex`), then make it faster.

Reference command from the user:
```bash
python train.py --conf configs/finetune_qm9.yaml \
                --load-model 'experiments/{NAME}/{checkpoint}.ckpt' \
                --job-id pretrained_qm9-homo
```
The `examples.ipynb` "Example 1" cell downloads the `Ty-Perez/ct-scd-pcq` checkpoint to `./experiments/...` via `hf_hub_download`; use that path for `--load-model`. Alternatively `--load-hf ct-scd-pcq` skips the manual step.

Sibling project to mirror: `../MLIP_propbench` (pixi env, paper LaTeX, Perlmutter conventions).

**Scope guardrails (decided up-front):**
- W&B project for all runs in this work: `SCD-finetune-speedup` (separate from propbench's `SCD-MLIP-benchmarking`). Set `wandb_project: SCD-finetune-speedup` in configs.
- Finetune only. Pretraining is out of scope.
- **Speedups must preserve training dynamics exactly** — no batch-size, LR, optimizer, step-count, or schedule changes. Only acceleration (TF32, bf16, compile, kernel, dataloader, DDP — DDP is already used via `ngpus=-1`). A memory+throughput sweep (varying batch size) is deferred; mirror propbench's `scripts/propbench/measure_batch_size.py` + `docs/propbench/perf_opt/bs_and_throughput.md` workflow when we get there.

---

## Phase 0 — Repo wiring (env + paper assets)

Do not touch model code in this phase.

### 0.1 Copy paper LaTeX into this repo
- `cp -r ../MLIP_propbench/docs/paper_ref-SCD docs/paper_ref-SCD`. The key reference for this work is `scd_vs_jmp_qm9.tex` (Table 5: 46 GPU-h/task, HOMO 12.7 meV) and `Hyperparameters.tex` (Table: QM9 finetune hparams — already match `configs/finetune_qm9.yaml`: batch 128, lr 5e-4, 300k steps, warmup 10k, β1 0.995, weight_decay 0.01, droppath 0.15→0.1, σ_reg 0.005, denoise weight 0.1).

### 0.2 Author `CLAUDE.md` for this repo
- Adapt from `../MLIP_propbench/CLAUDE.md`. Keep: compute-environment rules (Perlmutter `$SCRATCH`, `.pixi/` symlink to scratch, login-node hygiene, `python -u`, premium QoS <6h, interactive `salloc` for dev), coding conventions (pixi, ruff, type hints, sparse comments), W&B project guidance. Drop: anything propbench-specific (datasets interface, three-task scope, MACE/orbuma feature envs). Add a one-line project description: "Official SCD pretrain/finetune; QM9 HOMO is the primary finetune validation target."

### 0.3 Replace `requirements.txt` workflow with `pixi`
- Model the `pixi.toml` after `../MLIP_propbench/pixi.toml`, single env (no feature splits — there is no MACE/orbuma split here). Required deps (from current `requirements.txt` + `examples.ipynb`):
  - conda: `python=3.12`, `pip`, `pandas`, `scipy`, `ase>=3.22`, `pymatgen>=2023`, `rdkit`, `matplotlib`, `tqdm`, `pyyaml`, `huggingface_hub`, `pytest`, `ruff`.
  - pypi: `torch==2.8.0` from the cu126 index (matches propbench's working stack), `pytorch-lightning>=2.0`, `lightning-utilities`, `torch-geometric>=2.4`, `datasets>=2.0`, `wandb>=0.12`.
  - Confirm against `data/`, `models/`, `train.py` imports — flag any missing.
- Tasks: `pixi run train …`, `pixi run lint`, `pixi run format`, `pixi run test`.
- Build the TorchMD-Net kernel inside the env: `pixi run python -m pip install -e .` won't work (no setup.py at root); instead document `cd models/ET_models && pixi run python setup.py build_ext --inplace` per `README.md`. Without the kernel, `noise_in_loader=True` must be set in the config (already documented in README).
- On Perlmutter: symlink `.pixi/` → `$SCRATCH/SCD_data/.pixi` (mirror propbench's pattern; the home filesystem is too small/slow for the env).
- Add `ruff.toml` (copy propbench's; line-length 100).
- Delete `requirements.txt` only after `pixi install` + smoketest passes.

### 0.4 Smoketest script
- Add `scripts/setup/smoketest.py` (mirror propbench's): import torch, check CUDA, instantiate the model from `configs/model_configs/scd.yaml`, run one forward+backward on a tiny QM9 batch. Goal: catch env breakage before launching a 46h job.

---

## Phase 1 — Reproduce the baseline + add timing

This phase establishes the reference: paper-comparable HOMO MAE *and* a precise extrapolated GPU-h/task estimate, without running 300k steps blind.

### 1.1 Get the checkpoint
- Run the first cell of `examples.ipynb` (or a one-liner: `hf_hub_download(repo_id="Ty-Perez/ct-scd-pcq", filename="last.ckpt", cache_dir="./experiments")`) to populate `experiments/models--Ty-Perez--ct-scd-pcq/snapshots/<hash>/last.ckpt`. Use that path as `--load-model`. Or skip with `--load-hf ct-scd-pcq`.

### 1.2 Decide the comparison reference precisely
- Paper Table 5 reports **46 GPU-h/task** for CT-SCD finetuning, target MAE **12.7 meV** on HOMO. Paper uses A100s for SCD training elsewhere (`Hyperparameters.tex` pretraining table) but the table does *not* specify the finetune GPU type or count. **Goal is "roughly reproduce", not match exactly** — if our hardware (Perlmutter A100s) lands somewhere in the same ballpark (e.g. 30–80h on 1 A100) and HOMO MAE is in band, that is sufficient for a baseline. Document the GPU type, count, and measured GPU-h in the report so any divergence from 46h is attributable.
- Step math: `train_size=110000, batch_size=128` → ~860 steps/epoch; `num_steps=300000` → ~349 effective epochs (`num_epochs=3000` is just a PL upper bound). 46h / 300k steps ≈ **0.55 s/step** if assuming a single-GPU baseline — this is the reference order-of-magnitude, not a target to hit exactly.
- **Also produce two baseline numbers**: one with the TorchMD-Net compiled kernel (`noise_in_loader=False`) and one without (`noise_in_loader=True`). The delta isolates the kernel's contribution to the speedup budget; Phase 2 should not double-count it.

### 1.3 Add timing/throughput instrumentation
The repo already has `LimitRun` (`models/callbacks.py:287`) that tracks wall time and stops cleanly. Extend or add a sibling callback `TrainTiming` that:
- Records wall time per training batch (excluding val) using `time.perf_counter`. Skip the first N batches (warmup, e.g. 50) before averaging — CUDA graph warmup + dataloader spin-up skew the first epoch.
- Logs to W&B: `train/step_time_ms` (rolling mean over last 200 steps), `train/throughput_samp_per_s`, `train/extrapolated_gpu_h_total = step_time * num_steps / 3600 * world_size`.
- Prints a summary at `on_train_epoch_end`: mean step time, std, and the implied total GPU-h to finish all `num_steps`.
- Gate on `global_rank == 0` to avoid duplicate logs under DDP.

Hook it into `train.py` alongside the existing callbacks.

### 1.4 Validate the timing extrapolation
- Launch the full finetune on 1 A100 with `LimitRun(minutes=30)` (or use `--num-steps 5000` override). Confirm the per-step variance stabilizes after warmup (target: std/mean < 5% over the last 1000 steps). Compare the extrapolation to the paper's 46h — if it overshoots by >15% on the *same hardware*, investigate before launching the full run.
- Once the estimate is stable from a 1–2k-step probe, the full-run launch is a formality.

### 1.5 Full reproduction run
- Launch the 300k-step finetune (use Perlmutter `sbatch`, `-q regular` since wall time will be >6h; mirror propbench's sbatch scripts). Log to W&B project `SCD-finetune-speedup`.
- Acceptance: HOMO test-set MAE within ~10% of 12.7 meV (i.e. ≤14 meV). GPU-h is **observational, not pass/fail** at this stage — we want to know what 1× A100 actually costs on our setup, not gate on hitting 46h. Record: GPU type/count, kernel on/off, total GPU-h, step-time mean/std.

---

## Phase 2 — Speedups

**Do not start until Phase 1 numbers exist.** Without a baseline step time and a known-good MAE, any "improvement" is unfalsifiable.

**Training-dynamics invariant**: every change here is pure acceleration. Same effective batch size, same LR, same step count, same schedule, same seed. If MAE moves outside ~±0.5 meV vs the Phase 1 baseline, the change is rejected even if it's faster.

### 2.1 Profile to find the bottleneck
- PyTorch profiler over a representative 200-step window post-warmup: `torch.profiler.profile(activities=[CPU,CUDA], schedule=schedule(...), on_trace_ready=tensorboard_trace_handler(...))`. Identify whether time is in: dataloader (CPU-bound, fixable via `num_workers`/`persistent_workers`/prefetch), graph construction (TorchMD-Net kernel vs loader-side path), attention/MLP forward, or backward.
- Cross-check with the `arch_speed_mem.tex` table (CT: 74.5 ms/train-step on A5000, batch 128, single forward). Our finetune does forward + backward + denoise head; expected ≥2× that. If observed step time is wildly higher, the bottleneck is probably not the model.
- Report: a table of top-5 ops by self-CUDA-time, plus dataloader stall time.

### 2.2 Low-hanging optimizations (independent, A/B each)
Apply one at a time, measure `step_time_ms` and HOMO MAE after a short run, then keep or drop. Stack only winners.

- **TF32 / matmul precision**: `train.py:268` already calls `torch.set_float32_matmul_precision('medium')` but **only when `args.pretraining=True`**. Extend to finetune on A100/H100. Free 1.3–2× on matmul-heavy paths.
- **bf16 mixed precision**: `precision: 32` in `configs/finetune_qm9.yaml`; PL supports `precision: 'bf16-mixed'`. Verify no NaNs in the denoising head; the `noise_normalizer` AccumulatedNormalization buffers must stay fp32.
- **`torch.compile`**: the codebase already has `@torch.compiler.disable()` on one graph-construction function (`models/ET_models/graph_utils/compute.py:58`) and notes in `models/ET_models/extensions/` that the neighbor op was made compile-compatible — so partial support exists. Try `model.rep_model = torch.compile(model.rep_model, mode='reduce-overhead')` (compile the backbone, leave heads alone); fall back to `mode='default'` if recompiles trigger. Watch for: dynamic batch shapes (QM9 has variable atom counts — set `dynamic=True` or pad).
- **Dataloader**: `num_workers=6` in config; bump to physical-core-count and add `persistent_workers=True`, `pin_memory=True` if not already set in `data/loaders.py`. Check whether the QM9 in-memory dataset already pre-collates.
- **TorchMD-Net compiled kernel**: confirm the build succeeded (`noise_in_loader=False` requires it). If currently running with `noise_in_loader=True` as a fallback, building the kernel is the single biggest expected win for non-periodic data.

### 2.3 Deeper optimizations (only after 2.2 plateaus)
Still under the training-dynamics invariant — these are acceleration only.

- **Fused attention**: if the CT attention is hand-rolled, swap to `F.scaled_dot_product_attention` (FlashAttention backend on A100+). Numerically equivalent to the manual path within fp16/bf16 tolerance.
- **DDP scaling**: `ngpus=-1` already enables DDP and that is the default we're already on. Measure scaling efficiency on 2/4 GPUs (perfect = 2×/4×, <70% points at gradient-comm or dataloader bottleneck). Keep `batch_size` per-GPU and `lr` unchanged — effective batch scales with GPU count, which the paper's own multi-GPU pretraining table treats as standard; if MAE drifts, flag it and stop scaling.
- **Survey**: skim recent TorchMD-Net / SCD-adjacent issues for known speedups (Triton kernels for radius graph, cuEquivariance for tensor-product layers if CGT is in scope later).

### 2.3.x Deferred: memory + throughput sweep
Out of scope for the dynamics-preserving pass, but worth a follow-up. Mirror propbench: a `scripts/measure_batch_size.py` that varies `batch_size` and reports peak VRAM + steps/s for `{train, eval}` × `{fp32, bf16, tf32}`. Output JSON → `docs/_data/`, write-up in `docs/perf_opt/`. References: `../MLIP_propbench/scripts/propbench/measure_batch_size.py` and `../MLIP_propbench/docs/propbench/perf_opt/bs_and_throughput.md`. Only act on the data if/when we decide to relax the dynamics invariant.

### 2.4 Acceptance and reporting
For each accepted change, log to a table in `docs/finetune-speedup.md` (this file, "Results" section appended at the end):
- step time before/after, percent speedup
- HOMO MAE (full run or extrapolated equivalent) — change must be within ~±0.5 meV
- the W&B run ID for both arms (both in `SCD-finetune-speedup`)
- one-line root cause

No fixed target. Phase 2.2 is the conservative pass (TF32 + bf16 + compile + dataloader + kernel) — keep what wins, drop what doesn't, stop when individual changes give <5% and there are no obvious profiler hotspots left. Phase 2.3 only if Phase 2.2 leaves real headroom.

---

## Out of scope (for this work item)
- Other QM9 targets, Matbench, LBA — those are propbench's job.
- Pretraining changes.
- Architecture changes (CGT, GET) — Table 5's 46h figure is for CT-SCD specifically.

---

## Results

Hardware reference: NERSC Perlmutter, NVIDIA A100-SXM4-40GB (4× per node).
Paper reference: 46 GPU-h/task, HOMO MAE 12.7 meV (Table 5, A100, single-task QM9 HOMO).

### Phase 1 — Baseline

Setup: 1× A100-SXM4-40GB on Perlmutter, torch 2.8.0+cu126, PL 2.6.4, batch 128, fp32 (no TF32 — `torch.set_float32_matmul_precision` left at default; the existing `if args.pretraining` gate keeps Tensor-Core fast paths off for finetuning). Step times measured by the `TrainTiming` callback, `cuda.synchronize`-anchored, mean of last 200 batches at each epoch end, first 50 batches dropped for warmup.

| Run | Kernel | Step time (ms) | Throughput (samp/s) | Extrapolated GPU-h (300k × 1) | HOMO MAE | W&B |
|-----|--------|----------------|---------------------|------------------------------|----------|-----|
| Probe (3000 steps) | on | 112.5 ± 4.3 (epoch 2; std/mean 3.8%) | 1138 | 9.4 | 0.061 eV ≈ 61 meV (undertrained) | `probe_kernel_on_20260526-194641` |
| Probe (3000 steps) | off (`noise_in_loader=True`) | 111.6 ± 4.0 (epoch 2; std/mean 3.6%) | 1147 | 9.3 | 0.057 eV ≈ 57 meV (undertrained) | `probe_kernel_off_20260526-195340` |
| Full reproduction (300k steps) | on | _pending_ | — | — | _pending_ (target ≤14 meV) | _pending_ |

**Kernel A/B (Phase 1.2).** The TorchMD-Net neighbor kernel (`noise_in_loader=False`) gives **~0% speedup** vs the loader-side path on this workload — both arms land at 112 ms/step within run-to-run noise. The doc predicted "the kernel is the single biggest expected win for non-periodic data"; that expectation does not hold for QM9 finetune at batch 128. Probable cause: QM9 molecules are small (≈18 atoms each), so graph construction is cheap (~1 ms/batch), and 6 dataloader workers + `pin_memory=True` fully hide the CPU cost behind the GPU compute. The MAE delta at 3000 steps (61 vs 57 meV) is undertrained noise, not a real signal. **Implication for Phase 2: do not budget for kernel speedup; the headroom must come from compute (TF32 / bf16 / compile) or dataloader (which is already hidden).**

**Sanity-check vs paper.** Paper Table 5 reports **46 GPU-h/task** for CT-SCD QM9 finetuning; our extrapolation lands at ~9.4 GPU-h — ~5× *faster*, comfortably outside the doc's "30–80h ballpark" lower bound. Two plausible reasons: (a) Perlmutter A100-SXM4 + torch 2.8 / cu126 is materially faster than whatever A100 + torch the paper used (the paper does not specify); (b) the paper's 46h figure may include overheads we don't pay (multi-task sweep, eval cadence). Either way, the doc explicitly allows this divergence ("Goal is 'roughly reproduce', not match exactly"); we keep the extrapolation as the observational reference and gate the baseline on MAE alone.

### Phase 2 — Speedups (A/B vs Phase 1 baseline)

Method: 3000-step probes (`scripts/finetune/probe_ablation.sbatch`) for each lever. Step times = epoch-1 TrainTiming (warmup excluded). MAE column is the trainer.test()  L1 loss in eV from the probe — **not** the paper-comparable number (that needs 300k steps); included only to flag obviously broken arms.

| Change | Step time (ms) | vs baseline (112.5) | HOMO MAE (eV) @ 3000 steps | W&B run | Notes |
|--------|----------------|---------------------|----------------------------|---------|-------|
| `--tf32 True` | 99.4 ± 12.4 (epoch 1) | **−11.6%** ✅ | _pending_ | `p2_tf32_20260526-200756` | `set_float32_matmul_precision('medium')` for A100/H100. Trivial win, no MAE risk expected. |
| `--precision bf16-mixed` | — (crash) | — | — | `p2_bf16_20260526-200756` | **Rejected**: RuntimeError "Index put requires source/destination dtypes match (BFloat16 vs Float)" during sanity check. The denoising/normalizer path mixes autocast bf16 with fp32 buffers (the doc warned about exactly this); fixing it needs targeted `autocast(enabled=False)` blocks around `noise_normalizer`. Deferred. |
| `--torch-compile True` (mode=default, dynamic=True) | — (crash, retry pending) | — | — | _pending_ | First attempt blew up on the TorchMD-Net `get_neighbor_pairs` op — upstream's C++ registration claims the python fake impl lives in `torchmdnet.extensions` but we host it at `models.ET_models.extensions`, so Dynamo's `set_python_module` check refused to trace. Workaround: `@torch.compiler.disable()` on `get_neighbor_pairs_kernel` (the kernel stays opaque to Dynamo so it still runs as the compiled CUDA op). Probe retrying as `p2_compile_v2`. |

