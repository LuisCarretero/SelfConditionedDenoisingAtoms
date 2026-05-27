# Restart note for the next agent

Snapshot: 2026-05-27 ~08:45 PDT. **The finetune-speedup work item is done.** Canonical full run 53465652 completed (with a recoverable TIMEOUT footnote), all three Phase 2 lanes harvested, every documented lever evaluated. Final verdict lives in `docs/finetune-speedup-worklog.md`.

## TL;DR — final numbers

| Lane | Step ms | Wall GPU-h | Test MAE | vs paper (46 h / 12.7 meV) |
|---|---|---|---|---|
| baseline (no levers) | 114.4 | 11.3 | 14.5 meV | 4.1× faster, +14% MAE |
| `--tf32 True` | 104.8 | 10.4 | 14.41 meV | 4.4× faster, +13% MAE |
| **`--tf32 True --torch-compile True`** | **89.6** | **9.2** | **14.34 meV** | **5.0× faster, +13% MAE** |

Phase 2 winner: TF32 + compile. Training-dynamics invariant satisfied (MAE spread 0.16 meV ≪ ±0.5 meV bound). The bf16-mixed crash fixes landed but bf16 itself is a regression on this workload; the kernel A/B is null; SDPA isn't a viable swap for sparse equivariant attention. See worklog for the full table.

## What to know if you're picking this back up

- **No follow-up runs are queued.** All Slurm jobs (53461802, 53462549, 53462886, 53465532, 53465652, 53466061, 53466174, 53466322) are in terminal state. Nothing in `squeue -u luisc440`.
- **Code state is final and tested**: `models/callbacks.py` (TrainTiming with wall-h), `models/trainer.py` (bf16 autocast wrappers), `models/ET_models/output_modules.py` (torch.norm dtype cast), `configs/finetune_qm9.yaml` (whole epochs), `scripts/finetune/full_canonical_packed.sbatch` (MASTER_PORT lanes + 14h allocation), `scripts/finetune/profile_run.sbatch` (PL profiler wiring), `tests/test_traintiming.py` + `tests/test_bf16_autocast.py`. All on `main`.
- **53465652 TIMEOUT** killed the baseline lane 9 epochs short of `trainer.test()`. Recovered the MAE from the epoch 339 checkpoint filename — same `test_loss=0.0145` across the last 5 checkpoints, so the run had plateaued. No re-run needed. Allocation bumped to 14h for any future redo.
- **W&B project**: `SCD-finetune-speedup`. Per-run logs at `$SCRATCH/SCD_data/finetune_runs/<jobid>_*/train.log` and per-lane checkpoints under `$SCRATCH/SCD_data/finetune_runs/53465652_canonical/<tag>/experiments/<tag>/`.

## What you'd want to do if extending this work

Not low-hanging anymore — each of these requires a scope decision:

1. **Re-derive the paper's 46 GPU-h to understand the 5× gap.** Likely culprits: paper hardware was older A100s (probably A100-PCIe), torch version, or implicit overheads (eval cadence, multi-task wrappers). The plan explicitly accepts the divergence; tightening it would need access to the paper authors' setup.
2. **Trade dynamics invariance for further speedup.** The locked invariant blocks DDP×4-with-larger-effective-batch and any "more data per step" lever. If the user accepts a small MAE drift, gradient accumulation + bigger effective batch could ~halve wall time.
3. **Profile to find any remaining hotspots** — `scripts/finetune/profile_run.sbatch` is wired up but never launched (op-level granularity wasn't necessary to converge on the winner). Run it if you suspect non-obvious overhead. Note: the sbatch is on `gpu_premium` (full-node billing) — refactor to `gpu_shared` first if cost matters.
4. **Hopper / Blackwell hardware.** The bf16 regression here is workload-shaped (per-edge attention with small head_dim doesn't benefit from bf16 throughput). On Hopper+ where bf16 has even more headroom and tensor cores are bigger, the calculus could flip; re-run the bf16 probe.

## Key commits this work landed (most recent first)

- `56c034d` canonical 53465652 harvest + allocation bump
- `2df6ccb` bf16 verdict — rejected (regression)
- `4620983` output_modules: cast torch.norm result for bf16-mixed
- `45b0f60` trainer: wrap noise_normalizer in autocast(enabled=False)
- `a56441a` sbatch: MASTER_PORT per CUDA_VISIBLE_DEVICES (DDP collision fix)
- `153455e` finetune: full_canonical_packed sbatch
- `e01aa6c` Phase 2.5: --profile-trace-dir + profile_run.sbatch
- `e0effc0` configs/finetune_qm9: whole epochs (349 × 860 ≈ 300k)
- `e474f9e` TrainTiming: min(step_gate, epoch_gate) for num_steps=-1
- `c0789a1` TrainTiming: wall-clock-honest GPU-h

## Things deliberately not done

- **Resubmit baseline lane to recover post-fit `trainer.test()` number.** The checkpoint-filename MAE (14.5 meV) matches the in-fit test_loss for the last 5 ckpts; the plateau is unambiguous. Re-running for 0.05 meV precision wasn't worth 11h × 1 GPU.
- **Dataloader / `persistent_workers` tweak.** Saves ~0.2 GPU-h on a 9.2 GPU-h run (~3%). Requires changing `reload_dataloaders_every_n_epochs=val_interval` to 0 in `train.py:448`, which the original author tied to `val_interval` for a reason (comment says "to shuffle data if using small val set"). Not worth the disruption for the gain.
- **`test_interval` bump.** `data/loaders.py:208` iterates the test_loader inside every val epoch when `epoch % test_interval == 0`. Bumping `test_interval` from 1 to 10 would save ~0.15 GPU-h. Same risk/reward calculus as above.
- **Fused attention.** SCD attention is sparse equivariant, not dense Q/K/V. Not a clean SDPA swap.
