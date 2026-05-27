# Restart note for the next agent

You're picking up the SCD finetune-speedup work mid-flight. Snapshot taken 2026-05-26 ~20:30 PDT.

## TL;DR
- Phase 0 wiring (pixi, ruff, CLAUDE.md, paper LaTeX, smoketest, TrainTiming callback) → done.
- Phase 1 probes (kernel-on, kernel-off) → done. Step time 112.5 ms, extrapolated 9.4 GPU-h. Kernel A/B is a wash.
- Phase 2 probes (TF32, bf16, compile, stack) → done. **Winner: `--tf32 True --torch-compile True`, −28.3% step time**, MAE preserved at probe scale. bf16-mixed crashes (deferred).
- **Three 300k-step full runs are sitting in NERSC `gpu_regular` PENDING** (no estimated start at snapshot time). These produce the paper-comparable MAE and confirm the speedup at full scale. That's the only thing left.

Full numbers + per-run W&B IDs: [`finetune-speedup-worklog.md`](finetune-speedup-worklog.md). Plan / scope / acceptance: [`finetune-speedup.md`](finetune-speedup.md).

## Queued jobs to watch

```
53461802  full_kernel_on (Phase 1 baseline, fp32, no flags)
53462549  full_tf32      (TF32 only)
53462886  full_tf32_compile  (TF32 + torch.compile — the Phase 2 winner)
```

Quick poll: `sqs | grep luisc440` or `sacct -j 53461802,53462549,53462886 -o JobID,State,Elapsed,ExitCode -n`. Once a job is `RUNNING`, the live log is at `$SCRATCH/SCD_data/finetune_runs/<jobid>_<tag>/train.log` and `slurm_logs/full_<jobid>.out`. Each run is ~9–10 GPU-h estimated; with TF32+compile, ~7 GPU-h.

When each one finishes:
1. Pull `test_loss` from the bottom of its slurm log (eV). Multiply by 1000 for meV.
2. Pull steady-state step time from the last `[TrainTiming] epoch …` line in the log (not the first — recompile spikes inflate epoch 0/1 for compile runs).
3. Update the corresponding `_pending_` row in `finetune-speedup-worklog.md`.
4. The acceptance gate in the plan is **HOMO MAE ≤ 14 meV** for the Phase 1 baseline (paper is 12.7 meV; ~10% slop). Phase 2 candidates additionally need MAE within ~±0.5 meV of the Phase 1 baseline (per the plan's training-dynamics invariant). Anything outside that → reject the lever and flag it.
5. Commit each update (`finetune-speedup-worklog.md` only; the plan file shouldn't move unless scope changes).

## If the queue stays stuck

NERSC `gpu_regular` waited >20 min at the time of the snapshot with no estimated start. If that persists, options in order of preference:

- **Wait it out.** A 10h run is happy to start overnight; this is the cheapest path.
- **Switch to `gpu_premium` + LimitRun chunking.** Premium caps at 6h wall, so a 10h run needs two submissions chained via `--restart True`. The `LimitRun(minutes=...)` callback in `models/callbacks.py` already saves checkpoints on stop; cap the first submission at e.g. `--max_time_minutes 330` (5h30m with 30s buffer) and resubmit with `--restart True` for the tail. Costs 2× node-hours vs regular.
- **Try `gpu_shared`** (per-GPU billing, slower queue than premium). Was tried once earlier and was also slow — your mileage may vary.

## Useful runtime facts

- Env: pixi default env. `.pixi/` is a symlink to `$SCRATCH/SCD_data/.pixi` (per CLAUDE.md). Pixi binary is at `/global/homes/l/luisc440/.pixi/bin/pixi`; sbatch scripts hardcode this absolute path (don't rely on `$HOME` expansion on compute — Slurm's `--export=` can strip it).
- TorchMD-Net neighbor kernel is already built (`models/ET_models/extensions/torchmdnet_extensions.so`). If you blow away `.pixi/` you'll need to rebuild — see `scripts/setup/build_kernel_and_smoketest.sbatch`.
- `--precision` was relaxed to accept strings (`"bf16-mixed"`, `"16-mixed"`) as well as ints. Default still 32. Don't trip on this.
- `torch.compile` requires the `@torch.compiler.disable()` patch on `get_neighbor_pairs_kernel` in `models/ET_models/extensions/__init__.py`. That's committed, but if anyone rebuilds the C++ extension and "fixes" the `set_python_module` declaration upstream, the patch becomes redundant.
- W&B project for everything in this work item: `SCD-finetune-speedup`. The user is `luis-carretero-eth-zurich`.

## Things deliberately not done

- **bf16-mixed** — needs targeted `torch.amp.autocast(enabled=False)` blocks around `noise_normalizer` (an `AccumulatedNormalization` with side-effectful `update_statistics()` that mixes its fp32 buffers with bf16 autocast tensors). Not in scope for the "trivial wins" pass; revisit only if the TF32+compile stack hits the acceptance gate and we want more.
- **Phase 2.3 deeper optimizations** (fused attention, effective-batch increase, DDP scaling sweep, profiler trace) — defer until Phase 2.2 winners are confirmed at full scale.
- **Tests for `TrainTiming`** — the callback is GPU-only and small; per CLAUDE.md test policy ("would this test catch a real failure mode?") it's not worth the round-trip until something breaks.
