# CLAUDE.md

## Project

Self-Conditioned Denoising for Atomistic Representation Learning (SCD) — official pretrain/finetune implementation. Primary validation target for this work item: QM9 HOMO finetune of `ct-scd-pcq` (paper Table 5 — `docs/paper_ref-SCD/scd_vs_jmp_qm9.tex`: 46 GPU-h/task, 12.7 meV MAE). Plan + acceptance criteria live in [`docs/finetune-speedup.md`](docs/finetune-speedup.md).

Sibling project: `../MLIP_propbench` (benchmarks pretrained MLIPs on QM9/Matbench/LBA). Compute conventions and infra patterns are mirrored from there.

## Compute environment

Claude runs on the **login node** of an HPC cluster — no GPU, shared, CPU/RAM-throttled. Currently Perlmutter (NERSC).

### Perlmutter rules

Operational details (Slurm account, full QoS table, `$SCRATCH` layout, queue waits, internet on compute, tmux/SSH workflow): `../misc/setup-info.md`.

- Allowed on login: file edits, `pixi install`, package-metadata inspection, import smoketests (e.g. `python -c "import torch"`), git, sbatch submission.
- **Never** on login: training, full forward passes, dataset preprocessing, anything that needs a GPU. Everything heavier goes on a compute node — `salloc` interactive for hands-on iteration, `sbatch` for long jobs.
- **Always run Python unbuffered**: `python -u …` (or `PYTHONUNBUFFERED=1`) for every invocation. Without `-u`, Slurm `.out` files / piped progress sit empty for minutes.
- **Slurm QoS:** `sbatch` jobs with wall time < 6 h use `-q premium` (2× node-hour cost, dramatically shorter queue). Jobs ≥ 6 h stay on `regular`. `interactive` (salloc, ≤ 4 h, ~15 s wait) for hands-on iteration.
- **`.pixi/` must be a symlink** to `$SCRATCH/SCD_data/.pixi`. The home filesystem is quota-limited and slow for env solves and torch wheels.
  ```bash
  rm -rf .pixi && ln -s $SCRATCH/SCD_data/.pixi .pixi
  ```

## Experiment tracking

All training runs in *this* work item log to the W&B project **`SCD-finetune-speedup`** (separate from propbench's `SCD-MLIP-benchmarking`). Set `wandb_project: SCD-finetune-speedup` in any new config; the user is already authenticated via `wandb login` on the login node.

## Coding conventions

- Simple and concise code. No speculative abstraction.
- Only comment non-trivial aspects — and when you do, comment them properly (explain *why*, not *what*). Keep comments **concise and precise**: one tight sentence beats a paragraph. The "why" is a load-bearing constraint, invariant, or surprising fact — not a tour of how the code works.
- Type hints on new function signatures (args and return). Existing code is not type-annotated end-to-end; do not retrofit unless you're rewriting the function.
- **Pixi** for env management. Dependencies in `pixi.toml`; run via `pixi run <cmd>`. No ad-hoc `pip install` into the env. Cluster-specific `.pixi/` placement rules live in **Compute environment** above.
- **Ruff** for lint + format (config in `ruff.toml`, line-length 100). Run `pixi run lint` and fix any issues before committing.

## Commits

Make a separate commit for every self-contained change. A "self-contained change" is one logical thing — a new callback, a new sbatch script, a config bump, a doc section, a single Phase 2 lever — anything that can be reverted on its own without leaving the tree broken. Resist the urge to bundle: small commits make `git log` legible and let us roll back individual changes without untangling.

This holds even when work is in flight (multiple things touched at once): commit each piece as it's finished rather than waiting for a "natural" stopping point. Skip the commit only when explicitly asked to wait, or when the change is genuinely entangled with another in-progress edit and splitting it would mean broken intermediate states.

## Tests

When a change adds non-trivial source code (a new callback, a new training-path branch, a new data adapter), ship a pytest test under `tests/`. Each test must regression-guard a **specific failure mode** — a bug we've already hit, an invariant the code silently relies on, or a known-fragile interface with an external library. Skip "the function returns a tensor of the expected shape" / "the constructor produces an object" — those are guaranteed by construction. Bar: *"if I broke this, would this test catch it?"*. Document the regression in the test docstring. GPU-only tests go behind a `cuda_only` fixture.
