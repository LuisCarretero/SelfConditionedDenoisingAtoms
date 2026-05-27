"""Regression tests for the wall-clock extrapolation in TrainTiming.

The Phase 2.5 extension to TrainTiming produces the headline "X GPU-h vs the
paper's 46 GPU-h" number. If the formula gets a factor wrong — world_size
applied twice, overhead added with the wrong sign, num_epochs and num_steps
mixed up in `_planned_train_epochs` — the comparison silently lies. These
tests exercise `_extrapolate()` directly by priming the internal buffers, so
they don't need a GPU or a real training loop.
"""

import pytest

from models.callbacks import TrainTiming


def _prime(tt: TrainTiming, step_s: float, epoch_wall_s: float, steps_per_epoch: int) -> None:
    # Skip the warmup gate so the buffers count for `_extrapolate()`.
    tt._buf = [step_s] * 10
    tt._epoch_wall_buf = [epoch_wall_s] * 3
    tt._steps_per_epoch = steps_per_epoch
    tt._seen = tt.warmup_batches + 1


def test_extrapolate_compute_only_matches_legacy_formula():
    # If no epoch-wall data is captured (e.g. probe stops mid-epoch-1) the
    # wall-clock number must fall back to the compute-only formula — otherwise
    # the cells in finetune-speedup-worklog.md silently change scale.
    tt = TrainTiming(num_steps=3000, batch_size=128, world_size=1)
    tt._buf = [0.1] * 10
    tt._seen = tt.warmup_batches + 1
    step_s, compute_h, wall_h = tt._extrapolate()
    assert step_s == pytest.approx(0.1)
    assert compute_h == pytest.approx(0.1 * 3000 / 3600.0)
    assert wall_h == compute_h


def test_extrapolate_wall_h_adds_overhead_and_test():
    # Compute uses num_steps; overhead uses (epoch_wall - step*steps_per_epoch)
    # times the *planned* number of train epochs (here 349, from num_epochs).
    # World size multiplies the wall-clock sum, not the overhead-per-epoch.
    # Both gates aligned (860 × 349 = 300_140) so min() doesn't shift the count.
    tt = TrainTiming(
        num_steps=300_140,
        batch_size=128,
        world_size=4,
        num_epochs=349,
        val_interval=1,
    )
    _prime(tt, step_s=0.1, epoch_wall_s=100.0, steps_per_epoch=860)
    tt._test_epoch_s = 30.0

    step_s, compute_h, wall_h = tt._extrapolate()
    expected_compute = 0.1 * 300_140 / 3600.0 * 4
    overhead_per_epoch_s = 100.0 - 0.1 * 860  # = 14.0
    expected_wall = (
        (0.1 * 300_140 + overhead_per_epoch_s * 349) * 4 / 3600.0
        + 30.0 * 4 / 3600.0
    )

    assert step_s == pytest.approx(0.1)
    assert compute_h == pytest.approx(expected_compute)
    assert wall_h == pytest.approx(expected_wall)
    # Sanity: wall must be at least compute. If a refactor flips a sign this fails first.
    assert wall_h > compute_h


def test_extrapolate_overhead_clamped_at_zero():
    # If measurement noise gives an epoch wall *shorter* than steps × step_s,
    # the overhead would go negative and silently shrink the wall extrapolation
    # below the compute floor. Must clamp at zero.
    tt = TrainTiming(num_steps=3000, batch_size=128, world_size=1, num_epochs=10)
    _prime(tt, step_s=0.1, epoch_wall_s=10.0, steps_per_epoch=860)  # 10 < 0.1*860=86

    _, compute_h, wall_h = tt._extrapolate()
    assert wall_h == pytest.approx(compute_h)


def test_planned_train_epochs_takes_min_of_gates():
    # Lightning stops at min(max_steps, max_epochs). _planned_train_epochs must
    # mirror that — using only num_epochs would over-extrapolate when num_steps
    # is the tighter gate (or vice versa).
    tt_steps_tighter = TrainTiming(
        num_steps=3000, batch_size=128, world_size=1, num_epochs=349
    )
    tt_steps_tighter._steps_per_epoch = 860
    assert tt_steps_tighter._planned_train_epochs() == pytest.approx(3000 / 860)

    tt_epochs_tighter = TrainTiming(
        num_steps=300_000, batch_size=128, world_size=1, num_epochs=10
    )
    tt_epochs_tighter._steps_per_epoch = 860
    assert tt_epochs_tighter._planned_train_epochs() == 10
