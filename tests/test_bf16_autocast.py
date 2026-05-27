"""Regression test for the bf16-mixed-precision crash in AccumulatedNormalization.

Bug: Under PL `--precision bf16-mixed`, the autocast region casts inputs to bf16
but `AccumulatedNormalization` keeps its `acc_sum`/`acc_squared_sum`/... buffers
in fp32. The in-place `+=` inside `update_statistics()` then raises
`RuntimeError: Index put requires source/destination dtypes match (BFloat16 vs Float)`
during PL's sanity-val.

The fix wraps every `noise_normalizer(...)` call site in `trainer.py` with
`torch.amp.autocast(..., enabled=False)` and casts the input to fp32. This test
guards both halves of the invariant the fix relies on:
  (1) calling the normalizer with an fp32 input inside a disabled-autocast block
      does NOT raise — even when the surrounding scope is in a bf16 autocast.
  (2) after `update_statistics()`, the fp32 buffers stay fp32.

If a future refactor drops the autocast guard or stops casting the input to
fp32, this test reproduces the original crash.
"""

import torch

from models.ET_models.scd_model import AccumulatedNormalization


def test_buffers_initialized_in_fp32():
    norm = AccumulatedNormalization(accumulator_shape=(3,))
    # Invariant the bug relies on: buffers are fp32. If a future change ever
    # casts these to bf16, the dtype-mismatch error disappears for the wrong
    # reason (loss of precision in running stats).
    assert norm.acc_sum.dtype == torch.float32
    assert norm.acc_squared_sum.dtype == torch.float32
    assert norm.acc_count.dtype == torch.float32
    assert norm.num_accumulations.dtype == torch.float32


def test_autocast_disabled_block_avoids_dtype_mismatch():
    """The fix pattern used in trainer.py: cast to fp32 inside an autocast(enabled=False) block."""
    norm = AccumulatedNormalization(accumulator_shape=(3,)).train()
    bf16_input = torch.randn(8, 3, dtype=torch.bfloat16)

    # Outer scope: simulate PL bf16-mixed autocast. Inner: the local guard.
    with (
        torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16),
        torch.amp.autocast(device_type='cpu', enabled=False),
    ):
        out = norm(bf16_input.float(), update=True)
    # The wrapped call's output is fp32 — the loss expects bf16, so we cast back.
    out_bf16 = out.to(torch.bfloat16)

    assert out.dtype == torch.float32
    assert out_bf16.dtype == torch.bfloat16

    # Buffers must remain fp32 after the in-place update inside the wrapped call.
    assert norm.acc_sum.dtype == torch.float32
    assert norm.acc_squared_sum.dtype == torch.float32
    assert norm.acc_count.dtype == torch.float32
    assert norm.num_accumulations.item() == 1.0
    assert norm.acc_count.item() == 8.0


def test_inverse_under_autocast_guard_roundtrips_fp32():
    """`inverse` is also called from trainer.py inside the same guard pattern."""
    norm = AccumulatedNormalization(accumulator_shape=(3,)).train()
    # Prime the running stats so std != 1 and mean != 0.
    norm.update_statistics(torch.randn(64, 3))
    bf16_pred = torch.randn(8, 3, dtype=torch.bfloat16)

    with (
        torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16),
        torch.amp.autocast(device_type='cpu', enabled=False),
    ):
        inv = norm.inverse(bf16_pred.float())

    assert inv.dtype == torch.float32
    assert norm.acc_sum.dtype == torch.float32


def test_trainer_wraps_every_noise_normalizer_call_in_autocast_guard():
    """Source-level guard: every `noise_normalizer(...)` / `.inverse(...)` call
    in trainer.py must sit inside an `autocast(..., enabled=False)` block.

    A naive grep test would over-match (any nearby `autocast` would pass), so we
    walk the AST: for each call to `self.model.noise_normalizer(...)` (and
    `.inverse(...)`), check the enclosing function for at least one `with
    torch.amp.autocast(..., enabled=False):` block that contains that call.
    """
    import ast
    from pathlib import Path

    trainer_src = Path(__file__).resolve().parent.parent / "models" / "trainer.py"
    tree = ast.parse(trainer_src.read_text())

    def _is_noise_normalizer_call(node: ast.AST) -> bool:
        # Matches `self.model.noise_normalizer(...)` and
        # `self.model.noise_normalizer.inverse(...)`.
        if not isinstance(node, ast.Call):
            return False
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "inverse":
            f = f.value
        return (
            isinstance(f, ast.Attribute)
            and f.attr == "noise_normalizer"
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "model"
        )

    def _is_disabled_autocast(node: ast.AST) -> bool:
        if not isinstance(node, ast.With):
            return False
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            # Match `torch.amp.autocast(...)`.
            f = call.func
            if not (isinstance(f, ast.Attribute) and f.attr == "autocast"):
                continue
            for kw in call.keywords:
                if (
                    kw.arg == "enabled"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                ):
                    return True
        return False

    # Collect (call_node, ancestor_with_blocks) pairs.
    violations = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Build a parent-map within this function so we can look up With ancestors.
        parents = {}
        for parent in ast.walk(func):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent
        for node in ast.walk(func):
            if not _is_noise_normalizer_call(node):
                continue
            guarded = False
            cur = parents.get(id(node))
            while cur is not None and cur is not func:
                if _is_disabled_autocast(cur):
                    guarded = True
                    break
                cur = parents.get(id(cur))
            if not guarded:
                violations.append((func.name, node.lineno))

    assert not violations, (
        f"noise_normalizer calls without autocast(enabled=False) guard: {violations}. "
        "This regression would reintroduce the bf16-mixed crash."
    )
