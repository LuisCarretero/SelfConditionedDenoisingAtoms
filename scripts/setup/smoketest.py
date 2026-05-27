"""Env-sanity smoketest: imports, CUDA, the SCD model on a tiny QM9 batch.

Run after `pixi install` on a fresh GPU node to catch env breakage before
launching the 46h finetune. Exercises the same code paths as `train.py`:
build the SCD model from `configs/model_configs/scd.yaml`, push it to GPU,
run one forward + backward on a small synthetic molecular batch.

Usage:  pixi run smoketest
"""

from __future__ import annotations

import sys
from pathlib import Path

# This script imports `models.*` / `data.*`, which live at the repo root. When
# invoked as `python scripts/setup/smoketest.py`, sys.path[0] is the script's
# directory (not the repo root), so the imports fail. Add the repo root explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def check_imports() -> None:
    section("imports")
    from importlib.metadata import version

    for pkg in (
        "numpy",
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torch-cluster",
        "pytorch-lightning",
        "pymatgen",
        "ase",
        "rdkit",
        "huggingface_hub",
        "wandb",
    ):
        try:
            print(f"  {pkg:<22} {version(pkg)}")
        except Exception as e:
            print(f"  {pkg:<22} MISSING ({e})")


def check_cuda() -> None:
    section("CUDA")
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    n = torch.cuda.device_count()
    print(f"  device count: {n}")
    for i in range(n):
        print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")


def check_kernel_available() -> None:
    section("TorchMD-Net neighbor kernel")
    try:
        from models.ET_models.extensions import EXTENSIONS_AVAILABLE

        print(f"  EXTENSIONS_AVAILABLE = {EXTENSIONS_AVAILABLE}")
        if not EXTENSIONS_AVAILABLE:
            print("  --> kernel not built; set noise_in_loader=True in configs, or run:")
            print("      cd models/ET_models && pixi run python setup.py build_ext --inplace")
    except Exception as e:
        print(f"  import failed: {e}")


def check_scd_forward() -> None:
    """One forward+backward on a tiny synthetic batch.

    Builds an SCD model from configs/model_configs/scd.yaml with the QM9 finetune
    hparams that affect head/embedding/derivative wiring. This is the failure
    mode we care about: import-clean but model-construction-broken (mismatched
    kwargs, missing extensions, dtype mismatch with the QM9 batch shape).
    """
    section("SCD forward+backward on tiny synthetic batch")

    import torch
    from torch_geometric.data import Batch, Data

    from models.model_helper import create_model

    # create_model reads model_config (path), then overrides matching keys from this dict.
    # Pass only the keys that influence head/aggregation/wiring; let the YAML provide the rest.
    args = dict(
        model_config="configs/model_configs/scd.yaml",
        prior_model=None,
        set_head_agg=None,
    )

    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)

    model = create_model(args, prior_model=None, mean=mean, std=std)
    model = model.cuda()
    model.train()
    print(f"  model: {type(model).__name__}, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 4 toy molecules, 8 atoms each, on the unit cube.
    n_mols, n_atoms_per = 4, 8
    z = torch.randint(1, 10, (n_mols * n_atoms_per,), dtype=torch.long)
    pos = torch.randn(n_mols * n_atoms_per, 3) * 1.5
    batch = torch.arange(n_mols).repeat_interleave(n_atoms_per)
    y = torch.randn(n_mols, 1)
    data = Batch.from_data_list(
        [
            Data(
                z=z[i * n_atoms_per : (i + 1) * n_atoms_per],
                pos=pos[i * n_atoms_per : (i + 1) * n_atoms_per],
                y=y[i : i + 1],
            )
            for i in range(n_mols)
        ]
    ).to("cuda")
    data.batch = batch.to("cuda")

    out = model(data.z, data.pos, batch=data.batch)
    y_pred = out[0] if isinstance(out, tuple) else out["y"]
    loss = (y_pred.squeeze() - data.y.squeeze()).pow(2).mean()
    loss.backward()
    print(f"  forward y.shape={tuple(y_pred.shape)}  loss={loss.item():.4f}  backward OK")


def main() -> None:
    check_imports()
    check_cuda()
    check_kernel_available()
    check_scd_forward()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
