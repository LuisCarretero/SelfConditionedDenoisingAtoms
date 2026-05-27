# SCD paper: baseline numbers for QM9 HOMO, Matbench mp_gap, LBA

Sources read: `main.tex` (orientation + table inputs), `dataset_summary.tex` (task definitions), `Appendix.tex` (task descriptions), `results_summary.tex` (cross-task summary), `scd_vs_jmp_qm9.tex` (QM9 all targets incl. HOMO), `scd_vs_SSL_qm9.tex` (QM9 vs other SSL), `scd_vs_frad_slide.tex` (QM9 vs Frad/SliDe), `FE_pretrain.tex` (CT-FE-OMOL25 finetune protocol -> HOMO), `mpgap_brief.tex` and `mpgap_full.tex` (Matbench mp_gap), `lba_brief.tex` and `lba_full.tex` (LBA id30/id60), `Hyperparameters.tex` (training regimes). Skipped (out of scope per task brief): `md17_scd.tex` (MD17), `scd_on_conformers.tex` (conformer/UMAP ablation on QM9 HOMO — only SCD-self ablations, not external baselines), `scd_vs_coord.tex` (denoising flavor ablation, internal), `umap_fig.tex`, `umap_mini.tex` (UMAP plots), `arch_speed_mem.tex` (speed/memory only), `intro.tex`, `conclusion.tex`, `FE_pretrain.tex`'s force-energy pretraining metrics on OMol25-4M (`tab:fe_pretrain` reports force/energy MAEs, not the three target tasks — kept only the HOMO numbers from `tab:fe_finetune`).

Ambiguities and notes:
- The paper consistently labels pretraining regimes by prefix: "CT" / "CGT" alone = trained from scratch (no pretraining); "CT-SCD-X" = self-supervised SCD pretraining on dataset X then SFT; "CT-FE-OMOL25" = supervised force-energy pretraining then SFT. JMP-S/L appear in both "No Pretraining" and "FE Pretrained" rows in the mp_gap tables, and `tab:scd_vs_jmp_qm9` reports a single QM9 row per JMP variant — by cross-reference with the mp_gap tables and JMP's own paper this row is the FE-pretrained variant; flagged below.
- HOMO appears in `tab:scd_vs_jmp_qm9` together with all other 10 QM9 targets; per the task brief these extra targets are kept as context in a secondary sub-table, not as primary baselines.
- `results_summary.tex` and `mpgap_brief.tex` give slightly different numbers for CT-SCD-AMP20 on mp_gap (0.122 vs 0.123). Per `mpgap_full.tex` 0.122 is Fold-0 and 0.123 is the mean of folds 0–4 — both reported below.
- For HOMO, `tab:scd_vs_jmp_qm9` reports CT-SCD as "CT-SCD" without a pretraining-dataset suffix; cross-referencing `results_summary.tex` (CT-SCD-PCQ = 12.7 meV) and `scd_vs_SSL_qm9.tex` (CT pretrained on PCQ = 12.7 meV) confirms it is CT-SCD-PCQ.
- LBA EPT-MultiDomain is also written "EPT-Multi" in the brief table — same model.

---

## 1. QM9 HOMO

**Task**: predict the highest occupied molecular orbital (HOMO) energy of small organic molecules from 3D geometry.
**Dataset**: QM9 (134k molecules, cleaned to 130,831), random split 110k/10k/10,831 (train/val/test). No early stopping on val. Geometries from B3LYP/6-31G(2df,p) DFT. (`Appendix.tex`, dataset section)
**Metric**: Mean Absolute Error (MAE) on the test set, reported in **meV**.

### 1a. HOMO-only baselines

| Model | Regime | HOMO MAE (meV) | Source .tex | Notes |
|---|---|---|---|---|
| ET / CT (Baseline) | scratch | 20.3 | `results_summary.tex`, `scd_vs_jmp_qm9.tex`, `FE_pretrain.tex` | Same baseline number across all SCD-paper tables. |
| EquiformerV2 | scratch | 14.0 | `scd_vs_jmp_qm9.tex` | 11.2M params; no-pretraining column. |
| Frad | pretrained+SFT (denoising, on PCQ) | 15.2 | `scd_vs_frad_slide.tex` | GET backbone, 14M-ish. |
| SliDe | pretrained+SFT (denoising, on PCQ) | 13.6 | `scd_vs_frad_slide.tex` | GET backbone. |
| EPT-10 | pretrained+SFT (SSL, on 5.9M structs) | 15.2 | `scd_vs_SSL_qm9.tex` | 30M params; PCQ subset of pretraining. |
| Uni-Corn | pretrained+SFT (SSL, on 15M structs) | 13.0 | `scd_vs_SSL_qm9.tex` | Multi-objective SSL. |
| JMP-S | pretrained+SFT (force-energy, 120M structs) | 11.1 | `scd_vs_jmp_qm9.tex` | 27M params. Regime inferred from cross-ref with mp_gap tables; the QM9 table is single-row. |
| JMP-L | pretrained+SFT (force-energy, 120M structs) | 8.8 | `scd_vs_jmp_qm9.tex` | 235M params. Same caveat as JMP-S. |
| CT-FE-OMol25 | pretrained+SFT (force-energy, OMol25 4M) | 13.6 | `results_summary.tex`, `FE_pretrain.tex` (Reset Head, σ_reg=0) | 10M params. Best FE finetune protocol from `tab:fe_finetune`. |
| CT-SCD-PCQ | pretrained+SFT (SCD-SSL, PCQ 3.4M) | 12.7 | `results_summary.tex`, `scd_vs_jmp_qm9.tex`, `scd_vs_SSL_qm9.tex`, `scd_vs_frad_slide.tex` | SCD's headline small-molecule SSL number. |
| CT-SCD-GEOM10 | pretrained+SFT (SCD-SSL, GEOM10 2.7M) | 13.2 | `results_summary.tex` | |
| CT-SCD-AMP20 | pretrained+SFT (SCD-SSL, AMP20 675k) | 14.6 | `results_summary.tex` | Out-of-domain (materials) pretraining. |
| CT-SCD-SAIR | pretrained+SFT (SCD-SSL, SAIR 4.4M) | 14.6 | `results_summary.tex` | Out-of-domain (protein-ligand) pretraining. |
| CT-SCD-ALL | pretrained+SFT (SCD-SSL, AllAtoms 11.3M) | 12.7 | `results_summary.tex` | Mixed-domain pretraining. |
| CT-SCD-OMol25 | pretrained+SFT (SCD-SSL, OMol25 4M) | 12.8 | `results_summary.tex`, `FE_pretrain.tex` | |
| CGT-SCD (PCQ) | pretrained+SFT (SCD-SSL, PCQ 3.4M) | 9.65 | `scd_vs_jmp_qm9.tex`, `scd_vs_SSL_qm9.tex`, `scd_vs_frad_slide.tex` | 17M params, CGT (Clebsch-Gordan tensor) backbone — separate architecture line. |

### 1b. Other QM9 targets reported alongside HOMO in `tab:scd_vs_jmp_qm9` (context only)

Kept because the task brief asks for additional QM9 targets that *appear in the same table* as HOMO. MAE on test set; units in the row label.

| Target | EquiformerV2 (scratch) | CT Baseline (scratch) | CT-SCD-PCQ (pre+SFT, SCD-SSL) | CGT-SCD-PCQ (pre+SFT, SCD-SSL) | JMP-S (pre+SFT, FE) | JMP-L (pre+SFT, FE) |
|---|---|---|---|---|---|---|
| LUMO (meV) | 13.0 | 17.5 | 11.5 | 9.05 | 10.8 | 8.6 |
| Gap (meV) | 29.0 | 36.1 | 24.5 | 19.7 | 23.1 | 19.1 |
| ZPVE (meV) | 1.47 | 1.84 | 1.18 | 1.22 | 1.0 | 0.9 |
| α (a₀³) | 0.050 | 0.059 | 0.0377 | 0.0383 | 0.037 | 0.032 |
| Cv (cal/mol·K) | 0.023 | 0.026 | 0.021 | 0.019 | 0.018 | 0.017 |
| U₀ (meV) | 6.17 | 6.15 | 3.58 | 3.96 | 3.3 | 2.9 |
| U (meV) | 6.49 | 6.38 | 3.50 | 4.11 | 3.3 | 2.8 |
| H (meV) | 6.22 | 6.16 | 3.52 | 4.00 | 3.3 | 2.8 |
| G (meV) | 7.57 | 7.62 | 5.29 | 5.29 | 4.5 | 4.3 |

(Source: `scd_vs_jmp_qm9.tex`.)

---

## 2. Matbench mp_gap

**Task**: predict the DFT-computed band gap energy of periodic crystal structures.
**Dataset**: Matbench `mp_gap` (106k structures total, ~85k train / 21k test per fold, 5 folds). Most rows in the paper report Fold 0 only; CT-SCD-AMP20 and the strongest no-pretraining baselines also report mean over folds 0–4. (`Appendix.tex`, dataset section)
**Metric**: Mean Absolute Error (MAE), reported in **eV**.

| Model | Regime | MAE (eV) Fold 0 | MAE (eV) Mean 0–4 | Source .tex | Notes |
|---|---|---|---|---|---|
| MODNet | scratch | 0.215 | 0.220 | `mpgap_full.tex` | non-MLIP baseline. |
| coGN | scratch | 0.153 | 0.156 | `mpgap_full.tex` | best no-pretraining baseline. |
| JMP-S | scratch | 0.235 | — | `mpgap_full.tex` | 25.2M params, no pretraining row. |
| JMP-L | scratch | 0.228 | — | `mpgap_full.tex` | 235M params. |
| CT (Baseline) | scratch | 0.186 | — | `mpgap_full.tex`, `results_summary.tex` | SCD's own 10M-param scratch baseline. |
| JMP-S | pretrained+SFT (force-energy, 120M) | 0.119 | 0.121 | `mpgap_full.tex` | 27M params. |
| JMP-L | pretrained+SFT (force-energy, 120M) | 0.089 | 0.091 | `mpgap_full.tex` | 235M params; current SOTA in the paper's listing. |
| HackNIP (Orb-v2 + MODNet) | pretrained+SFT (force-energy, >32M est.) | — | 0.150 | `mpgap_full.tex` | >25.2M params. |
| CT-FE-OMOL25 | pretrained+SFT (force-energy, OMol25 4M) | 0.134 | — | `mpgap_full.tex` | 10M params. |
| Crystal-Twins | pretrained+SFT (SSL, 428k) | — | 0.264 | `mpgap_full.tex` | "Other SSL" category. |
| CT-SCD-AMP20 | pretrained+SFT (SCD-SSL, AMP20 675k) | 0.122 | 0.123 | `mpgap_full.tex`, `mpgap_brief.tex`, `results_summary.tex` | Mean 0.123 in summary table, 0.122 fold-0 in mpgap_brief. |
| CT-SCD-ALL | pretrained+SFT (SCD-SSL, ALL 11.3M) | 0.132 | — | `mpgap_full.tex`, `results_summary.tex` | |
| CT-SCD-PCQ | pretrained+SFT (SCD-SSL, PCQ 3.4M) | 0.174 | — | `mpgap_full.tex`, `results_summary.tex` | Out-of-domain (molecules). |
| CT-SCD-GEOM10 | pretrained+SFT (SCD-SSL, GEOM10 2.8M) | 0.177 | — | `mpgap_full.tex`, `results_summary.tex` | Out-of-domain. |
| CT-SCD-SAIR | pretrained+SFT (SCD-SSL, SAIR 4.4M) | 0.182 | — | `mpgap_full.tex`, `results_summary.tex` | Out-of-domain. |
| CT-SCD-OMOL25 | pretrained+SFT (SCD-SSL, OMol25 4M) | 0.136 | — | `mpgap_full.tex`, `results_summary.tex` | |

---

## 3. LBA (Ligand Binding Affinity)

**Task**: predict ligand binding affinity (−log K_d/K_i) from 3D pocket–ligand complex geometries.
**Dataset**: PDBbind-derived LBA from ATOM3D, 4,463 structures. Two test splits: **id30** (≤30% sequence overlap between train and test proteins) and **id60** (≤60% sequence overlap). (`Appendix.tex`, dataset section)
**Metric**: Test-set **RMSE** (lower is better). Pearson and Spearman correlations also reported in `lba_full.tex` and included below for completeness.

| Model | Regime | id30 RMSE | id30 Pear. | id30 Spear. | id60 RMSE | id60 Pear. | id60 Spear. | Source .tex | Notes |
|---|---|---|---|---|---|---|---|---|---|
| HoloProt-Full Surface | scratch | 1.464 | 0.509 | 0.500 | 1.365 | 0.749 | 0.742 | `lba_full.tex` | 1.4M params. |
| ProtNet-All-Atom | scratch | 1.463 | 0.551 | 0.551 | 1.343 | 0.765 | 0.761 | `lba_full.tex` | Also listed as "ProtNet" in brief table. |
| ATOM3D-3DCNN | scratch | 1.416 | 0.550 | 0.553 | 1.621 | 0.608 | 0.615 | `lba_full.tex` | |
| ATOM3D-ENN | scratch | 1.568 | 0.389 | 0.408 | 1.620 | 0.623 | 0.633 | `lba_full.tex` | |
| ATOM3D-GNN | scratch | 1.601 | 0.545 | 0.533 | 1.408 | 0.743 | 0.743 | `lba_full.tex` | |
| EPT-Scratch | scratch | 1.378 | 0.604 | 0.594 | 1.277 | 0.787 | 0.785 | `lba_full.tex` | 30M params. |
| CT (Baseline) | scratch | 1.510 | 0.501 | 0.486 | 1.386 | 0.736 | 0.720 | `lba_full.tex`, `results_summary.tex` | 10M; SCD's scratch baseline. |
| Frad | pretrained+SFT (denoising, PCQ) | 1.365 | 0.599 | 0.577 | 1.213 | 0.804 | 0.801 | `lba_full.tex` | 14M. |
| EGNN-PLM | pretrained+SFT (sequence PLM) | 1.403 | 0.565 | 0.544 | 1.559 | 0.644 | 0.646 | `lba_full.tex` | 650M params. |
| Uni-Mol | pretrained+SFT (SSL on molecules) | 1.520 | 0.558 | 0.540 | 1.619 | 0.645 | 0.653 | `lba_full.tex` | 47.6M params. |
| ProFSA | pretrained+SFT (unclear; SSL on pocket-ligand) | 1.377 | 0.628 | 0.620 | 1.377 | 0.764 | 0.762 | `lba_full.tex` | 47.6M. Pretraining flavor not specified in SCD paper → flag as `unclear` on flavor. |
| EPT-Molecule | pretrained+SFT (SSL, molecules) | 1.336 | 0.621 | 0.602 | 1.243 | 0.802 | 0.800 | `lba_full.tex` | 30M. |
| EPT-Protein | pretrained+SFT (SSL, proteins) | 1.329 | 0.628 | 0.613 | 1.235 | 0.804 | 0.800 | `lba_full.tex` | 30M. |
| EPT-MultiDomain | pretrained+SFT (SSL, multi-domain) | 1.322 | 0.644 | 0.630 | 1.227 | 0.811 | 0.803 | `lba_full.tex` | 30M. Also called "EPT-Multi" in brief table. |
| ADiT-S | pretrained+SFT (unclear flavor) | 1.337 | 0.626 | 0.618 | 1.413 | 0.740 | 0.740 | `lba_full.tex` | 12M. Pretraining flavor not specified in SCD paper → `unclear`. |
| ADiT-M | pretrained+SFT (unclear flavor) | 1.353 | 0.622 | 0.630 | 1.335 | 0.764 | 0.752 | `lba_full.tex` | 35M. Same caveat. |
| ADiT-L | pretrained+SFT (unclear flavor) | 1.308 | 0.645 | 0.647 | 1.246 | 0.767 | 0.765 | `lba_full.tex` | 253M. Same caveat. |
| CT-FE-OMOL25 | pretrained+SFT (force-energy, OMol25 4M) | 1.391 | 0.575 | 0.564 | 1.187 | 0.814 | 0.811 | `lba_full.tex`, `results_summary.tex` | 10M. |
| CT-SCD-PCQ | pretrained+SFT (SCD-SSL, PCQ) | 1.332 | 0.617 | 0.600 | 1.226 | 0.800 | 0.794 | `lba_full.tex`, `lba_brief.tex`, `results_summary.tex` | id30 from `lba_full.tex` "Pretrained on PCQ" block. |
| CT-SCD-AMP20 | pretrained+SFT (SCD-SSL, AMP20) | 1.408 | 0.584 | 0.533 | 1.219 | 0.803 | 0.800 | `lba_full.tex`, `results_summary.tex` | |
| CT-SCD-GEOM10 | pretrained+SFT (SCD-SSL, GEOM10) | 1.392 | 0.606 | 0.554 | 1.211 | 0.806 | 0.801 | `lba_full.tex`, `results_summary.tex` | |
| CT-SCD-ALL | pretrained+SFT (SCD-SSL, ALL) | 1.372 | 0.594 | 0.578 | 1.218 | 0.802 | 0.798 | `lba_full.tex`, `lba_brief.tex`, `results_summary.tex` | |
| CT-SCD-SAIR-Lig | pretrained+SFT (SCD-SSL, SAIR, ligand-only variant) | 1.412 | 0.566 | 0.547 | 1.283 | 0.779 | 0.770 | `lba_full.tex` | |
| CT-SCD-SAIR | pretrained+SFT (SCD-SSL, SAIR) | 1.337 | 0.617 | 0.599 | 1.196 | 0.810 | 0.802 | `lba_full.tex`, `lba_brief.tex`, `results_summary.tex` | |
| CT-SCD-SAIR-Pocket | pretrained+SFT (SCD-SSL, SAIR + pocket-conditional denoising) | 1.304 | 0.640 | 0.624 | 1.200 | 0.809 | 0.806 | `lba_full.tex`, `lba_brief.tex` | SCD's best id30 result. |
| CT-SCD-OMOL25 | pretrained+SFT (SCD-SSL, OMol25 4M) | 1.389 | 0.586 | 0.571 | 1.175 | 0.817 | 0.816 | `lba_full.tex`, `lba_brief.tex`, `results_summary.tex` | SCD's best id60 result. |
