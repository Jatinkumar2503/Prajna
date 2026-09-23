# PRAJNA Verified Provenance Tables

**Generated:** 2026-09-23T12:02:41.654355+00:00
**Git Commit:** `a081e2274253edfaa034ae8194b484f0a61e2c15` (branch: `main`)
**Working Tree Clean:** `False`
**Command:** `C:\Program Files\Python314\python.exe scripts/generate_results_provenance.py`
**Random Seeds:** `[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`
**Hardware Platform:** `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` on `Windows`

---

## Table: `baseline_comparison`

<!-- PROVENANCE_TABLE_START:baseline_comparison -->
| Model Architecture | Parameters | Single CPU Latency (ms) | Onset Acc (%) [95% CI] | $T_{\text{margin}}$ MAE (s) [95% CI] | Stated Test vs PRAJNA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CUSUM Change-Point Detector** | 0 | 0.0070 | 60.93% [59.46, 62.53] | 8.13 s [7.99, 8.26] | Wilcoxon signed-rank p=0.0019 (d=-13.84) |
| **Logistic Regression + Ridge** | 365 | 0.0956 | 75.73% [74.07, 77.47] | 5.86 s [5.71, 6.04] | Wilcoxon signed-rank p=0.0019 (d=-5.90) |
| **HistGradientBoosting Regressor** | 25000 | 10.9704 | 78.93% [77.60, 80.40] | 3.93 s [3.80, 4.04] | Wilcoxon signed-rank p=0.0019 (d=-2.74) |
| **GRU Forecaster** | 42374 | 0.3852 | 55.00% [50.40, 59.47] | 3.45 s [3.25, 3.64] | Wilcoxon signed-rank p=0.0039 (d=-1.66) |
| **LSTM Forecaster** | 55686 | 0.3970 | 43.87% [36.20, 53.53] | 6.50 s [3.58, 12.04] | Wilcoxon signed-rank p=0.0039 (d=-0.46) |
| **Temporal Transformer** | 40390 | 0.8010 | 68.60% [63.80, 72.00] | 2.74 s [2.36, 3.27] | Wilcoxon signed-rank p=0.4316 (d=-0.38) |
| **PRAJNA Reflex Engine (Ours)** | 24338 | 0.5035 | 74.07% [71.87, 75.87] | 2.46 s [2.21, 2.77] | Reference Architecture (Ours) |
<!-- PROVENANCE_TABLE_END:baseline_comparison -->

## Table: `conditional_coverage`

<!-- PROVENANCE_TABLE_START:conditional_coverage -->
| Condition Type | Condition Slice | Nominal Target (%) | Empirical Coverage (%) | Mean Interval Width (s) |
| :--- | :--- | :---: | :---: | :---: |
| Lead Time | **30s before breach** | 90.0% | **41.67%** | 44.9 s |
| Lead Time | **20s before breach** | 90.0% | **100.00%** | 30.0 s |
| Lead Time | **10s before breach** | 90.0% | **100.00%** | 24.6 s |
| Lead Time | **5s before breach** | 90.0% | **100.00%** | 31.9 s |
| Scenario | **LOCA** | 90.0% | **100.00%** | 37.3 s |
| Scenario | **RIA** | 90.0% | **100.00%** | 31.2 s |
| Scenario | **SBO** | 90.0% | **85.83%** | 29.1 s |
<!-- PROVENANCE_TABLE_END:conditional_coverage -->

## Table: `physics_residual_ablation`

<!-- PROVENANCE_TABLE_START:physics_residual_ablation -->
| Model Architecture | Regularization Formulation | OOD Onset Acc (%) [95% CI] | OOD T_margin MAE (s) [95% CI] | Wilcoxon vs λ_phys=0.0 |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Data-Driven Baseline** | λ_phys = 0.0 (Unconstrained) | **75.60%** [74.20, 77.07] | 2.27s [1.91, 2.77] | Reference (Ours) |
| **Balanced Physics Regularizer** | λ_phys = 0.1 (Dynamic Energy) | **76.67%** [75.13, 78.20] | 3.10s [2.81, 3.44] | p=0.3438 (d=0.32) |
| **Strong Physics Regularizer** | λ_phys = 1.0 (Dynamic Energy) | **76.80%** [75.80, 77.73] | 7.12s [6.98, 7.25] | p=0.1562 (d=0.53) |
| **Matched Non-Physics Regularizer** | Tuned L2 + Smoothness | **75.73%** [74.06, 77.47] | 2.24s [1.86, 2.70] | p=1.0000 (d=0.14) |
<!-- PROVENANCE_TABLE_END:physics_residual_ablation -->

## Table: `sensor_fragility`

<!-- PROVENANCE_TABLE_START:sensor_fragility -->
| Sensor Channel Dropped | Overall Acc (%) | Normal Recall (%) | LOCA Recall (%) | RIA Recall (%) | SBO Recall (%) | Drift Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Full 12 Channels)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Primary Coolant Flow** | 80.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| **Core Thermal Power** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Primary Header Pressure** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Core Exit Temperature** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
<!-- PROVENANCE_TABLE_END:sensor_fragility -->

## Table: `static_heat_balance_monitor`

<!-- PROVENANCE_TABLE_START:static_heat_balance_monitor -->
| Test Case | Applied Sensor Perturbation | First-Law Heat Residual (MWth) | Thermal Mismatch (%) | Diagnostic Outcome |
| :--- | :--- | :---: | :---: | :--- |
| **Steady-State Nominal (No Fault)** | None (0.00) | **0.0000** | 0.00% | **Nominal Normal** |
| **Realistic Sensor Gain Drift** | +1.5% Power Channel Drift over 24h | **19.9700** | 2.64% | **Flagged Advisory Alert (t=14.2h)** |
| **Realistic Thermocouple Step Bias** | +2.5 K Core Exit Offset | **34.0400** | 4.50% | **Flagged Immediate Sensor Bias Alarm** |
<!-- PROVENANCE_TABLE_END:static_heat_balance_monitor -->

## Table: `cross_domain_generalization`

<!-- PROVENANCE_TABLE_START:cross_domain_generalization -->
| Evaluation Protocol | Observed Score (%) | Std Dev (%) | Scientific Finding & Diagnosis |
| :--- | :---: | :---: | :--- |
| **Zero-Shot Transfer (PHWR-220 -> PCTRAN PWR-1000)** | **9.24%** | ±1.83% | Confirms fundamental physics domain gap (D2O vs H2O kinetics) |
| **Supervised Transfer Learning (PCTRAN PWR-1000 Fine-Tuned)** | **84.72%** | ±0.00% | In-domain adaptation delta of +75.48% |
| **LOATO Out-of-Distribution SGTR Detection** | **100.00%** | ±0.00% | Flagged as physical energy balance violation within 2.1s |
<!-- PROVENANCE_TABLE_END:cross_domain_generalization -->

