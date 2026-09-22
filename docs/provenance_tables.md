# PRAJNA Verified Provenance Tables

**Generated:** 2026-09-22T17:48:58.574499+00:00
**Git Commit:** `7f9c9f41323da9f4e7f3eaba786b911b8ae891d5` (branch: `main`)
**Working Tree Clean:** `False`
**Command:** `C:\Program Files\Python314\python.exe scripts/generate_results_provenance.py`
**Random Seeds:** `[42, 43, 44, 45, 46]`
**Hardware Platform:** `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` on `Windows`

---

## Table: `baseline_comparison`

<!-- PROVENANCE_TABLE_START:baseline_comparison -->
| Model Architecture | Parameters | Single CPU Latency (ms) | Onset Acc (%) | $T_{\text{margin}}$ MAE (s) | Nuisance Alerts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CUSUM Change-Point Detector** | 0 | 0.0045 | 61.07 ± 2.93% | 8.18 ± 0.19 s | 14 |
| **Logistic Regression + Ridge** | 365 | 0.0786 | 75.20 ± 3.14% | 6.01 ± 0.33 s | 0 |
| **HistGradientBoosting Regressor** | 25000 | 7.0896 | 78.00 ± 2.91% | 4.00 ± 0.21 s | 0 |
| **GRU Forecaster** | 42374 | 0.3247 | 59.47 ± 5.02% | 3.47 ± 0.16 s | 0 |
| **LSTM Forecaster** | 55686 | 0.3433 | 42.67 ± 12.80% | 3.66 ± 0.51 s | 0 |
| **Temporal Transformer** | 40390 | 0.7291 | 68.27 ± 8.16% | 2.99 ± 1.09 s | 0 |
| **PRAJNA Reflex Engine (Ours)** | 24338 | 0.4258 | 74.14 ± 3.18% | 2.62 ± 0.57 s | 0 |
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
| Model Architecture | Physics Loss Weight | Dynamic Residual Error (MWth) | Nuisance Advisory Alerts | Safety Limit Violations (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Model A (Pure Neural Forecaster)** | 0.0 | **187.95** | 0 | 4.0% |
| **Model B (Physics-Regularized Neural)** | 1.0 | **78.94** | 0 | 0.0% |
| **Model C (Hybrid Physics-Gated Reflex)** | 1.0 | **78.94** | 0 | 0.0% |
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

