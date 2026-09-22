# PRAJNA Verified Provenance Tables

**Generated:** 2026-09-22T17:34:28.932729+00:00
**Git Commit:** `672831f02b5735716497946c642e60809bf0dd06` (branch: `main`)
**Working Tree Clean:** `True`
**Command:** `C:\Program Files\Python314\python.exe scripts/generate_results_provenance.py`
**Random Seeds:** `[42, 43, 44, 45, 46]`
**Hardware Platform:** `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` on `Windows`

---

## Table: `baseline_comparison`

<!-- PROVENANCE_TABLE_START:baseline_comparison -->
| Model Architecture | Parameters | Single CPU Latency (ms) | Onset Acc (%) | $T_{\text{margin}}$ MAE (s) | Nuisance Alerts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CUSUM Change-Point Detector** | 0 | 0.0118 | 88.67 ± 1.94% | 9.70 ± 0.21 s | 14 |
| **Logistic Regression + Ridge** | 485 | 0.1539 | 96.00 ± 0.94% | 4.79 ± 0.30 s | 0 |
| **HistGradientBoosting Regressor** | 25000 | 9.4474 | 99.07 ± 0.76% | 1.47 ± 0.16 s | 0 |
| **GRU Forecaster** | 42374 | 0.5706 | 78.00 ± 11.70% | 2.80 ± 0.83 s | 0 |
| **LSTM Forecaster** | 55686 | 0.5613 | 83.87 ± 14.59% | 5.96 ± 8.28 s | 0 |
| **Temporal Transformer** | 40390 | 1.3566 | 94.40 ± 3.79% | 4.93 ± 8.78 s | 0 |
| **PRAJNA Reflex Engine (Ours)** | 24338 | 0.8623 | 96.13 ± 1.10% | 0.85 ± 0.27 s | 0 |
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

