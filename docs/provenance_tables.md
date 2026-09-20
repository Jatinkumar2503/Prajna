# PRAJNA Verified Provenance Tables

**Generated:** 2026-09-20T18:34:15.870590+00:00
**Git Commit:** `5ed5f1706aaed129519a4355860b90f90184f829` (branch: `main`)
**Working Tree Clean:** `True`
**Command:** `C:\Program Files\Python314\python.exe scripts/generate_results_provenance.py`
**Random Seeds:** `[42, 43, 44, 45, 46]`
**Hardware Platform:** `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` on `Windows`

---

## Table: `baseline_comparison`

<!-- PROVENANCE_TABLE_START:baseline_comparison -->
| Model Architecture | Parameters | Single CPU Latency (ms) | Onset Acc (%) | $T_{\text{margin}}$ MAE (s) | Nuisance Alerts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CUSUM Change-Point Detector** | 0 | 0.0042 | 82.40 ± 0.80% | 14.99 ± 0.23 s | 14 |
| **Logistic Regression + Ridge** | 905 | 0.0681 | 100.00 ± 0.00% | 2.98 ± 0.19 s | 0 |
| **HistGradientBoosting Regressor** | 15000 | 5.9690 | 99.60 ± 0.53% | 0.53 ± 0.32 s | 0 |
| **GRU Forecaster** | 16517 | 0.3069 | 100.00 ± 0.00% | 6.26 ± 4.31 s | 0 |
| **LSTM Forecaster** | 26629 | 0.3345 | 100.00 ± 0.00% | 4.25 ± 4.30 s | 0 |
| **Temporal Transformer** | 40390 | 0.6608 | 100.00 ± 0.00% | 1.31 ± 0.48 s | 0 |
| **PRAJNA Reflex Engine (Ours)** | 24338 | 0.4238 | 100.00 ± 0.00% | 4.72 ± 5.00 s | 0 |
<!-- PROVENANCE_TABLE_END:baseline_comparison -->

## Table: `conditional_coverage`

<!-- PROVENANCE_TABLE_START:conditional_coverage -->
| Condition Type | Condition Slice | Nominal Target (%) | Empirical Coverage (%) | Mean Interval Width (s) |
| :--- | :--- | :---: | :---: | :---: |
| Lead Time | **30s before breach** | 90.0% | **41.67%** | 28.4 s |
| Lead Time | **20s before breach** | 90.0% | **100.00%** | 14.2 s |
| Lead Time | **10s before breach** | 90.0% | **100.00%** | 8.1 s |
| Lead Time | **5s before breach** | 90.0% | **100.00%** | 4.6 s |
| Scenario | **LOCA** | 90.0% | **100.00%** | 12.3 s |
| Scenario | **RIA** | 90.0% | **100.00%** | 11.8 s |
| Scenario | **SBO** | 90.0% | **85.83%** | 13.6 s |
<!-- PROVENANCE_TABLE_END:conditional_coverage -->

## Table: `physics_residual_ablation`

<!-- PROVENANCE_TABLE_START:physics_residual_ablation -->
| Model Architecture | Physics Loss Weight | Dynamic Residual Error (MWth) | Nuisance Advisory Alerts | Safety Limit Violations (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Model A (Pure Neural Forecaster)** | 0.0 | **187.95** | 2 | 4.0% |
| **Model B (Physics-Regularized Neural)** | 1.0 | **78.94** | 0 | 0.0% |
| **Model C (Hybrid Physics-Gated Reflex)** | 1.0 | **78.94** | 0 | 0.0% |
<!-- PROVENANCE_TABLE_END:physics_residual_ablation -->

## Table: `sensor_fragility`

<!-- PROVENANCE_TABLE_START:sensor_fragility -->
| Sensor Channel Dropped | Overall Acc (%) | Normal Recall (%) | LOCA Recall (%) | RIA Recall (%) | SBO Recall (%) | Drift Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Full 12 Channels)** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Primary Coolant Flow** | 80.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| **Core Thermal Power** | 80.0% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| **Primary Header Pressure** | 80.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| **Core Exit Temperature** | 80.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |
<!-- PROVENANCE_TABLE_END:sensor_fragility -->

## Table: `static_heat_balance_monitor`

<!-- PROVENANCE_TABLE_START:static_heat_balance_monitor -->
| Test Case | Applied Sensor Perturbation | First-Law Heat Residual (MWth) | Thermal Mismatch (%) | Diagnostic Outcome |
| :--- | :--- | :---: | :---: | :--- |
| **Steady-State Nominal (No Fault)** | None (0.00) | **0.0000** | 0.00% | **Nominal Normal** |
| **Realistic Sensor Gain Drift** | +1.5% Power Channel Drift over 24h | **19.9702** | 2.64% | **Flagged Advisory Alert (t=14.2h)** |
| **Realistic Thermocouple Step Bias** | +2.5 K Core Exit Offset | **34.0371** | 4.50% | **Flagged Immediate Sensor Bias Alarm** |
<!-- PROVENANCE_TABLE_END:static_heat_balance_monitor -->

## Table: `cross_domain_generalization`

<!-- PROVENANCE_TABLE_START:cross_domain_generalization -->
| Evaluation Protocol | Observed Score (%) | Std Dev (%) | Scientific Finding & Diagnosis |
| :--- | :---: | :---: | :--- |
| **Zero-Shot Transfer (PHWR-220 -> PCTRAN PWR-1000)** | **9.19%** | ±1.83% | Confirms fundamental physics domain gap (D2O vs H2O kinetics) |
| **Supervised Transfer Learning (PCTRAN PWR-1000 Fine-Tuned)** | **84.67%** | ±0.00% | In-domain adaptation delta of +75.48% |
| **LOATO Out-of-Distribution SGTR Detection** | **100.00%** | ±0.00% | Flagged as physical energy balance violation within 2.1s |
<!-- PROVENANCE_TABLE_END:cross_domain_generalization -->

