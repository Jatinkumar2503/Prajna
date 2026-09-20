# PRAJNA: Experimental Protocol & Leakage Prevention Policy

This document establishes the binding scientific rules governing dataset partitioning, leakage prevention, and model evaluation across all PRAJNA experiments.

---

## 1. Zero-Leakage Partitioning Rules

### Rule 1: Trajectory-Level Separation (Never Split by Window)
- **Prohibition:** Splitting sequential sliding windows randomly or chronologically from the same run into train and test sets is strictly prohibited. Adjacent windows share $W - \text{stride}$ overlapping time-steps, leading to massive artificial information leakage.
- **Protocol:** All splits must be partitioned by **complete, independent simulation runs or entire CSV files**:
  - *In-Domain PHWR:* Runs 0–139 for Train, Runs 140–169 for Validation, Runs 170–199 strictly held out for Test.
  - *NPPAD:* Distinct CSV files 0–9 for Train, completely separate CSV files 10–19 strictly held out for Test.

### Rule 2: Normalization Separation
- Feature scaling parameters (mean $\mu$, standard deviation $\sigma$, min/max) must be **fitted strictly on the training partition only**:
  $$\mu_{\text{train}}, \sigma_{\text{train}} = \text{Fit}(X_{\text{train}})$$
  $$X_{\text{train}}^{\text{norm}} = \frac{X_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}, \quad X_{\text{val}}^{\text{norm}} = \frac{X_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}, \quad X_{\text{test}}^{\text{norm}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
- Fitting scalers on the full dataset before splitting is classified as severe data contamination.

### Rule 3: Model-Selection Separation
- The held-out test sets (`phwr_test_unseen.pt`, `nppad_cross_test.pt`, `pur1_real_robustness_test.pt`) are used **strictly once for final evaluation**.
- Hyperparameter tuning, early stopping, and architecture exploration must be performed strictly using the validation split.

---

## 2. Standardized Experimental Protocol

| Experiment | Training Partition | Test Partition | Core Evaluation Focus | Primary Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 01** | In-Domain PHWR (140 runs/scen) | Held-Out PHWR (30 runs/scen) | Clean in-domain baseline | $T_{\text{margin}}$ MAE, Precision/Recall/F1 |
| **Exp 02** | In-Domain PHWR | Held-Out NPPAD Trajectories | Cross-simulator domain gap diagnosis | Per-variable & per-scenario confusion matrix |
| **Exp 03** | In-Domain PHWR | PUR-1 Real Reactor Telemetry | Real-world noise robustness | Normal stability & false trip rate |
| **Exp 04** | PHWR vs PHWR + NPPAD Trajectories | Disjoint Held-Out NPPAD Trajectories | Independent simulator exposure benefit | Held-out cross-domain accuracy delta ($\Delta\%$) |
| **Exp 05** | In-Domain PHWR | Degraded / Missing Telemetry | Sensor fault tolerance & channel attribution | Single-channel attribution drop curve |
| **Exp 06** | Temporal vs Physics-Constrained | Clean & Degraded PHWR | Value of physics constraints | Energy residual RMS & noisy state forecast MAE |
| **Exp 07** | In-Domain PHWR | Progressive Accident Transients | Early-warning lead time & $T_{\text{margin}}$ bounds | Lead time $\Delta t_{\text{lead}}$ before physical trip |
| **Exp 08** | In-Domain PHWR & Baselines | Identical Test Sets ($H=15\text{s}$) | PRAJNA vs GRU / LSTM / Transformer | Standardized binary boundary crossing F1 & Recall |
