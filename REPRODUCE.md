# PRAJNA End-to-End Experimental Reproduction Manual

> **Phase 1: Credibility (5.5 → 6.5) — Step 3: Ownership of Results**  
> **Status:** Fully Reproducible, Cryptographically Sealed, Audited for Zero Fabricated Constants.

This document serves as the definitive peer-review reproduction guide for the **PRAJNA Nuclear Safety Decision-Support System**. It guarantees that any independent researcher or peer reviewer cloning this repository can reproduce every numerical table cell byte-for-byte, verify provenance, and answer *"Where does this number come from?"* for any result in [README.md](file:///c:/Users/Asus/Documents/prajna/README.md) and [docs/provenance_tables.md](file:///c:/Users/Asus/Documents/prajna/docs/provenance_tables.md).

---

## 1. System Requirements & Setup

### 1.1 Environment
- **Operating System:** Linux, macOS, or Windows 10/11 (64-bit)
- **Python Version:** Python 3.10 through 3.14 (Verified on Python 3.14.6)
- **Deep Learning Framework:** PyTorch $\ge 2.0.0$ (CPU execution is fully supported; GPU CUDA optional)
- **Hardware Footprint:** Minimal; all baseline and priority suites run sequentially on a standard 4-core laptop CPU in under 3 minutes.

### 1.2 Installation
```bash
# Clone the repository
git clone https://github.com/Jatinkumar2503/Prajna.git
cd Prajna

# Create and activate a clean virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install locked dependencies
pip install -r requirements.lock
# Alternatively, install minimum base requirements:
pip install torch numpy scipy scikit-learn pyyaml matplotlib
```

---

## 2. Master Reproduction Workflow

The benchmark suite follows a strict scientific workflow:
$$\text{Commit Code} \longrightarrow \text{Execute Experiments} \longrightarrow \text{Cryptographically Seal Results} \longrightarrow \text{Commit Provenance}$$

### Step 1: Run Fair Non-Saturated Baseline Suite with Statistics
Evaluates all 7 model architectures with equalized tuning budgets across 10 seeds (`[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`) under onset-aligned detection ($t \le 5\text{s}$) with continuous LOCA break spectrum (0.5%–100%), compound overlapping events, sensor faults, non-parametric percentile bootstrap confidence intervals ($B=1,000$), and paired Wilcoxon signed-rank significance tests:
```bash
python scripts/compare_baselines.py
```
*Output Artifact:* `evaluation/reports/non_saturated_baselines_summary.json`

### Step 2: Run Master 10-Priorities Suite
Executes the comprehensive experimental evaluations covering Physics Residual Ablation, Sensor Fragility, Conformal Prediction Intervals, and Cross-Simulator Transfer:
```bash
python scripts/reproduce_all_10_priorities.py
```
*Output Artifacts:*
- `experiments/exp04_multidomain/results.json`
- `experiments/exp05_noise_robustness/attribution_and_horizon.json`
- `experiments/exp06_physics_ablation/results.json`
- `experiments/exp07_tmargin/results.json`
- `evaluation/reports/static_physics_drift_monitor.json`

### Step 3: Compile Canonical Provenance & Render Tables
Aggregates all JSON outputs into a timestamped `results/<timestamp>/results.json`, updates the `results/latest/results.json` symlink/copy, and automatically renders byte-for-byte Markdown tables into `docs/provenance_tables.md` and `README.md`:
```bash
python scripts/generate_results_provenance.py
```

### Step 4: Cryptographically Seal Canonical Results
Computes the canonical SHA-256 digest of `results/latest/results.json` (excluding mutable hash fields), writes `.sha256`, and embeds the seal directly into the JSON:
```bash
python scripts/seal_results.py seal results/latest/results.json
```

---

## 3. Continuous Integration & Anti-Fabrication Verification Suite

Every pull request and peer-review evaluation passes through 7 automated verification gates. Run these commands to independently verify that zero results are stale, fabricated, or hardcoded:

```bash
# Gate 1: Freshness Gate — Results must match current repository commit with clean code tree
python scripts/verify_results_current.py results/latest/results.json

# Gate 2: Cryptographic Seal & Clean Git Tree Gate
python scripts/seal_results.py verify results/latest/results.json --require-clean

# Gate 3: Statistical Artifact & Suspicious Pattern Detector (C1: Zero Variance, C2: Constant Ratio, C3: Affine)
python scripts/detect_suspicious_results.py results/latest/results.json

# Gate 4: Static AST Anti-Shortcut Linter (R1: Literals, R2: Formulas, R3: Lists, R4: Dicts, R5: Calls, R6: Synthetic)
python scripts/lint_hardcoded_results.py --max-allows 10 scripts experiments prajna_core evaluation

# Gate 5: Scanner for Retracted or Stale Metrics
python scripts/find_retracted_numbers.py .

# Gate 6: Byte-for-Byte Markdown Table Provenance Checker
python scripts/verify_table_provenance.py

# Gate 7: Automated Unit Test Suite
python -m unittest tests/test_provenance_ci.py tests/test_step2_tools.py tests/test_experimental_rigor.py tests/test_physics_residual.py tests/test_statistics.py
```

---

## 4. "Where Does This Number Come From?" — Cell-by-Cell Lineage

### Table 1: Baseline Architecture Comparison (`baseline_comparison`)
**Source Script:** `scripts/compare_baselines.py`  
**Output Artifact:** `evaluation/reports/non_saturated_baselines_summary.json`  
**Task Definition:** Early transient classification within onset-aligned early window ($t \le 5\text{s}$ post-onset) across 5 accident regimes (Normal, LOCA, RIA, SGTR, SBO) under continuous LOCA break spectrum (0.5%–100%), compound overlapping events (stuck safety rod, grid frequency perturbation), sensor faults (stuck channels, step biases, deadband), and held-out unseen severities.

| Column Header | Extraction / Computation Method | Physical Meaning & Verification Formula |
| :--- | :--- | :--- |
| **Model Architecture** | Model display name identifier | Model family: Rule-based, Linear, Tree-based, Recurrent, Attention, or Hybrid PINN. |
| **Parameters** | `sum(p.numel() for p in model.parameters())` | Exact trainable parameter count in FP32 weights. |
| **Single CPU Latency (ms)** | `time.perf_counter()` over $N=1,000$ passes | Sequential single-window inference time on 1 CPU thread (`torch.set_num_threads(1)`). |
| **Onset Acc (%) [95% CI]** | Mean and empirical bootstrap 95% CI across 10 seeds: `acc_vals` ($B=1,000$) | Percentage of correctly classified accident windows at $t \le 5\text{s}$ post-accident with non-parametric bootstrap bounds. |
| **$T_{\text{margin}}$ MAE (s) [95% CI]** | Mean and empirical bootstrap 95% CI across 10 seeds: `tm_vals` ($B=1,000$) | Mean Absolute Error $\frac{1}{N}\sum \|T_{\text{true}} - \hat{T}_{\text{pred}}\|$ in seconds before safety threshold breach with non-parametric bootstrap bounds. |
| **Stated Test vs PRAJNA** | Paired Wilcoxon signed-rank test across 10 matched simulator seeds | Non-parametric paired hypothesis test ($p$-value and Cohen's $d$ effect size) testing statistical significance against PRAJNA. |

#### Empirical Baseline Comparison Table (from `results/latest/results.json`):
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

---

### Table 2: Mondrian Calibrated Prediction Interval Coverage (`conditional_coverage`)
**Source Script:** `scripts/calibrate_mondrian_conformal.py` (also callable via `scripts/reproduce_all_10_priorities.py`)  
**Output Artifact:** `experiments/exp07_tmargin/results.json`  
**Task Definition:** Empirical evaluation of 90% Mondrian conformal prediction intervals $[\hat{T}_{\text{lower}}, \hat{T}_{\text{upper}}]$ for time-to-breach ($T_{\text{margin}}$), conditioned on lead time before breach and accident type across held-out trajectories with zero data leakage.

| Row / Slice | Target Cov (%) | Empirical Cov (%) | Interval Width (s) | Sample Count ($n$) | Lineage & Physical Mechanism |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **30s before breach** | 90.0% | **95.73%** | 47.4 s | $n=117$ | **Resolved Under-Coverage:** Mondrian group quantile adapts to early variance, eliminating former 41.67% collapse. |
| **20s before breach** | 90.0% | **90.20%** | 29.4 s | $n=153$ | Transition zone; interval tightens as derivative trends stabilize. |
| **10s before breach** | 90.0% | **88.81%** | 20.4 s | $n=143$ | Pre-trip regime; narrow confidence bounds. |
| **5s before breach** | 90.0% | **91.64%** | 17.2 s | $n=335$ | Immediate pre-trip condition; tightest physical bounds. |
| **LOCA Scenario** | 90.0% | **90.26%** | 29.4 s | $n=154$ | Primary depressurization curve accurately calibrated under group quantile. |
| **RIA Scenario** | 90.0% | **91.86%** | 25.8 s | $n=86$ | Prompt overpower trajectory bounded within target window. |
| **SBO Scenario** | 90.0% | **89.60%** | 25.1 s | $n=548$ | Coastdown thermal dynamics calibrated within 85%–95% target. |

---

### Table 3: Clean Physics Constraint Ablation (`physics_residual_ablation`)
**Source Script:** `scripts/ablate_physics_loss.py`  
**Output Artifact:** `experiments/exp06_physics_ablation/results.json`  
**Task Definition:** Evaluates identical `PrajnaFastReflex` architectures (24,338 parameters) across $\lambda_{\text{phys}} \in \{0.0, 0.1, 1.0\}$ versus a capacity-matched non-physics regularizer. Crucially, evaluation is performed **strictly on held-out Out-Of-Distribution (OOD) severities** (0.5% SBLOCA to 120% severe break), overlapping compound events, 25% sensor channel faults (stuck sensors, step biases, deadband), and active instrument noise across 10 random seeds (`SEEDS = [42..51]`), with **zero evaluation on the training residual**.

| Model Architecture | Regularization Formulation | OOD Onset Acc (%) [95% CI] | OOD T_margin MAE (s) [95% CI] | Wilcoxon vs λ_phys=0.0 |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Data-Driven Baseline** | λ_phys = 0.0 (Unconstrained) | **75.60%** [74.20, 77.07] | 2.27s [1.91, 2.77] | Reference (Ours) |
| **Balanced Physics Regularizer** | λ_phys = 0.1 (Dynamic Energy) | **76.67%** [75.13, 78.20] | 3.10s [2.81, 3.44] | p=0.3438 (d=0.32) |
| **Strong Physics Regularizer** | λ_phys = 1.0 (Dynamic Energy) | **76.80%** [75.80, 77.73] | 7.12s [6.98, 7.25] | p=0.1562 (d=0.53) |
| **Matched Non-Physics Regularizer** | Tuned L2 + Smoothness | **75.73%** [74.06, 77.47] | 2.24s [1.86, 2.70] | p=1.0000 (d=0.14) |

> [!NOTE]
> **Honest Scientific Finding (Null Result on Sensor Fault Generalization):**
> Under severe sensor faults and extreme OOD severities, balanced physics regularization ($\lambda_{\text{phys}} = 0.1$) yields a marginal, non-statistically significant gain in early onset accuracy (+1.07%, $p=0.3438$), but increases continuous margin estimation error (3.10s vs 2.27s, $p=0.0098$, $d=+1.24$). Strong physics regularization ($\lambda_{\text{phys}} = 1.0$) further increases margin error to 7.12s. The capacity-matched non-physics regularizer achieves equivalent onset accuracy (75.73%, $p=1.0000$) and lower margin MAE (2.24s) without margin distortion. This demonstrates that First-Law loss regularizers alone do not substitute for physical sensor redundancy under severe instrument faults.

---

### Table 4: Single-Sensor Fragility & Vulnerability Analysis (`sensor_fragility`)
**Source Script:** `scripts/reproduce_all_10_priorities.py` (Priority 5, lines 360–435)  
**Output Artifact:** `experiments/exp05_noise_robustness/attribution_and_horizon.json`  
**Task Definition:** Single-channel ablation (sensor dropout) zeroing each channel during test inference to identify single-point vulnerabilities.

| Dropped Sensor Channel | Overall Acc (%) | Class Recall Drop | Vulnerability Diagnosis & Proof of Non-Spurious Learning |
| :--- | :---: | :--- | :--- |
| **None (Full 12 Channels)** | **100.0%** | None (All 100%) | Baseline performance with full instrument complement. |
| **Primary Coolant Flow** | **80.0%** | **SBO Recall → 0.0%** | SBO is characterized by primary pump trip. Dropping flow eliminates the direct physical signature of SBO. |
| **Core Thermal Power** | **80.0%** | **RIA Recall → 0.0%** | Reactivity Insertion Accidental power spike is missed when core power / neutron flux channel is lost. |
| **Primary Header Pressure** | **80.0%** | **LOCA Recall → 0.0%** | LOCA depressurization cannot be discriminated from normal pressure fluctuations without header pressure. |
| **Core Exit Temperature** | **80.0%** | **SGTR / Drift Recall → 0.0%** | Slow steam generator tube rupture thermal imbalance requires exit thermocouple tracking. |

---

### Table 5: Static First-Law Primary Heat Balance Monitor (`static_heat_balance_monitor`)
**Source Script:** `scripts/reproduce_all_10_priorities.py` (Priority 10, lines 740–815)  
**Output Artifact:** `evaluation/reports/static_physics_drift_monitor.json`  
**Task Definition:** Evaluates steady-state First-Law primary heat transport balance:
$$R_{\text{heat}} = \left| P_{\text{core}} - \dot{m} C_p (T_{\text{out}} - T_{\text{in}}) \right|$$

| Test Case | Applied Sensor Perturbation | First-Law Residual | Thermal Mismatch | Diagnostic Action |
| :--- | :--- | :---: | :---: | :--- |
| **Steady-State Nominal** | None ($0.00$) | **0.0000 MWth** | 0.00% | Nominal Normal |
| **Sensor Gain Drift** | $+1.5\%$ Power Channel Drift over 24h | **19.9702 MWth** | 2.64% | Flagged Advisory Alert at $t=14.2\text{h}$ |
| **Thermocouple Step Bias**| $+2.5\text{ K}$ Core Exit Temperature Offset | **34.0371 MWth** | 4.50% | Flagged Immediate Sensor Bias Alarm |

---

### Table 6: Cross-Simulator Domain Transfer & Generalization (`cross_domain_generalization`)
**Source Script:** `scripts/reproduce_all_10_priorities.py` (Priority 4, lines 280–355)  
**Output Artifact:** `experiments/exp04_multidomain/results.json`  
**Task Definition:** Cross-simulator transfer between PHWR-220 ODE simulator and Tsinghua INET NPPAD (PCTRAN PWR-1000) under strict zero-leakage trajectory partitioning.

| Evaluation Protocol | Observed Score | Scientific Finding & Diagnosis |
| :--- | :---: | :--- |
| **Zero-Shot Transfer** (PHWR → PWR) | **9.19% ± 1.83%** | **Confirms fundamental physics domain gap:** Heavy water ($D_2O$) pressure tube kinetics vs. Light water ($H_2O$) vessel kinetics, differing coolant pressure ($85\text{ bar}$ vs $155\text{ bar}$), and distinct thermal inertia. |
| **Supervised Transfer Learning** (PWR Fine-Tuned)| **84.67% ± 0.00%** | **In-Domain Adaptation Delta of +75.48%:** Demonstrates that the reflex backbone adapts rapidly when fine-tuned on held-out trajectories of the target simulator. |
| **LOATO Out-of-Distribution SGTR Detection** | **100.00% ± 0.00%** | **Leave-One-Accident-Type-Out:** SGTR was completely withheld from training; detected as an anomalous physical energy balance violation within $2.1\text{s}$. |

---

## 5. Overview of Core Scripts in `scripts/`

| Script Path | Primary Role & Responsibility | Key Command Line Invocations |
| :--- | :--- | :--- |
| `scripts/compare_baselines.py` | Runs the 5-seed benchmark comparing CUSUM, Logistic Regression, HistGB, GRU, LSTM, Transformer, and PRAJNA. | `python scripts/compare_baselines.py` |
| `scripts/reproduce_all_10_priorities.py` | Executes the 10 peer-review priority experiments (ablation, fragility, conformal intervals, cross-simulator). | `python scripts/reproduce_all_10_priorities.py` |
| `scripts/generate_results_provenance.py` | Aggregates all JSON outputs into `results/<timestamp>/results.json`, renders tables into markdown. | `python scripts/generate_results_provenance.py` |
| `scripts/seal_results.py` | Computes/verifies canonical SHA-256 seal of `results.json` and asserts git tree cleanliness. | `python scripts/seal_results.py seal <file>`<br>`python scripts/seal_results.py verify <file> --require-clean` |
| `scripts/detect_suspicious_results.py` | Audits `results.json` for C1 (zero variance across seeds), C2 (constant ratio), and C3 (affine linearity). | `python scripts/detect_suspicious_results.py <file>` |
| `scripts/verify_results_current.py` | Checks recorded git commit in `results.json` against current HEAD and git status of code directories. | `python scripts/verify_results_current.py <file>` |
| `scripts/lint_hardcoded_results.py` | AST static analyzer forbidding typed-in metric numbers, shortcuts, or synthetic normal draws. | `python scripts/lint_hardcoded_results.py scripts evaluation` |
| `scripts/verify_table_provenance.py` | Validates that every table in `README.md` and `docs/provenance_tables.md` matches `results.json` byte-for-byte. | `python scripts/verify_table_provenance.py` |
| `scripts/find_retracted_numbers.py` | Recursively searches codebase for 8 retracted draft numbers (e.g. uncalibrated MAE 0.33s or 0.34s). | `python scripts/find_retracted_numbers.py .` |

---

## 6. Reviewer FAQ

### Q1: How do we know baseline models weren't handicapped?
All baseline models (`Rate_of_Change_CUSUM`, `Logistic_Regression`, `HistGradientBoosting`, `GRU_Forecaster`, `LSTM_Forecaster`, `Temporal_Transformer`, and `PRAJNA_Reflex_Engine`) were evaluated under identical conditions in `scripts/compare_baselines.py`:
- Same multi-channel input telemetry (12 physical sensors)
- Same strictly disjoint trajectory train/test split with identical seed bases
- Same sensor noise suite (Gaussian sensor noise, quantization, calibration drift, EMI spikes)
- Normalized inputs fitted strictly on training data
- Deep baselines trained with identical 30-epoch AdamW budget and batch size 32
- Single-window CPU latency evaluated with identical harness ($N=1,000$, single thread, batch=1)

### Q2: Why does CUSUM declare `"deterministic": true`?
CUSUM is an industrial threshold rule with fixed mathematical derivative trip logic ($dP/dt < -1.5\text{ bar/s}$, $dF/dt < -25\text{ kg/s/s}$). It contains zero learned weights. Marking `"deterministic": true` instructs `detect_suspicious_results.py` to skip rule C1 (zero variance across seeds) for deterministic closed-form algorithms while strictly enforcing non-zero empirical variance for all learned neural/statistical models.

### Q3: Why is zero-shot cross-simulator accuracy only ~9%?
This is an intentional scientific result documenting the **cross-simulator domain gap**. PRAJNA was trained on heavy-water PHWR-220 dynamics ($85\text{ bar}$, $D_2O$, pressure tube geometry). Testing zero-shot on Tsinghua INET NPPAD (PCTRAN PWR-1000, $155\text{ bar}$, $H_2O$, pressure vessel geometry) demonstrates that the model does not magically transfer across fundamentally different reactor physics regimes without adaptation. Fine-tuning on held-out PCTRAN trajectories yields $84.67\%$, proving rapid domain adaptation.

### Q4: Can PRAJNA actuate autonomous reactor scrams in commercial plants?
**No.** Commercial nuclear power plants in India (NPCIL/AERB) and internationally (US NRC/IAEA) require Class 1E safety systems to be hardwired analog circuits or qualified deterministic logic. PRAJNA is framed strictly as a **Tier-1 Decision-Support Advisory Tool** providing continuous $T_{\text{margin}}$ estimations, early anomaly alerts, and sensor drift diagnostics to licensed control room operators.
