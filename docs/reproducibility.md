# PRAJNA: End-to-End Reproducibility Guide

This guide provides the exact command sequence required to reproduce all 10 scientific priorities and experiments from a clean state.

---

## 1. Prerequisites & Environment Verification

```bash
# Verify Python version (3.10+) and PyTorch
python --version
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

Required dependencies: `torch`, `numpy`, `pandas`, `scipy`, `pyyaml`, `scikit-learn`.

---

## 2. Full Replication Sequence

### Step 1: Formal Dataset Audit
Audits all 97 NPPAD columns and 9 PUR-1 columns against the validated variable schema:
```bash
python scripts/audit_datasets.py
```
*Output Artifact:* `artifacts/formal_data_audit.json`

### Step 2: Zero-Leakage Trajectory Harmonization
Generates trajectory-level isolated training, validation, and held-out test splits:
```bash
python scripts/harmonize_data.py
```
*Output Artifacts:* `data/processed/train/`, `data/processed/validation/`, `data/processed/test/`, `data/processed/dataset_manifest.json`

### Step 3: Execute the Master 10-Priority Experimental Suite
Runs Experiments 01 to 07, including trajectory baselines, deep cross-simulator transfer diagnosis, zero-leakage multi-domain evaluation, physics constraint ablation, single-channel attribution, and horizon degradation testing:
```bash
python scripts/reproduce_all_10_priorities.py
```
*Output Artifacts:*
- `experiments/exp01_in_domain/results.json`
- `experiments/exp02_cross_simulator/results.json`
- `experiments/exp03_real_data/results.json`
- `experiments/exp04_multidomain/results.json`
- `experiments/exp05_noise_robustness/results.json`
- `experiments/exp06_physics_ablation/results.json`
- `experiments/exp07_tmargin/results.json`
- `evaluation/reports/master_experiment_summary.json`

### Step 4: Standardized Deep Temporal Baselines
Executes the apples-to-apples comparison against GRU, LSTM, Temporal Transformer, and static anomaly detectors on the standardized 15-second boundary crossing task:
```bash
python scripts/compare_baselines.py
```
*Output Artifact:* `evaluation/reports/baseline_comparison.json`
