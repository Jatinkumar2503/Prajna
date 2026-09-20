# PRAJNA Data Provenance Specification

This document establishes the formal provenance, physical fidelity boundaries, and allowable scientific claims for all datasets integrated within the PRAJNA experimental ecosystem.

---

## 1. Inventory & Dataset Provenance Matrix

| Parameter | PHWR-220 Target Simulator | NPPAD (PCTRAN) | PUR-1 (Purdue Research Reactor) | Kaggle NPP Anomaly Dataset | NRC Event Reports |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset Identifier** | `phwr_sim_v1` | `nppad_pctran_v1` | `pur1_purdue_v1` | `kaggle_npp_v1` | `nrc_ler_catalog_v1` |
| **Source / Institution** | PRAJNA In-House Physics Engine | Tsinghua University INET | Purdue University Nuclear Engineering | Kaggle Open Benchmark | US Nuclear Regulatory Commission |
| **Repository / Origin** | `prajna_core/simulator.py` | [thu-inet/NuclearPowerPlantAccidentData](https://github.com/thu-inet/NuclearPowerPlantAccidentData) | [zachdahm/Nuclear-data-TS-forecasting](https://github.com/zachdahm/Nuclear-data-TS-forecasting) | Kaggle Synthetic NPP Faults | NRC Licensee Event Reports (LER) |
| **License** | Project Proprietary / MIT | Academic Research License | Open Academic License | CC0 / Open Public License | Public Domain (US Govt) |
| **Reactor Type** | Pressurized Heavy Water Reactor (PHWR) | Pressurized Water Reactor (PWR) | Swimming Pool Research Reactor (MTR type) | Generic Pressurized Reactor (Abstract) | Commercial US LWR Fleet (PWR & BWR) |
| **Moderator / Coolant** | Heavy Water ($D_2O$) / $D_2O$ | Light Water ($H_2O$) / $H_2O$ | Light Water ($H_2O$) / $H_2O$ | Synthetic / Abstract | Light Water ($H_2O$) |
| **Nominal Thermal Power**| $756.0\text{ MWth}$ ($220\text{ MWe}$) | $2,775\text{ MWth}$ ($900\text{ MWe}$) | $1.0\text{ kWth}$ ($0.001\text{ MWth}$) | $3,000\text{ MWth}$ (Abstract) | $1,500 - 3,800\text{ MWth}$ |
| **Primary Pressure** | $85.0\text{ bar}$ ($8.5\text{ MPa}$) | $155.0\text{ bar}$ ($15.5\text{ MPa}$) | $1.0\text{ bar}$ (Atmospheric pool) | $150.0\text{ bar}$ | $70 - 155\text{ bar}$ |
| **Core Inlet / Exit Temp**| $249.0^\circ\text{C} \to 293.4^\circ\text{C}$ ($\Delta T = 44.4\text{ K}$) | $291.0^\circ\text{C} \to 326.0^\circ\text{C}$ ($\Delta T = 35.0\text{ K}$) | $20.0^\circ\text{C} \to 25.0^\circ\text{C}$ ($\Delta T \approx 5\text{ K}$) | $285^\circ\text{C} \to 315^\circ\text{C}$ | Plant-specific |
| **Nature of Data** | Numerical Physics Simulation (ODE) | Numerical Simulation (PCTRAN Code) | **Real Physical Reactor Telemetry** | Synthetic Benchmark | Semi-structured Regulatory Text |
| **Sampling Interval ($\Delta t$)**| $1.0\text{ s}$ | $10.0\text{ s}$ | $1.0\text{ s}$ | $1.0\text{ s}$ | Discrete Incident Timestamps |
| **Total Available Records** | Unlimited parametric generation | 6,783 windows across 18 scenarios | 200,000 continuous time-steps (18,595 windows) | 50,000 time-steps (3,331 windows) | Catalog of standard transients |

---

## 2. Allowed Scientific Roles & Explicit Negative Constraints

### Dataset A: PHWR-220 Physics Simulator (`phwr_sim`)
- **Primary Scientific Role:** Ground-truth in-domain model development, physics-constrained PINN training, dynamic residual closure, and safety threshold margin ($T_{\text{margin}}$) validation.
- **Allowed Claims:** "Demonstrates physics-consistent forecasting and early margin collapse detection on the reference 220 MWe PHWR surrogate under first-principles thermodynamic laws."
- **Disallowed Claims:** Cannot claim validation against actual commercial Indian PHWR plant telemetry (NPCIL operational data is proprietary/classified).

### Dataset B: NPPAD (`nppad_pctran`)
- **Primary Scientific Role:** Independent simulator cross-validation testbed. Used strictly to quantify the **cross-simulator domain gap** on transferable physical variables.
- **Allowed Claims:** "Quantifies whether the model captures general reactor dynamics across independent simulation engines (PCTRAN vs PRAJNA ODE) or merely memorizes ODE solver fingerprints."
- **Disallowed Claims:** Cannot be used as evidence that PRAJNA works identically on PWRs without domain adaptation, because $PWR \neq PHWR$ (different coolant, pressure, thermal inertia, and reactivity coefficients).

### Dataset C: PUR-1 Research Reactor (`pur1_purdue`)
- **Primary Scientific Role:** External real-data robustness benchmark. Used to evaluate real sensor noise resilience, temporal forecasting stability, and actual scram signature triage on real operational data.
- **Allowed Claims:** "Demonstrates that PRAJNA's temporal feature extractors remain stable and responsive when exposed to real physical reactor measurements, real ionization chamber noise, and physical scram transients."
- **Disallowed Claims:** **PUR-1 CANNOT BE CLAIMED AS A PHWR VALIDATION DATASET.** PUR-1 is an open-pool 1 kW research reactor; it does not operate at $85\text{ bar}$ or $293^\circ\text{C}$, and has no pressurized primary loop.

### Dataset D: Kaggle NPP (`kaggle_npp`)
- **Primary Scientific Role:** Secondary anomaly detection benchmark for comparative analysis against traditional ML baselines (Isolation Forest, One-Class SVM, standard GRU/LSTM autoencoders).
- **Allowed Claims:** "Provides a standardized baseline comparison demonstrating the superiority of physics-constrained threshold margin forecasting over unconstrained black-box anomaly scoring."
- **Disallowed Claims:** Cannot be cited as real or high-fidelity reactor engineering proof.

### Dataset E: NRC Event Reports (`nrc_events`)
- **Primary Scientific Role:** Qualitative failure taxonomy and scenario catalog definition. Used strictly to ensure that simulated accident transients correspond to real-world historical nuclear events.
- **Allowed Claims:** "Simulator transient catalog is informed by historical failure modes documented in NRC Licensee Event Reports."
- **Disallowed Claims:** Text reports are not fed directly into the continuous time-series neural network.

---

## 3. Data Integrity & Storage Topology

All raw files are preserved in immutable storage. Transformed data must retain traceability metadata (source file, row range, calibration offsets, and transformation timestamp).

```
data/
├── raw/
│   ├── phwr_sim/        # Parametric synthetic runs generated by PRAJNA engine
│   ├── nppad/           # Immutable PCTRAN accident CSV runs (18 scenarios)
│   ├── pur1/            # Immutable Purdue PUR-1 telemetry files
│   ├── kaggle/          # Immutable Kaggle fault injection CSV
│   └── nrc_events/      # Parsed accident taxonomy & scenario definitions
├── harmonized/
│   ├── phwr/            # Harmonized into standard SI units and 12-channel format
│   ├── nppad/           # Mapped transferable subset with explicit unit conversion
│   └── pur1/            # Mapped real channels with preserved raw noise statistics
└── processed/
    ├── train/           # Strictly partitioned training splits
    ├── validation/      # In-domain validation splits
    └── test/            # Held-out in-domain and cross-domain test splits
```
