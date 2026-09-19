# PRAJNA: Physics-Informed Neural Network Foundation Model for Nuclear Safety

[![CI Pipeline](https://github.com/Jatinkumar2503/Prajna/actions/workflows/pinn_ci.yml/badge.svg)](https://github.com/Jatinkumar2503/Prajna/actions/workflows/pinn_ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Standards: AERB/SG/D-25 & IAEA SRS-29](https://img.shields.io/badge/Safety_Guides-AERB%2FSG%2FD--25%20%7C%20IAEA%20SRS--29-orange.svg)](docs/)

PRAJNA is a multi-scale Physics-Informed Neural Network (PINN) safety advisory system designed for real-time transient forecasting, diagnostic decision support, and anomaly detection in nuclear power plant (NPP) primary heat transport and reactor cores.

---

## 1. Verified Model Architectures & Parameter Counts

All parameter counts are empirically verified using live Python `sum(p.numel() for p in model.parameters())`:

| Model Scale | Parameter Count | Observable Channels | Precision & Format | Memory Footprint | Target Hardware / Cache | Checkpoint & Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`reflex_31k`** | **30,061** | 12 | FP32 / INT8 PTQ | 117.43 KB / 29.36 KB | CPU L2 Cache (<512 KB) | `prajna_reflex_12ch_noisy.pt` (Retrained & Verified) |
| **`pinn_60m`** | **60,848,728** | 12 | FP32 / ONNX | 243.39 MB / 60.85 MB | Edge Server / Mid GPU | `prajna_pinn_60m_12ch_noisy.pt` (Retrained & Verified) |
| **`pinn_265m`** | **264,729,688** | 12 | FP32 / FP16 AMP | 1.06 GB / 265 MB | NVIDIA RTX 3050 (6GB) | `prajna_pinn_foundation_1b_best.pt` (Trained Baseline) |
| **`config_2.27b`** | ~**2,270,646,272** | 12 | FP32 / INT8 | 9.08 GB / 2.27 GB | Multi-GPU Cluster (FSDP) | Architectural Scaling Specification (not trained) |
| **`config_3.08b`** | ~**3,089,139,712** | 12 | FP32 / INT8 | 12.36 GB / 3.09 GB | Distributed Supercomputer | Architectural Scaling Specification (not trained) |

### Cryptographic Checkpoint Provenance (SHA-256)
- `checkpoints/prajna_reflex_12ch_noisy.pt`: `d5473c4553d5e7e8c490ed16c253558d6c5811e4ac80f0e34824f15258f6d5b9`
- `checkpoints/prajna_pinn_60m_12ch_noisy.pt`: `6e7036cc36ef3965193ae4d4a878dd011be94f26d52b05e34a11cdd26389920d`
- Signed TPM 2.0 SHA-384 root manifest: `checkpoints/tpm_sha384_manifest.json`

---

## 2. Decoupled Physical Conservation Loss Core

The optimization objective $\mathcal{L}_{\text{total}}$ enforces strictly true physical conservation laws. Thermal-hydraulic safety limits (such as Departure from Nucleate Boiling Ratio and containment flammability) are **strictly decoupled from training backpropagation** and evaluated as diagnostic alarm trip flags, ensuring the network does not mask authentic severe accident trajectories.

$$\mathcal{L}_{\text{total}} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}} + \lambda_{\text{eop}} \mathcal{L}_{\text{eop}} + \lambda_{\text{xenon}} \mathcal{L}_{\text{xenon}}$$

### 2.1 First-Law Thermal Energy Balance Residual ($\mathcal{L}_{\text{energy}}$)
Enforces primary heat transport enthalpy conservation without artificial zero-clamping:
$$\mathcal{L}_{\text{energy}} = \frac{1}{N} \sum_{k=1}^N \left| \frac{Q_{\text{thermal}}^{(k)} - \dot{m}^{(k)} C_p (T_{\text{out}}^{(k)} - T_{\text{in}}^{(k)})}{50.0} \right|^2 + 0.05 \cdot \text{ReLU}\left(T_{\text{in}}^{(k)} - T_{\text{out}}^{(k)}\right)$$
*(Includes an explicit reverse thermal gradient penalty to prevent negative core temperature drops).*

### 2.2 Point Kinetics & Decay Heat Formulations
- **Delayed Neutron Kinetics:** Stiff six-group precursor kinetics ($dn/dt = \frac{\rho-\beta}{\Lambda}n + \sum \lambda_i C_i$).
- **Finite-Irradiation ANS-5.1 Decay Heat:**
  $$\frac{P_d(t, T)}{P_0} = \sum_{j=1}^{23} \alpha_j e^{-\lambda_j t} \left(1 - e^{-\lambda_j T}\right)$$
- **Bowring Critical Heat Flux Correlation:** Valid for $P \in [0.2, 19.0]\text{ MPa}$ and mass flux $G \in [136, 18600]\text{ kg}/(\text{m}^2\cdot\text{s})$ (using equivalent hydraulic diameter for rod bundle geometry) with safety criterion $\text{DNBR} \ge 1.30$.

---

## 3. Scenario-Specific Physics Loss Breakdown (Issue A9 Resolved)

Empirical evaluation from `scripts/reproduce_all_benchmarks.py` across 1,000 multi-physics transient sequences:

| Scenario Description | RMSE | MAE | $R^2$ Score | Enthalpy Loss | Xenon Loss | Total Conservation Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Steady-State Normal** | `42.389` | `19.595` | `0.7817` | `0.893402` | `0.000812` | `2.734445` |
| **Loss of Coolant Accident (LOCA)** | `44.674` | `24.962` | `0.8004` | `0.367084` | `0.004442` | `2.036980` |
| **Reactivity-Initiated Excursion (RIA / PHWR Zone Tilt)** | `59.974` | `31.790` | `0.5968` | `2.481215` | `0.003574` | `5.032229` |
| **Steam Generator Tube Rupture (SGTR) / Feeder Break** | `43.647` | `23.175` | `0.7828` | `0.698485` | `0.002097` | `2.349228` |
| **Station Blackout (SBO) & Natural Circulation** | `50.641` | `28.055` | `0.7409` | `0.413595` | `0.006180` | `3.458909` |

> **A9 Verification:** SGTR Enthalpy (`0.698485`) and SBO Enthalpy (`0.413595`) produce a **$0.284890$ difference** (differentiated by 40%), resolving the legacy zero-clamp bug that previously produced identical values to 6 digits.

---

## 4. Empirical Benchmark Suite & Evidence

All metrics generated by a single unified script:
```bash
python scripts/reproduce_all_benchmarks.py
```

### 4.1 True Ablation Study (PINN vs. Pure Data)
Matched architecture trained with physics loss ($\lambda_{\text{phys}} = 1.0$) vs. without physics loss ($\lambda_{\text{phys}} = 0.0$) across an instrument noise sweep:

| Noise Scale | Measurement $\sigma$ (RTD / Pressure) | PINN Accuracy | Pure Data Accuracy | Delta Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **0.0× (Clean)** | $\pm 0.00^\circ\text{C}$ / $\pm 0.00\text{ bar}$ | **100.00%** | 100.00% | 0.00% |
| **1.0× (Nominal Specs)** | $\pm 0.50^\circ\text{C}$ / $\pm 0.75\text{ bar}$ | **100.00%** | 100.00% | 0.00% |
| **2.0× (Degraded)** | $\pm 1.00^\circ\text{C}$ / $\pm 1.50\text{ bar}$ | **100.00%** | 100.00% | 0.00% |
| **3.5× (Severe Noise)** | $\pm 1.75^\circ\text{C}$ / $\pm 2.62\text{ bar}$ | **100.00%** | 100.00% | 0.00% |

### 4.2 Early Warning Lead Time vs. Strong Baselines
Measured detection advance before classical trip setpoint exceedance:

| Scenario Description | AI Lead Time (Mean ± 95% CI) | Min Lead Time | Rate-of-Change Baseline | CUSUM Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Loss of Coolant Accident (LOCA)** | $0.1\text{s} \pm 0.10\text{s}$ | $0.0\text{s}$ | $25.2\text{s}$ | $21.8\text{s}$ |
| **Reactivity Excursion (RIA)** | $3.8\text{s} \pm 0.95\text{s}$ | $0.0\text{s}$ | $20.6\text{s}$ | $23.6\text{s}$ |
| **SGTR / Feeder Break** | $17.5\text{s} \pm 1.11\text{s}$ | $4.0\text{s}$ | $39.7\text{s}$ | $36.1\text{s}$ |
| **Station Blackout (SBO)** | **$38.3\text{s} \pm 0.29\text{s}$** | **$30.0\text{s}$** | $39.2\text{s}$ | $39.6\text{s}$ |

### 4.3 False Alarm Rate (Rule of Three 95% Bound)
Evaluated across 25.0 operational hours ($400 \times 45\text{s}$ continuous windows) of steady-state operation under active Gaussian noise, first-order sensor lag, and slow calibration drift ($0.05\%/\text{h}$):
- **Observed False Alarms:** 0 events.
- **Empirical False Alarm Rate:** **0.0000 alarms / hour**.
- **95% Confidence Upper Bound:** **$< 0.1200\text{ alarms/hour}$** ($12.0\text{ per } 100\text{ hours}$ via Rule of Three: $\text{Upper Bound} = \frac{3.0}{N_{\text{hours}}}$).

### 4.4 Multi-Seed Confusion Matrix & LOCA Miss Rate
Evaluated across 3 independent seeds ($S \in \{42, 123, 999\}$):
- **Overall Multi-Seed Accuracy:** **$100.00\% \pm 0.00\%$**
- **LOCA $\to$ Normal Miss Rate:** **$0 / 200$ ($0.00\%$)**

```
      Steady-State   LOCA     RIA     SGTR     SBO
Steady       200        0       0        0       0
LOCA           0      200       0        0       0
RIA            0        0     200        0       0
SGTR           0        0       0      200       0
SBO            0        0       0        0     200
```

### 4.5 Feature Attribution Faithfulness (Deletion/Insertion Test)
Evaluating whether gradient/SHAP feature attributions faithfully reflect physical drivers:
- Top-Ranked Salient Channels: Radiation field ($0.0222$), Core Power ($0.0171$), Primary Pressure ($0.0155$), Core Exit Temp ($0.0127$).
- **Deletion Curve:**
  - 0 features deleted: $100.0\%$ accuracy.
  - Top-2 features deleted: $60.0\%$ accuracy ($40.0\%$ drop).
  - Top-5 features deleted: $46.0\%$ accuracy ($54.0\%$ drop).
  - *Result: Model degrades rapidly when top features are masked, confirming attribution faithfulness.*

### 4.6 Analytical Nuclear Physics Verification
- **Prompt Jump Ratio:** $\frac{n(0^+)}{n_0} = \frac{\beta}{\beta - \rho}$
  - Theoretical Asymptote ($\beta=0.0065, \rho=0.0010$): **$1.181818$**
  - Numerical 6-Group ODE at 80 ms: **$1.183786$** (Relative Error: **$0.1665\%$** $< 0.20\%$ tolerance).
- **Inhour Equation Stable Period ($50\text{ pcm}$ step):**
  - Effective precursor lifetime $\bar{\tau} = 13.04\text{ s}$; asymptotic stable period $T = 169.57\text{ s}$.

### 4.7 Latency Benchmark ($N=10,000$ Iterations)
Evaluated on **Intel(R) Core(TM) 5 210H CPU** (Single-threaded pinned affinity with instruction cache warm-up):
- **P50 Latency:** **$108.40\ \mu\text{s}$** ($0.1084\text{ ms}$)
- **P90 Latency:** **$144.72\ \mu\text{s}$** ($0.1447\text{ ms}$)
- **P99 Latency:** **$378.30\ \mu\text{s}$** ($0.3783\text{ ms}$)
- **ONNX Runtime (AVX2):** P50 = **$59.0\ \mu\text{s}$** (INT8) / **$70.0\ \mu\text{s}$** (FP32)

---

## 5. Industrial SCADA 88-Byte Frame Layout

The industrial SCADA bridge formats incoming sensor telemetry into an exact 88-byte binary frame:

| Byte Offset | Field Name | Data Type | Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `[00..03]` | Magic Sync Header | `uint32` | 4 B | `0x50524A4E` (`PRJN`) |
| `[04..07]` | Sequence Counter | `uint32` | 4 B | Monotonic Packet Sequence ID |
| `[08..15]` | Microsecond Timestamp | `uint64` | 8 B | POSIX epoch microsecond timestamp |
| `[16..79]` | Channel Telemetry | `16x float32` | 64 B | 12 observable + 4 auxiliary channels |
| `[80..83]` | OPC-UA Quality Bitfield | `uint32` | 4 B | 2 bits/channel (`00`=Good, `01`=Uncertain, `10`=Bad, `11`=Disconnected) |
| `[84..87]` | CRC-32 Checksum | `uint32` | 4 B | IEEE 802.3 Ethernet polynomial for link corruption detection |

---

## 6. Standards Compliance & Regulatory References

This project is engineered in accordance with:
- **AERB/SG/D-25:** *Design of Instrumentation and Control Systems for Nuclear Power Plants* (Atomic Energy Regulatory Board, India).
- **IAEA Safety Reports Series No. 29:** *Accident Analysis for Nuclear Power Plants with Pressurized Heavy Water Reactors (PHWRs)*.
- **NUREG-0700 (Rev. 3):** *Human-System Interface Design Review Guidelines* (mitigation of alarm flooding).
- **IEC 62645:** *Nuclear power plants - Instrumentation and control systems - Requirements for security programmes for computer-based systems*.
