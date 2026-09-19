# PRAJNA: Physics-Informed Neural Network Foundation Model for Nuclear Safety

[![CI Pipeline](https://github.com/Jatinkumar2503/Prajna/actions/workflows/pinn_ci.yml/badge.svg)](https://github.com/Jatinkumar2503/Prajna/actions/workflows/pinn_ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Standards: AERB/SG/D-25 & IAEA SRS-29](https://img.shields.io/badge/Safety_Guides-AERB%2FSG%2FD--25%20%7C%20IAEA%20SRS--29-orange.svg)](docs/)

PRAJNA is a multi-scale Physics-Informed Neural Network (PINN) safety advisory system designed for real-time transient forecasting, diagnostic decision support, and anomaly detection in nuclear power plant (NPP) primary heat transport and reactor cores.

---

## 1. System Architecture & Model Scales

The model family is structured across multiple tiers, from low-latency edge reflex engines to multi-scale foundation backbones:

| Model Scale | Parameter Count | Architecture | Precision & Format | Memory Footprint | Latency & Target Environment | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`reflex_31k`** | **30,577** (30.6k) | 4-Layer Residual SIMD-Aligned MLP | FP32 / INT8 PTQ | 119.44 KB (FP32) / 47.16 KB (INT8) | P50: 79 µs, P99: 149–171 µs (CPU L2 cache) | Trained & Distilled |
| **`pinn_60m`** | **59,796,944** (~59.8M) | Mamba-2 SSM ($d=768, L=16$) + FNO ($W=256$) | FP32 / ONNX | 239.2 MB | Sub-15 ms (Edge Server / Mid GPU) | Trained Baseline |
| **`pinn_265m`** | **264,749,520** (~264.7M) | Mamba-2 SSM ($d=1536, L=18$) + FNO ($W=384$) | FP32 / FP16 AMP | 1.06 GB (FP32) / 530 MB (FP16) | 24.3 ms (41.1 FPS on NVIDIA GPU) | Trained Teacher Baseline |
| **`config_2.27b`** | ~2,271,000,000 (~2.27B) | Mamba-2 SSM ($d=3072, L=40$) + FNO ($W=512$) | FP32 / INT8 | 9.08 GB (FP32) / 2.27 GB (INT8) | Multi-GPU Server (FSDP/ZeRO-3) | Defined Architectural Config |
| **`config_3.08b`** | ~3,083,000,000 (~3.08B) | Mamba-2 SSM ($d=3584, L=40$) + FNO ($W=512$) | FP32 / INT8 | 12.33 GB (FP32) / 3.08 GB (INT8) | Distributed Supercomputing Cluster | Defined Architectural Config |

> **Note on Model Capacities:** The primary trained foundation checkpoint is `pinn_265m` (264.7M parameters). The 2.27B and 3.08B configurations are exact parameter scaling specifications defined for distributed multi-GPU clusters, not yet trained.

---

## 2. Mathematical Formulation & Differentiable Loss Core

The core training loss $\mathcal{L}_{\text{total}}$ couples empirical telemetry reconstruction with physical conservation laws. Safety boundaries (such as Departure from Nucleate Boiling Ratio and containment flammability) are strictly decoupled from the loss and evaluated as diagnostic alarm triggers.

$$\mathcal{L}_{\text{total}} = \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{PKE}} \mathcal{L}_{\text{PKE}} + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}} + \lambda_{\text{decay}} \mathcal{L}_{\text{decay}}$$

### 2.1 Delayed Neutron Point Kinetics ($\mathcal{L}_{\text{PKE}}$)
Enforces stiff six-group precursor kinetics:
$$\frac{dn(t)}{dt} = \frac{\rho(t) - \beta}{\Lambda} n(t) + \sum_{i=1}^{6} \lambda_i C_i(t), \quad \frac{dC_i(t)}{dt} = \frac{\beta_i}{\Lambda} n(t) - \lambda_i C_i(t)$$

### 2.2 First-Law Energy Conservation ($\mathcal{L}_{\text{energy}}$)
Enforces primary heat transport balance:
$$\mathcal{L}_{\text{energy}} = \frac{1}{N} \sum_{k=1}^{N} \left| Q_{\text{thermal}}^{(k)} - \dot{m}^{(k)} C_p \left(T_{\text{out}}^{(k)} - T_{\text{in}}^{(k)}\right) \right|^2$$

### 2.3 Finite-Irradiation ANS-5.1 Decay Heat ($\mathcal{L}_{\text{decay}}$)
Evaluates post-trip core decay thermal generation as a function of operating history $T$ and cooling time $t$:
$$\frac{P_d(t, T)}{P_0} = \sum_{j=1}^{23} \alpha_j e^{-\lambda_j t} \left(1 - e^{-\lambda_j T}\right)$$

### 2.4 Decoupled Thermal and Flammability Evaluation Criteria
Accidents legitimately breach thermal and gas limits; penalizing them during training would incentivize the model to artificially suppress predicted damage. Therefore:
- **Departure from Nucleate Boiling Ratio (DNBR):** Evaluated via the Bowring correlation ($P \in [0.2, 16.0]\text{ MPa}, G \in [136, 18600]\text{ kg}/(\text{m}^2\cdot\text{s})$), flagging when $\text{DNBR} < 1.30$.
- **Radiolytic Hydrogen Generation:** Coolant dissolved $[\text{H}_2]$ is monitored in $\text{cc/kg}$ or $\text{mg/kg}$, while containment gas flammability is evaluated against the $4.0\text{ vol}\%$ flammability limit.

---

## 3. Observable Plant Telemetry vs. Virtual Sensors

To prevent data leakage from simulator internal states, input channels are strictly restricted to physically observable plant instruments:

| Channel ID | Telemetry Signal | Sensor Type | Units | Nominal Range |
| :--- | :--- | :--- | :--- | :--- |
| `P_core` | Core Neutron Thermal Power | SPND / Ex-Core Ion Chambers | % FP | 0 – 120 % |
| `T_fuel` | Fuel Pellet Average Temperature | Dynamic Plant Estimate | °C | 300 – 1200 °C |
| `T_coolant_in` | Primary Coolant Inlet Temperature | Narrow-Range RTD | °C | 250 – 295 °C |
| `T_coolant_out` | Primary Coolant Outlet Temperature | Narrow-Range RTD | °C | 290 – 330 °C |
| `P_primary` | Pressurizer / Header Pressure | Piezoelectric Pressure Cell | MPa | 8.0 – 16.5 MPa |
| `flow_primary` | Primary Coolant Mass Flow Rate | Electromagnetic / Venturi | kg/s | 0 – 28,000 kg/s |
| `P_steam` | Steam Generator Dome Pressure | Pressure Transmitter | MPa | 4.0 – 7.5 MPa |
| `flow_feedwater`| Secondary Feedwater Mass Flow | Orifice Plate Flowmeter | kg/s | 0 – 1,800 kg/s |
| `lvl_pressurizer`| Pressurizer Liquid Level | Differential Pressure Cell | % Span | 10 – 90 % |
| `lvl_steam_gen` | Steam Generator Water Level | Differential Pressure Cell | % Span | 20 – 80 % |
| `P_containment` | Reactor Building Pressure | Strain Gauge Transducer | kPa(g) | 0 – 350 kPa |
| `radiation_containment` | Reactor Building Gamma Activity | Area Gamma Monitor | Sv/h | $10^{-6} – 10^3$ Sv/h |

**Virtual Sensor Estimator Outputs (Internal Latent States):**
- Precursor concentrations ($\sum C_i$)
- Cladding peak temperature ($T_{\text{clad}}$)
- Channel exit steam quality ($X_{\text{exit}}$)
- Local critical heat flux ratio ($\text{DNBR}$)

---

## 4. Sensor Degradation & Noise Injection Suite

The simulation pipeline incorporates a rigorous degradation model (`prajna_core/noise.py`) mirroring real operational field conditions:
1. **Gaussian Measurement Noise:** Per-channel instrument uncertainty ($0.2\% – 1.0\%$ relative standard deviation).
2. **First-Order Sensor Lag:** First-order thermal well lag ($\tau = 0.5\text{ s}$ for pressure, $\tau = 3.5\text{ s}$ for RTDs).
3. **ADC Quantization:** 12-bit to 16-bit analog-to-digital converter discretization.
4. **Calibration Drift & Bias:** Slow linear drifts ($0.05\%/\text{h}$) to evaluate false-alarm resistance.
5. **Sensor Faults:** Stuck-at-last-value and channel dropouts.

---

## 5. Industrial SCADA Bridge & Cybersecurity

- **Industrial SCADA Bridge:** Conforms to an 88-byte binary frame carrying a microsecond timestamp (`uint64`), 16 telemetry channels (`float32`), an OPC-UA quality code bitfield (`uint32`), and a CRC-32 integrity checksum (`uint32`).
- **Cryptographic Provenance:** System integrity is verified via SHA-384 digest manifests against hardware TPM 2.0 / secure boot enclaves (`scripts/verify_tpm_integrity.py`).
- **Air-Gapped Zero-Actuation Architecture:** The system functions strictly as a read-only advisory layer isolated behind a unidirectional optical data diode, with zero electrical pathways to reactor control rod mechanisms or scram circuits.

---

## 6. Verification, Testing & Benchmarking

### 6.1 Unit Test Suite
Run the verified unit test suite:
```bash
python -m unittest discover -s tests
```
Tests evaluate:
- Delayed neutron precursor conservation.
- Analytical prompt-jump ratio verification:
  $$\frac{n(0^+)}{n_0} = \frac{\beta}{\beta - \rho}$$
  (Evaluated over 80 ms to ensure prompt relaxation with $\tau_p \approx 18.2\text{ ms}$; error $< 0.2\%$).
- First-law thermal-hydraulic energy conservation residuals.
- Bowring critical heat flux margin boundaries.
- OPC-UA 88-byte SCADA frame serialization and CRC-32 verification.

### 6.2 Ablation & Baseline Study
Evaluate physics loss contribution against non-physics baselines:
```bash
python scripts/ablation_study.py
```
Key metrics:
- **Lead Time Before Safety Threshold:** Measures detection advance relative to classical trip setpoints.
- **False Alarm Rate:** Verified on long normal operation runs.
- **Physical Plausibility / Energy Residual:** Energy conservation divergence rate.

### 6.3 Quantization & Parity Validation
Verify INT8 Post-Training Quantization parity against FP32 reference:
```bash
python scripts/validate_reflex_quantization.py
```
- FP32 Memory: 119.44 KB
- INT8 Memory: 47.16 KB
- Compression Ratio: 2.53× (60.5% memory reduction)
- EOP Classification Accuracy: 100.0% (FP32) vs. 99.0% (INT8)

---

## 7. Standards Compliance & Regulatory References

This project is developed in accordance with established nuclear safety codes:
- **AERB/SG/D-25:** Safety Guide on Computer-Based Systems in Nuclear Power Plants (Atomic Energy Regulatory Board, India).
- **IAEA Safety Reports Series No. 29:** Accident Analysis for Nuclear Power Plants with Pressurized Heavy Water Reactors.
- **NUREG-0700 (Rev. 3):** Human-System Interface Design Review Guidelines.
