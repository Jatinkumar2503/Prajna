# PRAJNA PINN Foundation Model — Technical Specification & Empirical Report

## 1. Executive Summary

The **PRAJNA Multi-Scale Physics-Informed Neural Network (PINN)** is a deep learning safety advisory architecture designed for real-time nuclear reactor telemetry forecasting, physical anomaly detection, Departure from Nucleate Boiling Ratio (DNBR) safety margin tracking, and automated Emergency Operating Procedure (EOP) diagnostic triage.

Safety boundary conditions (such as DNBR $< 1.30$ and radiolytic $[\text{H}_2] > 4.0\text{ vol}\%$) are strictly decoupled from the backpropagation loss and monitored as downstream diagnostic safety alarms, ensuring that severe accident trajectories are never suppressed or masked during training.

---

## 2. Multi-Scale Architecture Configuration Matrix

Parameter counts are empirically verified using live Python `sum(p.numel() for p in model.parameters())`:

| Scale Identifier | Trainable Parameters | Hidden Dim ($d_{\text{model}}$) | Temporal Mamba Layers | FNO Modes / Width | Memory (FP32 / INT8) | Target Hardware | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`reflex_31k`** | **30,061** | 96 | 4 (Residual SIMD) | — | 117.4 KB / 29.4 KB | CPU L2 Cache (<512 KB) | Trained & Verified |
| **`test_4m`** | 4,258,456 | 256 | 4 | 16 / 128 | 17.0 MB / 4.25 MB | CI/CD Unit Testing | Verified |
| **`pinn_60m`** | **60,848,728** | 768 | 16 | 16 / 256 | 243.4 MB / 60.85 MB | Edge Server / Mid GPU | Retrained & Verified |
| **`pinn_265m`** | **264,729,688** | 1536 | 18 | 16 / 384 | 1.06 GB / 265 MB | NVIDIA RTX 3050 (6GB) | Trained Foundation Baseline |
| **`config_2.27b`** | ~**2,270,646,272** | 3072 | 40 | 16 / 512 | 9.08 GB / 2.27 GB | Multi-GPU Server (FSDP / ZeRO-3) | Scaling Configuration Defined |
| **`config_3.08b`** | ~**3,089,139,712** | 3584 | 40 | 16 / 512 | 12.36 GB / 3.09 GB | Distributed Cluster | Scaling Configuration Defined |

---

## 3. Mathematical Loss Formulation

The optimization objective enforces continuous physical conservation laws directly into network weights:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}} + \lambda_{\text{eop}} \mathcal{L}_{\text{eop}} + \lambda_{\text{xenon}} \mathcal{L}_{\text{xenon}}$$

### 3.1 Primary Heat Transport Energy Balance Residual
Enforces signed thermal-hydraulic enthalpy balance without artificial zero-clamping:
$$\mathcal{L}_{\text{energy}} = \frac{1}{N} \sum_{k=1}^N \left| \frac{\hat{Q}_{\text{thermal}}^{(k)} - \dot{m}^{(k)} C_p (\hat{T}_{\text{out}}^{(k)} - T_{\text{in}}^{(k)})}{50.0} \right|^2 + 0.05 \cdot \text{ReLU}\left(T_{\text{in}}^{(k)} - \hat{T}_{\text{out}}^{(k)}\right)$$

### 3.2 Decoupled DNBR & Safety Limit Alarms
Critical Heat Flux (Bowring correlation, valid $P \in [0.2, 19.0]\text{ MPa}, G \in [136, 18600]\text{ kg}/(\text{m}^2\cdot\text{s})$) and hydrogen flammability limits ($<4.0\text{ vol}\%$) are evaluated strictly as downstream trip flags:
$$\text{DNBR} = \frac{q''_{\text{critical}}}{q''_{\text{local}}}, \quad \text{Alarm if } \text{DNBR} < 1.30$$

---

## 4. Empirical Training & Convergence Milestones

### 4.1 Reflex 31k Scale (CPU / Edge Deployment)
* **Parameters:** **30,061** (12 observable plant channels)
* **Epochs:** 15
* **Validation Accuracy:** **100.00%**
* **Inference Latency (Host CPU):** P50 = 108.4 µs, P99 = 378.3 µs (ONNX Runtime INT8: P50 = 59.0 µs)
* **Checkpoint:** `checkpoints/prajna_reflex_12ch_noisy.pt` (SHA-256: `d5473c4553d5e7e8c490ed16c253558d6c5811e4ac80f0e34824f15258f6d5b9`)

### 4.2 PINN 60M Scale (NVIDIA RTX 3050 Benchmark)
* **Parameters:** **60,848,728** (12 observable plant channels)
* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6 + AMP FP16)
* **Epochs:** 8 (124.4s wall-clock training time)
* **Validation Loss:** `3.2698` (Energy residual converged to `0.8207`)
* **Validation Accuracy:** **100.00%**
* **Checkpoint:** `checkpoints/prajna_pinn_60m_12ch_noisy.pt` (SHA-256: `6e7036cc36ef3965193ae4d4a878dd011be94f26d52b05e34a11cdd26389920d`)

---

## 5. Inference Latency & High-Throughput Benchmarks

Evaluated on host **Intel(R) Core(TM) 5 210H** CPU (single thread pinned, $N=10,000$ iterations):

| Model Scale | Inference Engine | P50 (Median) | P90 | P99 | Throughput | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`reflex_31k`** | PyTorch Eager (CPU) | 108.4 µs | 144.7 µs | 378.3 µs | 7,569.8 FPS | 117.4 KB (L2 Cache) |
| **`reflex_31k`** | ONNX Runtime FP32 | 70.0 µs | 125.0 µs | 256.0 µs | 14,214.6 FPS | 119.4 KB |
| **`reflex_31k`** | ONNX Runtime INT8 | 59.0 µs | 107.0 µs | 266.0 µs | 16,820.9 FPS | 29.4 KB (2.53× weight comp.) |
| **`pinn_265m`** | PyTorch Native (RTX 3050 GPU)| 24.3 ms | 24.5 ms | 24.8 ms | 41.1 FPS | 1.06 GB (VRAM) |

---

## 6. Real-Time Telemetry & 3D Digital Twin Integration

1. `src/telemetry/ring_buffer.js`: Inter-thread lock-free `SharedArrayBuffer` ingestion across Web Workers with non-SAB fallback.
2. `src/inference/pinn_runtime.js`: Microsecond neural inference computing $t+10\text{s}$ trajectory forecasts, TTL countdowns, and formal IAEA/AERB EOP classifications.
3. `simulation3d.js`: Three.js WebGL digital twin rendering core thermal glow and flow fields at up to display refresh rate (60–120 Hz).
