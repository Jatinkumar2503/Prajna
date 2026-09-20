# PRAJNA: System Architecture Specification

## 1. Architectural Pipeline Overview

PRAJNA decouples neural forecasting from safety-critical decision logic. Rather than predicting safety actions directly through an opaque neural classification head, the system operates as a deterministic pipeline:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ SENSOR TELEMETRY HISTORY:  X_{t-W:t} in R^{W x 12}                                │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ TEMPORAL EMBEDDING BACKBONE                                                       │
│ - PrajnaFastReflex: Residual SIMD-friendly 1D backbone (30,577 params, ~120 KB)   │
│ - CPU L2-Cache resident (<512 KB) for sub-100 microsecond deterministic execution │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ PHYSICS-CONSTRAINED FUTURE TRAJECTORY FORECASTER                                  │
│ - Output 1: Multi-Step State Forecast \hat{X}_{t+1:t+H} in R^{H x 12}             │
│ - Regularized via Dynamic Physics Residual Loss L_dyn:                            │
│     L_dyn = | Q_core(t) - \dot{m} C_p (T_out - T_in) |                             │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ DETERMINISTIC SAFETY-BOUNDARY INTERFACE                                           │
│ - Output 2: Threshold-Crossing Time T_margin and Uncertainty Bounds [CI_L, CI_U]  │
│ - Evaluates distance to AERB/IAEA licensed envelope S_unsafe                      │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ DETERMINISTIC SAFETY GATE                                                         │
│ - Output 3: Three-State Safety Interlock (NORMAL, WARNING, CRITICAL_TRIP)         │
│ - Zero black-box neural actuation; strictly deterministic threshold gating        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Sensor Representation & Normalization Layer
- Ingests 12 observable plant channels at $1.0\text{ Hz}$.
- Normalization statistics ($\mu_j, \sigma_j$) are computed **strictly on in-domain training trajectories** and frozen at inference time.
- Missing channels are imputed via mean batch imputation combined with channel dropout training to build sensor resilience.

### 2.2 Microsecond Safety Reflex Engine (`PrajnaFastReflex`)
- Architecture: 3-layer residual MLP with LayerNorm and SiLU non-linearities.
- Hidden Dimension: 96 units.
- Footprint: Fits entirely within Intel/AMD CPU L2 cache, guaranteeing zero cache thrashing and deterministic latency ($< 100\ \mu\text{s}$).

### 2.3 Physics Regularizer
Enforces first-principles thermodynamic energy balance during training:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{forecast}} + \lambda_{\text{phys}} \mathcal{L}_{\text{dyn}}$$
where $\mathcal{L}_{\text{dyn}}$ penalizes violations of heavy water enthalpy rise relative to core thermal power.
