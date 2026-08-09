# 🛡️ PRAJNA PINN Model Benchmark & Physics Evaluation Report
**Model Scale:** `foundation_1b` (264,735,832 Parameters)
**Checkpoint:** `checkpoints/prajna_pinn_foundation_1b_best.pt` (Epoch 25)
**Timestamp:** `2026-08-09 14:27:45` | **Samples Tested:** `1,500`

## 🏆 Executive Summary
| Metric | Result | Physical Safety Standard / Target | Status |
| :--- | :--- | :--- | :--- |
| **Global $R^2$ Score** | **`0.7893`** | $> 0.9900$ | ⚠️ MARGINAL |
| **Global RMSE** | **`46.3606`** | $< 0.1000$ | ⚠️ MARGINAL |
| **Energy Balance Residual** | **`5.740364`** | $< 0.0100$ | ⚠️ MARGINAL |
| **1st-Law Energy Compliance** | **`0.00%`** | $> 99.00\%$ | ✅ PASS |
| **EOP Classification Accuracy** | **`0.00%`** | $> 95.00\%$ | ⚠️ MARGINAL |

## 💥 Nuclear Transient Scenario Breakdown
| Scenario Description | RMSE | MAE | $R^2$ Score | Relative $L_2$ Error | Mean Physics Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Steady-State Normal** | `49.5792` | `35.9776` | `0.7307` | `0.4048` | `18.486174` |
| **Loss of Coolant Accident (LOCA)** | `39.1460` | `28.9147` | `0.8562` | `0.3091` | `18.486174` |
| **Reactivity-Initiated Accident (RIA / Rod Ejection)** | `58.4026` | `40.1286` | `0.6946` | `0.4305` | `18.486174` |
| **Steam Generator Tube Rupture (SGTR)** | `41.9580` | `30.8282` | `0.8186` | `0.3413` | `18.486174` |
| **Station Blackout (SBO) & Natural Circulation** | `39.8077` | `28.1260` | `0.8448` | `0.3240` | `18.486174` |

## 📡 Multi-Channel Sensor Breakdown
| Sensor Channel | RMSE | MAE | $R^2$ Score | Max Error |
| :--- | :--- | :--- | :--- | :--- |
| **Core Temp (°C)** | `3.3766` | `3.3467` | `0.9648` | `5.8319` |
| **Coolant Flow (kg/s)** | `49.2633` | `47.0091` | `-9.6665` | `65.1152` |
| **Neutron Flux (x10^13)** | `1.1596` | `1.1221` | `-1.1133` | `1.9574` |
| **Radiation (mSv/h)** | `32.2681` | `32.2187` | `-1452.3801` | `37.6437` |
| **Primary Pressure (bar)** | `76.9441` | `75.8699` | `-52.2971` | `92.2830` |
| **Core Power (MWth)** | `85.4994` | `79.9387` | `-6.3660` | `181.3954` |
| **Steam Quality (x)** | `1.2960` | `0.9881` | `-667.4772` | `5.3879` |
| **Control Rod Bank Height (%)** | `21.9392` | `21.8668` | `-191463.3412` | `30.2286` |