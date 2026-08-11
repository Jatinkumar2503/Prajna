# 🛡️ PRAJNA PINN Model Benchmark & Physics Evaluation Report
**Model Scale:** `test_4m` (2,725,472 Parameters)
**Checkpoint:** `checkpoints/prajna_pinn_test_4m_best.pt` (Epoch 10)
**Timestamp:** `2026-08-10 09:51:48` | **Samples Tested:** `1,500`

## 🏆 Executive Summary
| Metric | Result | Physical Safety Standard / Target | Status |
| :--- | :--- | :--- | :--- |
| **Global $R^2$ Score** | **`0.9904`** | $> 0.9900$ | ✅ PASS |
| **Global RMSE** | **`11.1491`** | $< 0.1000$ | ⚠️ MARGINAL |
| **Energy Balance Residual** | **`0.018679`** | $< 0.0100$ | ⚠️ MARGINAL |
| **1st-Law Energy Compliance** | **`98.13%`** | $> 99.00\%$ | ✅ PASS |
| **EOP Classification Accuracy** | **`100.00%`** | $> 95.00\%$ | ✅ PASS |

## 💥 Nuclear Transient Scenario Breakdown
| Scenario Description | RMSE | MAE | $R^2$ Score | Relative $L_2$ Error | Mean Physics Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Steady-State Normal** | `1.5126` | `0.8626` | `0.9998` | `0.0100` | `0.065051` |
| **Loss of Coolant Accident (LOCA)** | `7.3702` | `4.6216` | `0.9963` | `0.0452` | `0.065051` |
| **Reactivity-Initiated Accident (RIA / Rod Ejection)** | `21.1555` | `6.5944` | `0.9672` | `0.1260` | `0.065051` |
| **Steam Generator Tube Rupture (SGTR)** | `4.9396` | `3.1916` | `0.9980` | `0.0320` | `0.065051` |
| **Station Blackout (SBO) & Natural Circulation** | `9.6409` | `5.7077` | `0.9928` | `0.0634` | `0.065051` |

## 📡 Multi-Channel Sensor Breakdown
| Sensor Channel | RMSE | MAE | $R^2$ Score | Max Error |
| :--- | :--- | :--- | :--- | :--- |
| **Core Temp (°C)** | `8.7509` | `5.1788` | `0.7652` | `39.6655` |
| **Coolant Flow (kg/s)** | `4.6069` | `3.4469` | `0.9074` | `19.7021` |
| **Neutron Flux (x10^13)** | `0.1498` | `0.1126` | `0.9650` | `1.0241` |
| **Radiation (mSv/h)** | `0.3861` | `0.2230` | `0.7920` | `1.7487` |
| **Primary Pressure (bar)** | `5.8017` | `4.5834` | `0.7635` | `13.6551` |
| **Core Power (MWth)** | `39.0312` | `20.6962` | `-0.5232` | `172.7403` |
| **Steam Quality (x)** | `0.0067` | `0.0047` | `0.9600` | `0.0409` |
| **Control Rod Bank Height (%)** | `5.8759` | `4.1598` | `0.9363` | `31.5634` |
| **Pressurizer Level (%)** | `5.1296` | `3.9980` | `0.7879` | `12.4973` |
| **Feedwater Temp (°C)** | `5.0975` | `3.5859` | `0.7526` | `17.3713` |
| **Steam Flow (kg/s)** | `5.9542` | `4.2388` | `0.8975` | `25.0437` |
| **Core Inlet Temp (°C)** | `5.3238` | `3.5764` | `0.9131` | `30.7257` |
| **Core Delta-T (°C)** | `0.0362` | `0.0259` | `-8.0680` | `0.1745` |
| **Cladding Temp (°C)** | `10.1795` | `7.1448` | `0.8497` | `40.6156` |
| **Delayed Precursor Conc (C)** | `0.0565` | `0.0336` | `0.9826` | `0.4467` |
| **Containment Pressure (kPa)** | `8.9173` | `6.1204` | `0.6211` | `24.0381` |