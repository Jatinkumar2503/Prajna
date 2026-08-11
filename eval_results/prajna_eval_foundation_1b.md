# 🛡️ PRAJNA PINN Model Benchmark & Physics Evaluation Report
**Model Scale:** `foundation_1b` (264,736,864 Parameters)
**Checkpoint:** `checkpoints/prajna_pinn_foundation_1b_best.pt` (Epoch 10)
**Timestamp:** `2026-08-11 14:25:53` | **Samples Tested:** `2,000`

## 🏆 Executive Summary
| Metric | Result | Physical Safety Standard / Target | Status |
| :--- | :--- | :--- | :--- |
| **Global $R^2$ Score** | **`0.9318`** | $> 0.9900$ | ⚠️ MARGINAL |
| **Global RMSE** | **`29.9258`** | $< 0.1000$ | ⚠️ MARGINAL |
| **Energy Balance Residual** | **`0.012643`** | $< 0.0100$ | ⚠️ MARGINAL |
| **1st-Law Energy Compliance** | **`98.74%`** | $> 99.00\%$ | ✅ PASS |
| **EOP Classification Accuracy** | **`100.00%`** | $> 95.00\%$ | ✅ PASS |

## 💥 Nuclear Transient Scenario Breakdown
| Scenario Description | RMSE | MAE | $R^2$ Score | Relative $L_2$ Error | Mean Physics Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Steady-State Normal** | `25.2165` | `14.5847` | `0.9430` | `0.1662` | `0.332273` |
| **Loss of Coolant Accident (LOCA)** | `30.7004` | `18.5874` | `0.9369` | `0.1879` | `0.332273` |
| **Reactivity-Initiated Accident (RIA / Rod Ejection)** | `42.7417` | `22.5481` | `0.8696` | `0.2516` | `0.332273` |
| **Steam Generator Tube Rupture (SGTR)** | `24.1064` | `12.9499` | `0.9536` | `0.1561` | `0.332273` |
| **Station Blackout (SBO) & Natural Circulation** | `22.1680` | `13.2650` | `0.9617` | `0.1464` | `0.332273` |

## 📡 Multi-Channel Sensor Breakdown
| Sensor Channel | RMSE | MAE | $R^2$ Score | Max Error |
| :--- | :--- | :--- | :--- | :--- |
| **Core Temp (°C)** | `55.1787` | `51.1842` | `-6.1663` | `146.9343` |
| **Coolant Flow (kg/s)** | `15.8742` | `12.8659` | `-0.0000` | `54.6193` |
| **Neutron Flux (x10^13)** | `1.0331` | `0.6874` | `-0.0000` | `5.6637` |
| **Radiation (mSv/h)** | `0.9718` | `0.7253` | `-0.0000` | `5.0317` |
| **Primary Pressure (bar)** | `14.9309` | `11.4872` | `-0.0000` | `69.1519` |
| **Core Power (MWth)** | `88.3949` | `78.4880` | `-3.6957` | `302.4789` |
| **Steam Quality (x)** | `0.0419` | `0.0321` | `-0.0001` | `0.1731` |
| **Control Rod Bank Height (%)** | `23.8910` | `19.2216` | `-0.0000` | `53.8363` |
| **Pressurizer Level (%)** | `13.2072` | `10.5685` | `-0.0000` | `33.9495` |
| **Feedwater Temp (°C)** | `13.2548` | `10.5126` | `-0.0005` | `46.4853` |
| **Steam Flow (kg/s)** | `20.6736` | `17.1213` | `-0.0000` | `57.4951` |
| **Core Inlet Temp (°C)** | `20.6151` | `15.9887` | `-0.0003` | `96.0953` |
| **Core Delta-T (°C)** | `0.0126` | `0.0101` | `-0.1071` | `0.0535` |
| **Cladding Temp (°C)** | `31.5651` | `23.5011` | `-0.0005` | `152.3551` |
| **Delayed Precursor Conc (C)** | `0.4784` | `0.3367` | `-0.0002` | `1.5438` |
| **Containment Pressure (kPa)** | `14.9575` | `9.4611` | `-0.0000` | `90.2071` |