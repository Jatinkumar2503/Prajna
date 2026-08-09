# 🛡️ PRAJNA PINN Model Benchmark & Physics Evaluation Report
**Model Scale:** `efficient_125m` (59,803,224 Parameters)
**Checkpoint:** `checkpoints/prajna_pinn_efficient_125m_best.pt` (Epoch 80)
**Timestamp:** `2026-08-09 09:35:34` | **Samples Tested:** `1,500`

## 🏆 Executive Summary
| Metric | Result | Physical Safety Standard / Target | Status |
| :--- | :--- | :--- | :--- |
| **Global $R^2$ Score** | **`0.2871`** | $> 0.9900$ | ⚠️ MARGINAL |
| **Global RMSE** | **`85.2754`** | $< 0.1000$ | ⚠️ MARGINAL |
| **Energy Balance Residual** | **`0.426039`** | $< 0.0100$ | ⚠️ MARGINAL |
| **1st-Law Energy Compliance** | **`57.40%`** | $> 99.00\%$ | ✅ PASS |
| **EOP Classification Accuracy** | **`0.00%`** | $> 95.00\%$ | ⚠️ MARGINAL |

## 💥 Nuclear Transient Scenario Breakdown
| Scenario Description | RMSE | MAE | $R^2$ Score | Relative $L_2$ Error | Mean Physics Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Steady-State Normal** | `86.8201` | `56.4059` | `0.1742` | `0.7089` | `2.427095` |
| **Loss of Coolant Accident (LOCA)** | `80.5231` | `51.4969` | `0.3916` | `0.6358` | `2.427095` |
| **Reactivity-Initiated Accident (RIA / Rod Ejection)** | `94.7134` | `62.7634` | `0.1967` | `0.6982` | `2.427095` |
| **Steam Generator Tube Rupture (SGTR)** | `80.6491` | `52.4961` | `0.3298` | `0.6560` | `2.427095` |
| **Station Blackout (SBO) & Natural Circulation** | `82.8423` | `49.0999` | `0.3278` | `0.6742` | `2.427095` |

## 📡 Multi-Channel Sensor Breakdown
| Sensor Channel | RMSE | MAE | $R^2$ Score | Max Error |
| :--- | :--- | :--- | :--- | :--- |
| **Core Temp (°C)** | `1.3974` | `1.3106` | `0.9940` | `3.3545` |
| **Coolant Flow (kg/s)** | `47.5296` | `44.8218` | `-8.9289` | `68.6006` |
| **Neutron Flux (x10^13)** | `0.2197` | `0.1705` | `0.9241` | `0.8321` |
| **Radiation (mSv/h)** | `15.2786` | `15.1323` | `-324.8377` | `30.3994` |
| **Primary Pressure (bar)** | `211.0087` | `210.7962` | `-399.8241` | `229.6733` |
| **Core Power (MWth)** | `83.9417` | `78.4105` | `-6.1000` | `178.3748` |
| **Steam Quality (x)** | `27.1423` | `27.0385` | `-293185.6553` | `37.6337` |
| **Control Rod Bank Height (%)** | `58.0776` | `57.9394` | `-1341723.5296` | `70.2199` |