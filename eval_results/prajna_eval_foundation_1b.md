# 🛡️ PRAJNA PINN Model Benchmark & Physics Evaluation Report
**Model Scale:** `foundation_1b` (264,736,864 Parameters)
**Checkpoint:** `checkpoints/prajna_pinn_foundation_1b_best.pt` (Epoch 10)
**Timestamp:** `2026-09-20 00:07:05` | **Samples Tested:** `1,000`

## 🏆 Executive Summary
| Metric | Result | Physical Safety Standard / Target | Status |
| :--- | :--- | :--- | :--- |
| **Global $R^2$ Score** | **`0.9326`** | $> 0.9900$ | ⚠️ MARGINAL |
| **Global RMSE** | **`29.6888`** | $< 0.1000$ | ⚠️ MARGINAL |
| **Energy Balance Residual** | **`0.012755`** | $< 0.0100$ | ⚠️ MARGINAL |
| **1st-Law Energy Compliance** | **`98.72%`** | $> 99.00\%$ | ✅ PASS |
| **EOP Classification Accuracy** | **`100.00%`** | $> 95.00\%$ | ✅ PASS |

## 💥 Nuclear Transient Scenario Breakdown
| Scenario Description | RMSE | MAE | $R^2$ Score | Relative $L_2$ Error | Mean Physics Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Steady-State Normal** | `25.0289` | `14.5288` | `0.9438` | `0.1651` | `0.014109` |
| **Loss of Coolant Accident (LOCA)** | `30.4925` | `18.4338` | `0.9376` | `0.1869` | `0.012461` |
| **Reactivity-Initiated Accident (RIA / Rod Ejection)** | `42.1012` | `22.0590` | `0.8707` | `0.2499` | `0.012470` |
| **Steam Generator Tube Rupture (SGTR)** | `24.0253` | `12.9234` | `0.9540` | `0.1555` | `0.012382` |
| **Station Blackout (SBO) & Natural Circulation** | `22.3868` | `13.3775` | `0.9610` | `0.1477` | `0.012382` |

## 📡 Multi-Channel Sensor Breakdown
| Sensor Channel | RMSE | MAE | $R^2$ Score | Max Error |
| :--- | :--- | :--- | :--- | :--- |
| **Core Temp (°C)** | `54.4086` | `50.6663` | `-6.5283` | `144.2774` |
| **Coolant Flow (kg/s)** | `16.1170` | `13.0406` | `-0.0008` | `54.6133` |
| **Neutron Flux (x10^13)** | `1.0199` | `0.6779` | `-0.0000` | `5.6139` |
| **Radiation (mSv/h)** | `0.9810` | `0.7328` | `-0.0000` | `4.9598` |
| **Primary Pressure (bar)** | `14.7017` | `11.3682` | `-0.0000` | `69.7224` |
| **Core Power (MWth)** | `88.2298` | `78.5584` | `-3.7989` | `300.7520` |
| **Steam Quality (x)** | `0.0418` | `0.0321` | `-0.0000` | `0.1772` |
| **Control Rod Bank Height (%)** | `23.8373` | `19.2176` | `-0.0000` | `53.8309` |
| **Pressurizer Level (%)** | `13.3967` | `10.7252` | `-0.0000` | `33.9396` |
| **Feedwater Temp (°C)** | `13.2419` | `10.4528` | `-0.0001` | `46.4408` |
| **Steam Flow (kg/s)** | `20.7139` | `17.1856` | `-0.0001` | `57.4965` |
| **Core Inlet Temp (°C)** | `19.8306` | `15.3735` | `-0.0001` | `93.4408` |
| **Core Delta-T (°C)** | `0.0126` | `0.0101` | `-0.1057` | `0.0539` |
| **Cladding Temp (°C)** | `30.3154` | `22.5232` | `-0.0000` | `150.4035` |
| **Delayed Precursor Conc (C)** | `0.4798` | `0.3370` | `-0.0003` | `1.5450` |
| **Containment Pressure (kPa)** | `14.6803` | `9.3302` | `-0.0003` | `90.3103` |