# PRAJNA Formal Variable Mapping & Transfer Audit

## Executive Summary
This document provides an exhaustive, physically justified audit of every variable in external datasets (NPPAD 97 columns, PUR-1 9 columns) against PRAJNA's 12 standard physical channels. 

Variables are audited and classified into four strict transfer validity tiers:
1. **HIGH (Valid Transfer):** Direct physical analog with identical thermodynamic or neutronic role (e.g., thermal power, coolant exit temperature, primary pressure).
2. **MEDIUM (Transfer with Normalization):** Physical analog requiring reactor-specific scaling or dimension normalization (e.g., total core flow, neutron flux).
3. **LOW (Weak Analogue / Contextual Only):** Divergent physical systems where quantitative transfer is questionable (e.g., radiation monitors calibrated to different geometries).
4. **DO NOT TRANSFER (Physically Incompatible):** System-specific variables that have no valid physical counterpart in a PHWR (e.g., PWR pressurizer level vs PHWR bleed condenser/surge tank, chemical boron PPM vs mechanical adjusters, open-pool water level vs closed 8.5 MPa primary circuit).

---

## 1. PRAJNA Target Physical Channels (Reference 220 MWe PHWR)

| Channel | Variable Name | Physical Symbol | Engineering Unit | Nominal Steady-State | Physical Role |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | Core Exit Temperature | $T_{\text{out}}$ | $^\circ\text{C}$ | $293.4$ | Bulk coolant temperature leaving fuel channels |
| **1** | Primary Coolant Flow | $\dot{m}_{\text{cool}}$ | $\text{kg/s}$ | $3,500.0$ | Total mass flow through 306 fuel channels |
| **2** | Thermal Neutron Flux | $\phi_{\text{th}}$ | $\times 10^{13}\text{ n/cm}^2\cdot\text{s}$ | $2.25$ | Core average thermal neutron flux |
| **3** | Radiation Field | $R_{\text{field}}$ | $\text{mSv/h}$ | $0.40$ | Containment area radiation monitor |
| **4** | Primary System Pressure | $P_{\text{prim}}$ | $\text{bar}$ | $85.0$ | Primary heat transport (PHT) system pressure |
| **5** | Core Thermal Power | $Q_{\text{core}}$ | $\text{MWth}$ | $756.0$ | Total thermal power transferred to coolant |
| **6** | Control Rod Position | $Z_{\text{rod}}$ | $\%$ withdrawn | $65.0$ | Adjuster / regulating rod bank position |
| **7** | Pressurizer / Level | $L_{\text{pzer}}$ | $\%$ | $50.0$ | PHT inventory / surge tank level |
| **8** | Steam Generator Temp | $T_{\text{sec}}$ | $^\circ\text{C}$ | $245.0$ | Secondary saturation temperature ($36.5\text{ bar}$) |
| **9** | Main Steam Flow | $\dot{m}_{\text{steam}}$ | $\text{kg/s}$ | $364.0$ | Total steam delivery to turbine |
| **10** | Core Inlet Temperature| $T_{\text{in}}$ | $^\circ\text{C}$ | $249.0$ | Coolant temperature entering fuel channels |
| **11** | Containment Pressure | $P_{\text{cont}}$ | $\text{kPa}$ | $101.325$ | Reactor building atmospheric containment pressure |

---

## 2. NPPAD (PCTRAN PWR Simulator) — Complete 97-Variable Audit

Sampling Rate: $\Delta t = 10.0\text{ s}$.

| Variable | Raw Unit | Physical Meaning | Reactor System | PRAJNA Target | Transfer Validity | Physical Justification / Notes |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| `TIME` | s | Transient elapsed time | Simulation Clock | `time` | **HIGH** | Common temporal axis. |
| `P` | bar/psia | Reactor Coolant System (RCS) Pressure | Primary Loop | **Ch 4** | **HIGH** | Direct physical equivalent of primary loop pressure. Scale factor converts psia to bar if needed. |
| `TAVG` | °C/°F | RCS Average Coolant Temp | Primary Loop | None | **MEDIUM** | Redundant linear combination of $(T_{\text{hot}} + T_{\text{cold}})/2$. Excluded to prevent collinearity. |
| `THA` | °C/°F | Hot Leg A Coolant Temperature | Core Exit | **Ch 0** | **HIGH** | Direct physical equivalent of core exit coolant temperature. |
| `THB` | °C/°F | Hot Leg B Coolant Temperature | Core Exit | None | **HIGH** | Identical to THA under symmetric loop operation. |
| `TCA` | °C/°F | Cold Leg A Coolant Temperature | Core Inlet | **Ch 10** | **HIGH** | Direct physical equivalent of core inlet coolant temperature. |
| `TCB` | °C/°F | Cold Leg B Coolant Temperature | Core Inlet | None | **HIGH** | Identical to TCA under symmetric loop operation. |
| `WRCA` | kg/s | RCS Loop A Coolant Mass Flow | Primary Loop | **Ch 1** | **MEDIUM** | Physical analogue; requires scaling from 2-loop PWR (total ~14,000 kg/s) to PHWR (3,500 kg/s). |
| `WRCB` | kg/s | RCS Loop B Coolant Mass Flow | Primary Loop | None | **MEDIUM** | Symmetric loop flow. |
| `PSGA` | bar | Steam Generator A Pressure | Secondary Loop | None | **MEDIUM** | Secondary pressure analog ($36.5\text{ bar}$ in PHWR vs $\sim 65\text{ bar}$ in PWR). |
| `PSGB` | bar | Steam Generator B Pressure | Secondary Loop | None | **MEDIUM** | Symmetric secondary pressure. |
| `WFWA` | kg/s | Feedwater Flow Loop A | Secondary Feed | None | **MEDIUM** | Secondary balance variable. |
| `WFWB` | kg/s | Feedwater Flow Loop B | Secondary Feed | None | **MEDIUM** | Secondary balance variable. |
| `WSTA` | kg/s | Steam Flow Loop A | Secondary Steam | **Ch 9** | **MEDIUM** | Physical analogue of turbine steam mass flow; scaled by total thermal rating. |
| `WSTB` | kg/s | Steam Flow Loop B | Secondary Steam | None | **MEDIUM** | Symmetric steam flow. |
| `VOL` | m³ | Pressurizer Liquid Volume | Primary Inventory | None | **DO NOT TRANSFER** | PWR pressurizer geometry is physically distinct from PHWR bleed condenser/surge tank. |
| `LVPZ` | % | Pressurizer Level | Primary Inventory | **Ch 7** | **LOW** | Contextual analogue only; different inventory management physics. |
| `VOID` | % | Core Bulk Steam Void Fraction | Core Hydraulics | None | **MEDIUM** | Relevant for boiling/void reactivity, but PHWR void coefficient is positive whereas PWR is negative. |
| `WLR` | kg/s | Lower Plenum Mass Flow | Reactor Vessel | None | **DO NOT TRANSFER** | Vessel-specific internal geometry not present in horizontal pressure tube PHWR. |
| `WUP` | kg/s | Upper Plenum Mass Flow | Reactor Vessel | None | **DO NOT TRANSFER** | Vessel-specific internal geometry not present in horizontal pressure tube PHWR. |
| `HUP` | kJ/kg | Upper Plenum Specific Enthalpy | Reactor Vessel | None | **DO NOT TRANSFER** | Specific to light water enthalpy tables. |
| `HLW` | kJ/kg | Lower Plenum Specific Enthalpy | Reactor Vessel | None | **DO NOT TRANSFER** | Specific to light water enthalpy tables. |
| `WHPI` | kg/s | High Pressure Safety Injection Flow| Safety ECCS | None | **MEDIUM** | Emergency injection flow indicator; scenario-specific. |
| `WECS` | kg/s | Emergency Core Cooling Flow | Safety ECCS | None | **MEDIUM** | ECCS actuation indicator. |
| `QMWT` | MWth | Core Total Thermal Power | Core Fission | **Ch 5** | **HIGH** | Direct physical equivalent of thermal power. Normalized by rated plant power ($Q/Q_0$). |
| `LSGA` | % | Steam Generator A Liquid Level | Secondary Steam | None | **MEDIUM** | Secondary inventory indicator. |
| `LSGB` | % | Steam Generator B Liquid Level | Secondary Steam | None | **MEDIUM** | Secondary inventory indicator. |
| `QMGA` | MWth | Heat Transferred to SG A | Secondary Heat | None | **MEDIUM** | Heat balance validation variable. |
| `QMGB` | MWth | Heat Transferred to SG B | Secondary Heat | None | **MEDIUM** | Heat balance validation variable. |
| `NSGA` | - | SG A Recirculation Ratio | Secondary SG | None | **DO NOT TRANSFER** | Internal SG separator parameter. |
| `NSGB` | - | SG B Recirculation Ratio | Secondary SG | None | **DO NOT TRANSFER** | Internal SG separator parameter. |
| `TBLD` | °C | SG Blowdown Temperature | Secondary Water | None | **LOW** | Water chemistry blowdown. |
| `WTRA` | kg/s | Turbine Relief Valve Flow Loop A | Steam System | None | **LOW** | Overpressure relief valve flow. |
| `WTRB` | kg/s | Turbine Relief Valve Flow Loop B | Steam System | None | **LOW** | Overpressure relief valve flow. |
| `TSAT` | °C | Primary Saturation Temperature | Thermodynamics | None | **MEDIUM** | Saturation temperature derived from primary pressure. |
| `QRHR` | MWth | Residual Heat Removal System Power | Decay Heat | None | **LOW** | Shutdown cooling system. |
| `LVCR` | m | Core Collapsed Liquid Level | Reactor Vessel | None | **DO NOT TRANSFER** | PHWR core has horizontal fuel channels submerged in calandria, not a vertical collapsed level. |
| `SCMA` | °C | Subcooling Margin Loop A | Primary Margin | None | **HIGH** | Direct safety margin variable ($T_{\text{sat}} - T_{\text{hot}}$). |
| `SCMB` | °C | Subcooling Margin Loop B | Primary Margin | None | **HIGH** | Subcooling margin loop B. |
| `FRCL` | % | Clad Oxidation Fraction | Fuel Integrity | None | **LOW** | Severe accident degradation parameter. |
| `PRB` | kPa | Containment Building Pressure | Containment | **Ch 11** | **HIGH** | Direct physical equivalent of containment envelope pressure. |
| `PRBA` | kPa | Containment Atmosphere Pressure | Containment | None | **HIGH** | Redundant to PRB. |
| `TRB` | °C | Containment Building Temperature | Containment | None | **MEDIUM** | Reactor building atmospheric temperature. |
| `LWRB` | m | Containment Sump Water Level | Containment | None | **LOW** | Post-LOCA water inventory. |
| `DNBR` | - | Departure from Nucleate Boiling Ratio| Thermal Margin | None | **HIGH** | Critical safety margin indicator (PWR DNBR vs PHWR Critical Channel Power margin). |
| `QFCL` | MW | Core Heat Flux to Coolant | Fuel Clad | None | **HIGH** | Physical heat transfer from fuel pins to coolant. |
| `WBK` | kg/s | Break Discharge Flow Rate | Accident Leak | None | **HIGH** | Primary leak rate during LOCA. |
| `WSPY` | kg/s | Containment Spray Flow | Safety Systems | None | **LOW** | Engineered safety feature actuation. |
| `WCSP` | kg/s | Core Spray Flow | Safety Systems | None | **LOW** | ECCS core spray. |
| `HTR` | kW | Pressurizer Electric Heater Power | Pressure Control | None | **DO NOT TRANSFER** | PWR pressurizer heating control; distinct from PHWR pressure control. |
| `MH2` | kg | Mass of Hydrogen Generated | Containment Gas | None | **LOW** | Severe accident oxidation product. |
| `CNH2` | % | Containment Hydrogen Concentration | Containment Gas | None | **LOW** | Flammability safety parameter. |
| `RHBR` | % | Reactor Building Relative Humidity | Containment | None | **LOW** | Environmental parameter. |
| `RHMT` | MW | Metal-Water Reaction Energy Rate | Severe Accident | None | **LOW** | Exothermic chemical energy release. |
| `RHFL` | MW | Fuel Fission Product Decay Heat | Decay Heat | None | **HIGH** | Physical decay heat curve after scram. |
| `RHRD` | MW | Radiation Decay Heat | Decay Heat | None | **LOW** | Radiation energy dissipation. |
| `RH` | % | Relative Humidity | Environment | None | **LOW** | Atmospheric humidity. |
| `PWNT` | % | Total Nuclear Fission Power | Neutronics | None | **HIGH** | Redundant to PWR. |
| `PWR` | % | Normalized Reactor Fission Power | Neutronics | **Ch 2** | **HIGH** | Direct physical equivalent of neutron power; mapped to flux via scaling. |
| `TFSB` | °C | Fuel Surface Clad Temperature Bulk | Fuel Clad | **Ch 8** | **MEDIUM** | Cladding surface temperature. |
| `TFPK` | °C | Peak Cladding Temperature | Fuel Clad | None | **HIGH** | Safety limit variable ($1204^\circ\text{C}$ licensing limit). |
| `TF` | °C | Fuel Centerline Temperature | Fuel Pellet | None | **HIGH** | Centerline ceramic $UO_2$ temperature. |
| `TPCT` | °C | Peak Clad Temperature Transformed | Fuel Clad | None | **HIGH** | Redundant to TFPK. |
| `WCFT` | kg/s | Accumulator / Flood Tank Flow | Safety ECCS | None | **LOW** | Passive nitrogen-driven accumulator discharge. |
| `WLPI` | kg/s | Low Pressure Safety Injection Flow | Safety ECCS | None | **LOW** | Long-term recirculation injection. |
| `WCHG` | kg/s | CVCS Charging Pump Flow | Chemical Control | None | **DO NOT TRANSFER** | PWR chemical volume control system flow. |
| `RM1` | mR/hr | Containment Radiation Monitor | Radiation | **Ch 3** | **MEDIUM** | In-containment gamma dose rate; qualitative indicator of fission product release. |
| `RM2` | mR/hr | Primary Coolant Activity Monitor | Radiation | None | **LOW** | Fuel defect / delayed neutron activity monitor. |
| `RM3` | mR/hr | Main Steam Line Radiation Monitor | Radiation | None | **HIGH** | Primary-to-secondary leak detection (crucial SGTR signature). |
| `RM4` | mR/hr | Stack Effluent Radiation Monitor | Radiation | None | **LOW** | Environmental release pathway. |
| `RC87` | Ci | Kr-87 Noble Gas Fission Activity | Radiochemistry | None | **LOW** | Specific isotopic tracking. |
| `RC131` | Ci | I-131 Volatile Radioiodine Activity | Radiochemistry | None | **LOW** | Biological hazard isotopic tracking. |
| `STRB` | kg/s | Atmospheric Steam Dump Flow | Secondary Relief | None | **LOW** | Secondary pressure control. |
| `STSG` | kg/s | Steam Generator Safety Relief Flow | Secondary Relief | None | **LOW** | ASME code overpressure protection. |
| `STTB` | kg/s | Steam Bypass to Condenser Flow | Secondary Bypass | None | **LOW** | Turbine trip bypass flow. |
| `RBLK` | %/day | Reactor Building Leakage Rate | Containment | None | **LOW** | Containment envelope leak rate. |
| `SGLK` | kg/s | SG Tube Leakage Flow | Secondary Leak | None | **HIGH** | SGTR physical mass transfer rate. |
| `DTHY` | s | Core Thermal Delay Time | Hydraulics | None | **LOW** | Simulation time constant. |
| `DWB` | kg/s | Downcomer Annulus Flow | Vessel Hydraulics | None | **DO NOT TRANSFER** | PWR downcomer geometry does not exist in PHWR. |
| `WRLA` | kg/s | Reactor Coolant Pump A Flow | Primary Pumps | None | **MEDIUM** | Primary pump delivery. |
| `WRLB` | kg/s | Reactor Coolant Pump B Flow | Primary Pumps | None | **MEDIUM** | Primary pump delivery. |
| `WLD` | kg/s | CVCS Letdown Flow | Chemical Control | None | **DO NOT TRANSFER** | PWR chemical letdown flow. |
| `MBK` | kg | Integrated Break Mass Loss | Accident Balance| None | **MEDIUM** | Cumulative mass lost from primary system. |
| `EBK` | MJ | Integrated Break Energy Release | Accident Balance| None | **MEDIUM** | Cumulative thermal energy dumped into containment. |
| `TKLV` | m | Refueling Water Tank Level | Water Storage | None | **LOW** | ECCS water supply reservoir. |
| `FRZR` | - | Fuel Relocation Fraction | Severe Accident | None | **DO NOT TRANSFER** | Severe accident core melt geometry. |
| `TDBR` | °C | Core Debris Bed Temperature | Severe Accident | None | **DO NOT TRANSFER** | Corium pool temperature. |
| `MDBR` | kg | Core Debris Bed Mass | Severe Accident | None | **DO NOT TRANSFER** | Relocated core debris mass. |
| `MCRT` | kg | Molten Core Mass | Severe Accident | None | **DO NOT TRANSFER** | Molten fuel mass. |
| `MGAS` | kg | Containment Non-Condensable Gas | Containment | None | **LOW** | Atmospheric gas accumulation. |
| `TCRT` | °C | Core Relocation Temperature | Severe Accident | None | **DO NOT TRANSFER** | Melt relocation temperature. |
| `TSLP` | °C | Containment Sump Water Temperature | Containment | None | **LOW** | Recirculation sump temperature. |
| `PPM` | ppm | Boric Acid Concentration | Reactivity | None | **DO NOT TRANSFER** | PWR soluble boron reactivity control; PHWR uses liquid zone controllers & adjusters. |
| `RRCA` | steps | Control Bank A Position | Reactivity | None | **LOW** | Discrete control rod steps. |
| `RRCB` | steps | Control Bank B Position | Reactivity | None | **LOW** | Discrete control rod steps. |
| `RRCO` | fraction | Regulating Rod Out Position (0-1) | Reactivity | **Ch 6** | **MEDIUM** | Normalized rod position ($0 \to 100\%$). |
| `WFLB` | kg/s | Feedwater Line Break Leakage Flow | Secondary Leak | None | **HIGH** | Secondary side break mass flow rate. |

---

## 3. PUR-1 (Purdue Research Reactor) — Complete 9-Variable Audit

Sampling Rate: $\Delta t = 1.0\text{ s}$.

| Variable | Raw Column | Unit | Physical Meaning | Reactor System | PRAJNA Target | Transfer Validity | Physical Justification / Notes |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| `index` | `index` | count | Temporal sample index | Data Logger | None | **DO NOT TRANSFER** | Raw integer row index. |
| `nfd-1-cps` | `nfd-1-cps` | counts/s | Nuclear Flux Detector 1 Count Rate | Core Neutronics | **Ch 2** | **HIGH (Noise / Dynamics)** | Real ionization chamber response; excellent for testing temporal noise, reactor period, and trip response. Scaled to relative flux. |
| `nfd-1-cr` | `nfd-1-cr` | counts/s | Compensated Ion Chamber Count Rate | Core Neutronics | None | **HIGH (Noise / Dynamics)** | Redundant neutron flux detection channel. |
| `rr-active-state` | `rr-active-state` | binary | Regulating Rod Auto-Control Mode | Control Console | None | **LOW** | Operational state indicator (manual vs automatic regulation). |
| `rr-position` | `rr-position` | % | Regulating Rod Physical Position | Reactivity System | **Ch 6** | **HIGH (Control Dynamics)** | Fine reactivity shim control position ($0 \to 100\%$). |
| `ss1-active-state`| `ss1-active-state`| binary | Safety Shim Rod 1 State | Safety Interlocks | None | **LOW** | Rod latch status. |
| `ss1-position` | `ss1-position` | % | Safety Shim Rod 1 Position | Reactivity System | **Ch 6** | **HIGH (Scram Dynamics)** | Coarse control & rapid gravity scram rod position ($0 \to 100\%$). Drop time captures physical scram kinetics. |
| `ss2-active-state`| `ss2-active-state`| binary | Safety Shim Rod 2 State | Safety Interlocks | None | **LOW** | Rod latch status. |
| `ss2-position` | `ss2-position` | % | Safety Shim Rod 2 Position | Reactivity System | **Ch 6** | **HIGH (Scram Dynamics)** | Safety rod 2 position; redundant scram verification channel. |

---

## 4. Validated Ingestion Rules & Transfer Policies

### Rule 1: Zero Artificial Temperature / Pressure Transfer from Research Reactors
- PUR-1 is an open atmospheric pool reactor operating at ambient pool temperatures ($20 - 25^\circ\text{C}$) and $1\text{ bar}$ hydrostatic head.
- **Strict Prohibition:** Under no circumstances will PUR-1 temperatures or pressures be scaled to $85\text{ bar}$ or $293^\circ\text{C}$ and claimed as commercial PHWR thermal-hydraulic validation.
- **Allowed Transfer:** Only the **neutron flux kinetics, rod drop transients, and genuine measurement noise distributions** are transferred to test PRAJNA's temporal feature extractors.

### Rule 2: Multi-Simulator Cross-Validation Subsetting
- For cross-simulator testing against NPPAD, only the **7 verified High-validity physical channels** enter the evaluation:
  1. Primary Pressure (`P` $\to$ Ch 4)
  2. Core Exit Temperature (`THA` $\to$ Ch 0)
  3. Core Inlet Temperature (`TCA` $\to$ Ch 10)
  4. Core Thermal Power (`QMWT` $\to$ Ch 5)
  5. Coolant Flow (`WRCA` $\to$ Ch 1, normalized by $Q_0/\Delta T_0$)
  6. Normalized Neutron Flux (`PWR` $\to$ Ch 2)
  7. Containment Pressure (`PRB` $\to$ Ch 11)
- Variables classified as **DO NOT TRANSFER** (e.g. `PPM`, `VOL`, `LVCR`, `HTR`) are permanently excluded from cross-domain models to ensure physical validity.
