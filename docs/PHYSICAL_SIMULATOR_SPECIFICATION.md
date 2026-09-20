# PRAJNA v1: Physical ODE Simulator Specification & Design Document
**Document ID:** PRAJNA-SPEC-SIM-V1  
**Target Reference:** Scaled Pressurized Heavy Water Reactor (PHWR-like Surrogate)  
**Status:** Approved for Implementation (v1-physical-ode)

---

## 1. Physical State Vector $\mathbf{x}(t)$

The complete dynamic state of the closed-loop nuclear steam supply system is represented by the 17-dimensional state vector:

$$\mathbf{x}(t) = \begin{bmatrix} n(t) \\ C_1(t) \\ C_2(t) \\ C_3(t) \\ C_4(t) \\ C_5(t) \\ C_6(t) \\ C_7(t) \\ T_{\text{fuel}}(t) \\ T_{\text{out}}(t) \\ T_{\text{in}}(t) \\ \dot{m}(t) \\ P_{\text{primary}}(t) \\ L_{\text{pzr}}(t) \\ M_{\text{primary}}(t) \\ P_{\text{cont}}(t) \\ A_{\text{rad}}(t) \end{bmatrix} \in \mathbb{R}^{17}$$

| State Variable | Symbol | SI Units | Nominal Steady-State (t=0) | Physical Description |
| :--- | :--- | :--- | :--- | :--- |
| **Neutron Density** | $n(t)$ | $\text{n/m}^3$ (norm) | $1.000$ (normalized) | Total thermal neutron population |
| **Delayed Precursors (6-group)** | $C_{1..6}(t)$ | $\text{atoms/m}^3$ | Equilibrium $C_i(0) = \frac{\beta_i}{\Lambda \lambda_i} n(0)$ | Standard Keepin delayed neutron groups |
| **Photoneutron Precursors** | $C_7(t)$ | $\text{atoms/m}^3$ | Equilibrium $\frac{\beta_{\text{photo}}}{\Lambda \lambda_{\text{photo}}} n(0)$ | $(\gamma, n)$ reactions on deuterium ($D_2O$) |
| **Fuel Centerline Temp** | $T_{\text{fuel}}(t)$ | $^\circ\text{C}$ (or $\text{K}$) | $580.0^\circ\text{C}$ | Lumped fuel pin ceramic pellet ($UO_2$) node |
| **Core Outlet Coolant Temp** | $T_{\text{out}}(t)$ | $^\circ\text{C}$ | $293.0^\circ\text{C}$ | Reactor Outlet Header (ROH) coolant temperature |
| **Core Inlet Coolant Temp** | $T_{\text{in}}(t)$ | $^\circ\text{C}$ | $249.0^\circ\text{C}$ | Reactor Inlet Header (RIH) from Steam Generator |
| **Primary Mass Flow Rate** | $\dot{m}(t)$ | $\text{kg/s}$ | $3,700.0\text{ kg/s}$ (Total Core) | Primary Heat Transport (PHT) mass flow |
| **Primary Pressure** | $P_{\text{primary}}(t)$ | $\text{MPa}$ (or $\text{bar}$) | $8.70\text{ MPa}$ ($87.0\text{ bar}$) | Primary header operating pressure |
| **Pressurizer Water Level** | $L_{\text{pzr}}(t)$ | $\%$ | $50.0\%$ | Level in surge tank / pressurizer |
| **Primary Coolant Mass** | $M_{\text{primary}}(t)$ | $\text{kg}$ | $45,000\text{ kg}$ | Total primary circuit inventory |
| **Containment Pressure** | $P_{\text{cont}}(t)$ | $\text{kPa}$ | $101.325\text{ kPa}$ | Reactor containment building pressure |
| **Containment Radiation** | $A_{\text{rad}}(t)$ | $\text{mSv/h}$ | $0.40\text{ mSv/h}$ | Core & containment gamma area activity |

---

## 2. Dynamic Governing Equations (Coupled Closed Loop)

### 2.1 Stiff Point Kinetics with Exact Matrix-Exponential Integrator
The 7-group delayed neutron system (Keepin 6 groups + 1 effective photoneutron group) is expressed as a linear state space:

$$\frac{d}{dt} \begin{bmatrix} n(t) \\ \mathbf{C}(t) \end{bmatrix} = \mathbf{A}(\rho(t)) \begin{bmatrix} n(t) \\ \mathbf{C}(t) \end{bmatrix}$$

$$\mathbf{A}(\rho) = \begin{bmatrix} \frac{\rho(t) - \beta_{\text{tot}}}{\Lambda} & \lambda_1 & \lambda_2 & \dots & \lambda_7 \\ \frac{\beta_1}{\Lambda} & -\lambda_1 & 0 & \dots & 0 \\ \frac{\beta_2}{\Lambda} & 0 & -\lambda_2 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \frac{\beta_7}{\Lambda} & 0 & 0 & \dots & -\lambda_7 \end{bmatrix} \in \mathbb{R}^{8 \times 8}$$

Over each timestep $\Delta t$, with $\rho$ treated as piecewise constant, the state is integrated **exactly** without numerical stiffness or truncation error:
$$\begin{bmatrix} n(t + \Delta t) \\ \mathbf{C}(t + \Delta t) \end{bmatrix} = \exp\left( \mathbf{A}(\rho(t)) \cdot \Delta t \right) \begin{bmatrix} n(t) \\ \mathbf{C}(t) \end{bmatrix}$$
implemented via `torch.linalg.matrix_exp`.

#### Net Reactivity Feedback:
$$\rho(t) = \rho_{\text{control}}(t) + \alpha_{\text{fuel}} \left(T_{\text{fuel}}(t) - T_{\text{fuel},0}\right) + \alpha_{\text{coolant}} \left(\bar{T}_{\text{coolant}}(t) - \bar{T}_{c,0}\right) + \alpha_{\text{void}} \cdot \alpha_{\text{void}}(t)$$
*Note:* In PHWRs, the fuel temperature coefficient $\alpha_{\text{fuel}} < 0$ (Doppler broadening), while the void reactivity coefficient $\alpha_{\text{void}} > 0$ (positive void feedback on coolant boiling during LOCA).

### 2.2 Two-Node Core Thermal Hydraulics (Fuel + Coolant)
1. **Fuel Node Energy Balance**:
   $$M_{\text{fuel}} C_{p,\text{fuel}} \frac{d T_{\text{fuel}}}{dt} = P(t) - U_{\text{fc}} A_{\text{clad}} \left( T_{\text{fuel}}(t) - \bar{T}_{\text{coolant}}(t) \right)$$
   where $\bar{T}_{\text{coolant}}(t) = \frac{T_{\text{out}}(t) + T_{\text{in}}(t)}{2}$, and $P(t) = P_{\text{fission}}(t) + P_{\text{decay}}(t)$.

2. **Core Coolant Lumped Energy Balance**:
   $$M_{\text{core}} C_{p,\text{cool}} \frac{d T_{\text{out}}}{dt} = U_{\text{fc}} A_{\text{clad}} \left( T_{\text{fuel}}(t) - \bar{T}_{\text{coolant}}(t) \right) - \dot{m}(t) C_{p,\text{cool}} \left( T_{\text{out}}(t) - T_{\text{in}}(t) \right)$$
   Summing both equations yields the overall core heat balance:
   $$\frac{d}{dt} \left( M_{\text{fuel}} C_{p,\text{fuel}} T_{\text{fuel}} + M_{\text{core}} C_{p,\text{cool}} T_{\text{out}} \right) = P(t) - \dot{m}(t) C_{p,\text{cool}} \left( T_{\text{out}}(t) - T_{\text{in}}(t) \right)$$

### 2.3 Closed-Loop Steam Generator (Heat Sink) Node
The inlet temperature $T_{\text{in}}(t)$ is **never scripted**; it is governed by primary-to-secondary heat transfer in the steam generator:
$$M_{\text{sg}} C_{p,\text{cool}} \frac{d T_{\text{in}}}{dt} = \dot{m}(t) C_{p,\text{cool}} \left( T_{\text{out}}(t) - T_{\text{in}}(t) \right) - U_{\text{sg}} A_{\text{sg}} \left( \bar{T}_{\text{sg}}(t) - T_{\text{secondary}}(t) \right)$$
During Station Blackout (SBO), secondary feedwater trips ($\dot{m}_{\text{fw}} \to 0$) and the secondary sink temperature rises, causing $T_{\text{in}}$ to dynamically evolve according to physical heat sink degradation.

### 2.4 Momentum & Natural Circulation Flow
$$\frac{d\dot{m}}{dt} = \frac{\Delta P_{\text{pump}}(t) - K_{\text{friction}} \dot{m}^2 + \Delta P_{\text{buoyancy}}}{\tau_{\text{momentum}}}$$
- Normal operation: Pump head balances friction $\Delta P_{\text{pump}} = K_{\text{friction}} \dot{m}_0^2$.
- SBO (Pump Coastdown): $\Delta P_{\text{pump}}(t) \to 0$. As flow coasts down, buoyancy driving head engages:
  $$\Delta P_{\text{buoyancy}} = \rho_0 g \beta_{\text{exp}} \Delta z_{\text{thermal}} \left( T_{\text{out}}(t) - T_{\text{in}}(t) \right)$$
  Yielding the established natural circulation scaling law:
  $$\dot{m}_{\text{nat}} \propto \left( \Delta T \right)^{1/2} \quad \text{and} \quad \dot{m}_{\text{nat}} \propto Q^{1/3}$$

### 2.5 Primary Inventory & Pressure (LOCA / Break Flow)
$$\frac{d M_{\text{primary}}}{dt} = \dot{m}_{\text{makeup}}(t) - \dot{m}_{\text{break}}(t)$$
- Break flow is modeled via choked two-phase orifice dynamics:
  $$\dot{m}_{\text{break}}(t) = C_d A_{\text{break}} \sqrt{2 \rho_c \max\left(0, P_{\text{primary}}(t) - P_{\text{cont}}(t)\right)}$$
- Primary pressure depressurization:
  $$\frac{d P_{\text{primary}}}{dt} = \frac{1}{V_{\text{pzr}} \kappa_T} \left( \beta_T \frac{d\bar{T}_c}{dt} - \frac{\dot{m}_{\text{break}}}{\rho_c} + \frac{\dot{m}_{\text{surge}}}{\rho_c} \right)$$
- Containment mass & pressure rise:
  $$\frac{d P_{\text{cont}}}{dt} = \frac{R_{\text{steam}}}{V_{\text{cont}}} \dot{m}_{\text{break}}(t) T_{\text{break}} - \lambda_{\text{condense}} \left( P_{\text{cont}} - P_{\text{cont},0} \right)$$

---

## 3. Observation Map: $\mathbf{y}(t) = h(\mathbf{x}(t))$ (The 12 SCADA Instruments)

All 12 telemetry channels are derived directly from the physical state vector $\mathbf{x}(t)$:

| Ch ID | Instrument Name | State Mapping Function $h_i(\mathbf{x})$ | Units | Nominal Value |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Core Exit Temperature** | $y_0 = T_{\text{out}}(t)$ | $^\circ\text{C}$ | $293.0^\circ\text{C}$ |
| **1** | **Coolant Mass Flow** | $y_1 = \dot{m}(t)$ | $\text{kg/s}$ | $3,700.0\text{ kg/s}$ |
| **2** | **Neutron Flux** | $y_2 = \Phi_0 \cdot n(t)$ | $10^{13}\text{ n/cm}^2\text{s}$ | $2.25 \times 10^{13}$ |
| **3** | **Radiation Level** | $y_3 = A_{\text{rad}}(t) = A_0 + \gamma_{\text{fp}} \int \dot{m}_{\text{break}} dt$ | $\text{mSv/h}$ | $0.40\text{ mSv/h}$ |
| **4** | **Primary Pressure** | $y_4 = P_{\text{primary}}(t)$ | $\text{bar}$ | $87.0\text{ bar}$ |
| **5** | **Core Power** | $y_5 = P(t) = P_{\text{fission}}(t) + P_{\text{decay}}(t)$ | $\text{MWth}$ | $756.0\text{ MWth}$ |
| **6** | **Control Rod Height** | $y_6 = \text{Rod Position}(\%)$ | $\%$ | $65.0\%$ |
| **7** | **Pressurizer Level** | $y_7 = L_{\text{pzr}}(t) = 50.0 + \Delta V_{\text{surge}} / V_0$ | $\%$ | $50.0\%$ |
| **8** | **Feedwater Temperature** | $y_8 = T_{\text{secondary}}(t)$ | $^\circ\text{C}$ | $185.0^\circ\text{C}$ |
| **9** | **Steam Flow Rate** | $y_9 = \dot{m}_{\text{steam}}(t)$ | $\text{kg/s}$ | $1,050.0\text{ kg/s}$ |
| **10** | **Core Inlet Temperature** | $y_{10} = T_{\text{in}}(t)$ | $^\circ\text{C}$ | $249.0^\circ\text{C}$ |
| **11** | **Containment Pressure** | $y_{11} = P_{\text{cont}}(t)$ | $\text{kPa}$ | $101.3\text{ kPa}$ |

---

## 4. Fundamental Physics Residuals (Verification Criteria)

### 4.1 Equilibrium Residual ($t = 0$)
At steady state, $dT/dt = 0$, requiring:
$$R_{\text{steady}} = \left| P_0 - \dot{m}_0 C_{p,\text{cool}} \left( T_{\text{out},0} - T_{\text{in},0} \right) \right| < 0.20\text{ MWth}$$

### 4.2 Dynamic Enthalpy Residual ($t > 0$)
During any transient, First-Law energy conservation demands:
$$R_{\text{dynamic}}(t) = M_{\text{core}} C_{p,\text{cool}} \frac{d T_{\text{out}}}{dt} - \left[ U_{\text{fc}} A \left(T_{\text{fuel}} - \bar{T}_c\right) - \dot{m}(t) C_{p,\text{cool}} \left(T_{\text{out}}(t) - T_{\text{in}}(t)\right) \right] = 0$$

### 4.3 Whole-Run Stored Energy Integral Test
Over an entire operational or transient window of duration $T_{\text{trans}}$:
$$\int_0^{T_{\text{trans}}} \left( P(t) - \dot{m}(t) C_{p,\text{cool}} \left( T_{\text{out}}(t) - T_{\text{in}}(t) \right) \right) dt = \Delta E_{\text{stored}} = M_{\text{fuel}} C_{p,\text{fuel}} \Delta T_{\text{fuel}} + M_{\text{core}} C_{p,\text{cool}} \Delta T_{\text{out}}$$
The relative error on this integral must be $< 0.1\%$ to numerical integrator tolerance.

---

## 5. Parameter Table & Sources

| Parameter | Symbol | Value | Units | Source / Reference |
| :--- | :--- | :--- | :--- | :--- |
| Core Thermal Power | $P_0$ | $756.0$ | $\text{MWth}$ | NPCIL Indian 220 MWe PHWR Safety Report |
| Total Primary Flow | $\dot{m}_0$ | $3,700.0$ | $\text{kg/s}$ | 306 channels $\times 12.1\text{ kg/s}$ per feeder pair |
| Coolant Specific Heat ($D_2O$ at $270^\circ\text{C}, 9\text{ MPa}$) | $C_{p,\text{cool}}$ | $4.650$ | $\text{kJ/(kg}\cdot\text{K)}$ | IAPWS / CoolProp HeavyWater Formulation |
| Fuel Specific Heat ($UO_2$ at $600^\circ\text{C}$) | $C_{p,\text{fuel}}$ | $0.315$ | $\text{kJ/(kg}\cdot\text{K)}$ | MATPRO Nuclear Materials Database |
| Core Coolant Mass | $M_{\text{core}}$ | $12,500.0$ | $\text{kg}$ | 306 pressure tubes + headers |
| Fuel Inventory Mass | $M_{\text{fuel}}$ | $52,000.0$ | $\text{kg}$ | 19-rod bundle fuel inventory |
| Prompt Neutron Lifetime | $\Lambda$ | $1.05 \times 10^{-3}$ | $\text{s}$ | Natural Uranium / Heavy Water Lattice |
| Total Delayed Fraction | $\beta_{\text{tot}}$ | $0.0075$ | $-$ | Keepin ($\beta_{\text{fiss}} = 0.0065$) + Photoneutrons ($\beta_{\text{photo}} = 0.0010$) |
| Doppler Coefficient | $\alpha_{\text{fuel}}$ | $-1.85 \times 10^{-5}$ | $\Delta k/k / ^\circ\text{C}$ | IAEA SRS No. 29 |
| Coolant Void Coefficient | $\alpha_{\text{void}}$ | $+1.20 \times 10^{-4}$ | $\Delta k/k / \%\text{void}$ | Positive void feedback in natural U PHWR |
