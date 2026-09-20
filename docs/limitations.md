# PRAJNA Scientific Limitations & Boundary Conditions

This document explicitly delineates what PRAJNA cannot claim, detailing the mathematical and empirical boundary conditions of the system.

---

## 1. Domain Transferability Limitations

1. **PHWR vs PWR Fundamental Domain Gap:**
   - PRAJNA’s primary physics engine is developed for a horizontal pressure-tube Heavy Water Reactor (PHWR-220 surrogate).
   - Pressurized Water Reactors (PWRs) utilize vertical pressure vessels, enriched uranium oxide fuel, light water coolant, soluble chemical boron for reactivity control, and have a strongly negative coolant temperature coefficient.
   - **Limitation:** Cross-simulator validation against PCTRAN (PWR) can only evaluate macro-thermodynamic trends (mass/energy balance, power excursions) and **cannot** be claimed as direct licensing-grade validation for PWR operations without reactor-specific calibration.
2. **Research Reactor vs Commercial Power Reactor Gap (PUR-1):**
   - PUR-1 is a $1.0\text{ kWth}$ open-pool research reactor operating at atmospheric pressure and ambient temperatures ($20 - 25^\circ\text{C}$).
   - **Limitation:** PUR-1 data **cannot validate** thermal-hydraulic models, pressurization dynamics, voiding feedback, or critical heat flux predictions for commercial $756\text{ MWth}$ power reactors. Its utility is strictly limited to real ionization chamber noise evaluation, scram dynamics, and temporal feature tracking under real hardware conditions.

---

## 2. Mathematical & Simulator Limitations

1. **Absence of 3D Spatial Neutronics:**
   - The physics engine utilizes 6-group point reactor kinetics. It does not resolve localized flux tilting, regional xenon oscillations, or 3D control rod shadow effects.
2. **Homogeneous Lumped Coolant Model:**
   - Coolant boiling is modeled with macroscopic bulk void fractions; individual subchannel flow distributions, two-phase drift-flux slip, and localized departure from nucleate boiling (DNB) are not resolved at subchannel grid level.
3. **Absence of Commercial Plant SCADA Access:**
   - Indian PHWR operating telemetry (NPCIL) is proprietary and national security restricted. PRAJNA relies on an open surrogate validated against published IAEA ARIS reports and thermodynamic property formulations.

---

## 3. Sensor Dependency & Single-Channel Fragility

1. **Primary Channel Vulnerability:**
   - Single-sensor dropout experiments ($k=1$ with mean imputation) reveal severe asymmetric dependence: masking Core Power produces a $\sim 40-60\%$ degradation in accident classification accuracy, and masking Coolant Flow produces a $\sim 20\%$ drop.
   - **Safety Boundary:** PRAJNA cannot claim intrinsic sensor redundancy for primary thermal-hydraulic variables. In an operational nuclear deployment, single-sensor fragility mandates independent defense-in-depth and hardware voting (e.g. 2-out-of-3 or 2-out-of-4 sensor channels) to prevent sensor-blindness or common-cause failures.

---

## 4. Advisory Scope & Regulatory Boundaries

1. **Provisional Thresholds & Advisory Gating:**
   - All safety margins ($T_{\text{margin}}$) are calculated relative to **provisional engineering thresholds** (e.g. $T_{\text{out}} \ge 305^\circ\text{C}$, $P \le 50\text{ bar}$, $\dot{m} \le 500\text{ kg/s}$).
   - The safety gate produces **CRITICAL advisory** and **WARNING** notifications for human operators. PRAJNA is an unlicensed algorithmic decision-support prototype and does not qualify under IEEE 603 / IEEE 7-4.3.2 / AERB Class 1E nuclear safety I&C standards.

---

## 5. Neural Surrogate Dynamic Energy Residual

1. **Finite Physics Loss Convergence:**
   - While the differentiable physics loss $L_{\text{dyn}}$ enforces thermodynamic coupling and reduces dynamic energy violations by $58.0\%$ compared to unconstrained temporal networks, neural surrogates retain a non-zero residual ($\sim 78\text{ MWth}$, $\sim 10\%$ of $P_0$) compared to clean physical ODE trajectories ($0.0107\text{ MWth}$).
   - This residual reflects the multi-objective Pareto trade-off between cross-entropy accident classification, finite MLP/CNN capacity, and discrete forward-Euler numerical time differentiation.
