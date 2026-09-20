# PRAJNA Modeling Assumptions

This document outlines the core scientific and engineering assumptions governing the PRAJNA physics engine and validation framework.

---

## 1. Reactor Kinetics & Neutronics Assumptions

1. **Seven-Group Point Reactor Kinetics with D2O Photoneutrons:**
   - Spatial flux distribution is assumed to be in the fundamental spatial mode during operational transients.
   - Six delayed neutron precursor groups follow standard Keepin parameters for $U^{235}$ thermal fission.
   - Heavy water ($D_2O$) moderation introduces high-energy gamma-induced deuterium photodisintegration ($^2\text{H} + \gamma \to ^1\text{H} + n$, $Q = 2.223\text{ MeV}$), modeled as a 7th precursor group ($\beta_7 = 0.001000, \lambda_7 = 0.0050\text{ s}^{-1}$).
   - Total effective delayed fraction: $\beta_{\text{eff}} = 0.007502$ ($\sim 750\text{ pcm}$).
   - Prompt neutron lifetime is $\Lambda = 1.05 \times 10^{-3}\text{ s}$, reflecting the long thermal diffusion time in heavy water.
   - The Inhour equation $\rho = \Lambda \omega + \sum_{i=1}^7 \frac{\beta_i \omega}{\omega + \lambda_i}$ for $\rho = 100\text{ pcm}$ ($0.133\ \$$) yields an analytical asymptotic root of $\omega = 0.00623858\text{ s}^{-1}$ ($T \approx 160.3\text{ s}$), matching the dominant eigenvalue of the 8x8 kinetics matrix $A(\rho)$ to within $0.000001\%$.
2. **Reactivity Feedback Coefficients:**
   - Doppler fuel temperature reactivity coefficient: $\alpha_{\text{fuel}} = -1.85 \times 10^{-5}\ \Delta k/k/^\circ\text{C}$.
   - Coolant temperature reactivity coefficient: $\alpha_{\text{cool}} = -0.45 \times 10^{-5}\ \Delta k/k/^\circ\text{C}$.
   - Positive coolant void coefficient: $\alpha_{\text{void}} = +1.20 \times 10^{-4}\ \Delta k/k/\%\text{void}$.

---

## 2. Thermal-Hydraulics & Heavy Water Thermodynamics

1. **Lumped-Parameter Single-Phase Primary Loop:**
   - The primary heat transport system is modeled as a lumped fuel node coupled to a lumped coolant node with heat transfer coefficient $h_c A = 4.0\text{ MW/K}$.
   - Primary coolant properties are evaluated using IAPWS / CoolProp formulations for heavy water ($D_2O$) at nominal $8.5\text{ MPa}$ ($85.0\text{ bar}$) across the operating range $249.0^\circ\text{C} \to 293.4^\circ\text{C}$:
     $$C_{p,\text{cool}} = 4.863\text{ kJ/(kg}\cdot\text{K)}, \quad \rho_{\text{cool}} \approx 870.0\text{ kg/m}^3$$
2. **Core Energy Conservation:**
   - Thermal power generation in fuel pellets transfers to primary coolant via convection:
     $$M_{\text{fuel}} C_{p,\text{fuel}} \frac{dT_{\text{fuel}}}{dt} = P(t) - h_c A (T_{\text{fuel}} - \bar{T}_{\text{cool}})$$
   - Coolant enthalpy rise across the core balances the heat removal:
     $$M_{\text{core}} C_{p,\text{cool}} \frac{dT_{\text{out}}}{dt} = h_c A (T_{\text{fuel}} - \bar{T}_{\text{cool}}) - \dot{m} C_{p,\text{cool}} (T_{\text{out}} - T_{\text{in}})$$
3. **Secondary Heat Sink:**
   - Steam generators are modeled as a lumped secondary node at saturation ($T_{\text{sec}} = 245.0^\circ\text{C}$, $P_{\text{sec}} = 36.5\text{ bar}$) with natural circulation heat transfer capability during loss of forced circulation.

---

## 3. Sensor & Instrumentation Modeling Assumptions

1. **Thermowell Thermal Lag:**
   - Primary RTD temperature sensors are modeled with a first-order lag filter ($\tau = 3.5\text{ s}$) representing physical thermowell thermal inertia.
2. **Measurement Noise & Quantization:**
   - Noise is modeled with realistic instrumentation standard deviations ($\sigma_T = \pm 0.50^\circ\text{C}$, $\sigma_P = \pm 0.75\text{ bar}$, $\sigma_{\dot{m}} = \pm 5.0\text{ kg/s}$, $\sigma_\phi = \pm 0.015 \times 10^{13}\text{ n/cm}^2\cdot\text{s}$) plus 14-bit ADC quantization.
3. **Sensor Drift:**
   - Monotonic sensor calibration drift is modeled up to $0.05\%/\text{hour}$ over long continuous operational windows.
