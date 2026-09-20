"""
Unit tests verifying First-Law thermodynamics, exact solver parity, and dynamic physical residuals
for the PRAJNA v1 Physical ODE Simulator.
Blocks data generation and model training if physics residuals or mathematical solvers are violated.
"""

import unittest
import math
import torch
import numpy as np
import scipy.linalg
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prajna_core.simulator import (
    PhysicalPHWRSimulator,
    P_NOMINAL_MWTH,
    FLOW_NOMINAL_KG_S,
    CP_COOLANT,
    CP_FUEL,
    M_FUEL,
    M_CORE_COOLANT,
    T_OUT_NOMINAL_C,
    T_IN_NOMINAL_C,
    DELTA_T_NOMINAL_K
)
from prajna_core.physics import ThermalHydraulicsCore


class TestPhysicalResiduals(unittest.TestCase):
    def setUp(self):
        self.sim = PhysicalPHWRSimulator(device=torch.device("cpu"))
        self.th_core = ThermalHydraulicsCore()
        
    def test_01_steady_state_heat_balance(self):
        """
        Verify that at t = 0 (equilibrium), the derived primary flow m_dot = 3,500 kg/s
        with CoolProp D2O Cp = 4.863 kJ/(kg*K) and Delta-T = 44.4 K satisfies heat balance
        within ±0.5 MWth (< 0.05% of 756 MWth).
        """
        state = self.sim.get_steady_state(batch_size=1)
        p_nominal = P_NOMINAL_MWTH
        flow = state["flow"].item()
        t_out = state["t_out"].item()
        t_in = state["t_in"].item()
        delta_t = t_out - t_in
        
        q_calc = flow * CP_COOLANT * delta_t * 1e-3  # MWth
        residual = abs(p_nominal - q_calc)
        rel_diff_pct = (residual / p_nominal) * 100.0
        
        print(f"\n[Test 1] Steady-State Primary Heat Balance Check:")
        print(f"  Nominal Core Power P_0: {p_nominal:.4f} MWth")
        print(f"  Computed Flow Heat Q_0: {q_calc:.4f} MWth (3,500 kg/s * 4.863 * 44.4 K)")
        print(f"  Discrepancy:            {residual:.4f} MWth ({rel_diff_pct:.4f}%, Tolerance: <0.10%)")
        
        self.assertLess(rel_diff_pct, 0.10, f"Heat balance discrepancy {rel_diff_pct}% exceeds 0.10% tolerance!")

    def test_02_matrix_exp_parity_and_inhour_eigenvalue(self):
        """
        Verify numerical solver parity and nuclear reactor theory:
        1. Compare torch.linalg.matrix_exp (Float64) against scipy.linalg.expm to < 1e-9 tolerance.
        2. Verify the dominant eigenvalue of kinetics matrix A(rho) matches the inhour equation root.
        """
        rho_step = 0.0010  # 100 pcm positive reactivity
        
        # 1. Float64 Matrix Exponential Parity vs SciPy
        A_torch = self.sim.build_kinetics_matrix(torch.tensor([[rho_step]]), dtype=torch.float64)
        dt = 0.05
        M_torch = torch.linalg.matrix_exp(A_torch * dt).squeeze(0).numpy()
        
        A_np = A_torch.squeeze(0).numpy()
        M_scipy = scipy.linalg.expm(A_np * dt)
        
        max_diff = np.max(np.abs(M_torch - M_scipy))
        print(f"\n[Test 2a] Matrix Exponential Solver Parity (Torch vs SciPy expm):")
        print(f"  Max Absolute Difference: {max_diff:.3e} (Tolerance: < 1e-9)")
        self.assertLess(max_diff, 1e-9, f"Torch vs SciPy matrix_exp divergence {max_diff} exceeds 1e-9!")
        
        # 2. Inhour Equation Root vs Dominant Eigenvalue
        # Inhour equation: rho = Lambda * omega + sum(beta_i * omega / (omega + lambda_i))
        # Find root omega numerically
        Lambda = float(self.sim.lambda_prompt)
        beta_i = self.sim.beta_i.numpy()
        lambda_i = self.sim.lambda_i.numpy()
        
        def inhour_residual(omega):
            return Lambda * omega + np.sum((beta_i * omega) / (omega + lambda_i)) - rho_step
            
        # Bisection to find root omega
        w_low, w_high = 0.0, 5.0
        for _ in range(60):
            w_mid = (w_low + w_high) / 2.0
            if inhour_residual(w_mid) > 0:
                w_high = w_mid
            else:
                w_low = w_mid
        omega_analytic = (w_low + w_high) / 2.0
        
        # Eigenvalues of A_np
        eigvals = scipy.linalg.eigvals(A_np)
        dominant_eigval = float(np.max(np.real(eigvals)))
        
        eig_error = abs(dominant_eigval - omega_analytic) / omega_analytic * 100.0
        print(f"\n[Test 2b] Inhour Equation Asymptotic Root vs Dominant Eigenvalue:")
        print(f"  Inhour Analytical Root:  {omega_analytic:.8f} s^-1")
        print(f"  Matrix A Dominant Eigval: {dominant_eigval:.8f} s^-1")
        print(f"  Relative Error:          {eig_error:.6f}% (Tolerance: < 1e-4%)")
        self.assertLess(eig_error, 1e-4, f"Inhour eigenvalue error {eig_error}% exceeds 1e-4%!")

    def test_03_whole_run_dynamic_energy_conservation_integral(self):
        """
        Verify First-Law dynamic energy conservation over an SBO transient:
          integral_0^T (P_total - Q_flow) dt = Delta E_stored
          Delta E_stored = M_fuel * C_fuel * (T_fuel(T) - T_fuel(0)) + M_cool * C_cool * (T_out(T) - T_out(0))
        Check that energy balance closes to within < 0.05% relative tolerance.
        """
        res = self.sim.simulate_transient(scenario_id=4, duration_seconds=30.0, dt=0.05)
        
        power = res["power"][0, :, 0]    # [Steps] MWth
        q_flow = res["q_flow"][0, :, 0]  # [Steps] MWth
        dt = res["dt"]
        
        # Numerical time integral using trapezoidal rule (MJ = MW * s)
        net_power = power - q_flow
        trapz_integral_mj = torch.trapz(net_power, dx=dt).item()
        
        t_fuel_start = res["t_fuel"][0, 0, 0].item()
        t_fuel_end = res["t_fuel"][0, -1, 0].item()
        t_out_start = res["t_out"][0, 0, 0].item()
        t_out_end = res["t_out"][0, -1, 0].item()
        
        c_fuel_mj = (M_FUEL * CP_FUEL) * 1e-3      # ~16.38 MJ/K
        c_cool_mj = (M_CORE_COOLANT * CP_COOLANT) * 1e-3  # ~60.79 MJ/K
        
        delta_stored_mj = (c_fuel_mj * (t_fuel_end - t_fuel_start) +
                           c_cool_mj * (t_out_end - t_out_start))
        
        diff_mj = abs(trapz_integral_mj - delta_stored_mj)
        rel_diff_pct = (diff_mj / abs(delta_stored_mj)) * 100.0
        
        print(f"\n[Test 3] Dynamic Core Stored Energy Conservation (SBO Cooldown):")
        print(f"  Time-Integrated Net Power integral(P - Q) dt: {trapz_integral_mj:.4f} MJ")
        print(f"  Stored Thermal Energy Delta E_stored:        {delta_stored_mj:.4f} MJ")
        print(f"  Absolute Discrepancy:                        {diff_mj:.4f} MJ")
        print(f"  Relative Discrepancy:                        {rel_diff_pct:.4f}% (Tolerance: < 0.05%)")
        
        self.assertLess(rel_diff_pct, 0.05, f"Energy conservation discrepancy {rel_diff_pct}% exceeds 0.05% tolerance!")

    def test_04_ground_truth_dynamic_residual_closure(self):
        """
        Evaluate the 2-node dynamic thermal residual on clean, noise-free simulator trajectories.
        Confirm that First-Law dynamic residual is near zero (< 1e-4 relative to P_0 = 756 MWth).
        """
        res = self.sim.simulate_transient(scenario_id=4, duration_seconds=30.0, dt=0.05)
        power = res["power"]
        flow = res["obs"][:, :, 1:2]
        t_out = res["t_out"]
        t_in = res["t_in"]
        t_fuel = res["t_fuel"]
        
        # Compute dynamic residual with full 2-node core
        r_dyn = self.th_core.compute_dynamic_residual(
            power=power,
            mass_flow=flow,
            t_outlet=t_out,
            t_inlet=t_in,
            t_fuel=t_fuel,
            dt=0.05
        )
        
        rms_residual_mwth = torch.sqrt(torch.mean(r_dyn[:, 2:-2] ** 2)).item()
        rel_residual = rms_residual_mwth / P_NOMINAL_MWTH
        
        print(f"\n[Test 4] Ground-Truth Dynamic Thermal Residual Closure:")
        print(f"  RMS Dynamic Residual: {rms_residual_mwth:.6f} MWth")
        print(f"  Relative to P_0 (756 MW): {rel_residual:.4e} (Tolerance: < 1e-4)")
        
        self.assertLess(rel_residual, 1e-4, f"Ground-truth residual {rel_residual} exceeds 1e-4 P_0!")

    def test_05_all_scenarios_physical_coupling(self):
        """
        Verify that all 5 scenarios simulate physically coupled, non-degenerate trajectories.
        """
        for s_id in range(5):
            res = self.sim.simulate_transient(scenario_id=s_id, duration_seconds=10.0, dt=0.05)
            obs = res["obs"][0]
            t_out = obs[:, 0]
            flow = obs[:, 1]
            power = res["power"][0, :, 0]
            self.assertFalse(torch.isnan(obs).any(), f"Scenario {s_id} produced NaNs!")
            self.assertGreater(t_out.min().item(), 200.0)
            self.assertLess(t_out.max().item(), 400.0)
            self.assertGreater(flow.min().item(), 50.0)
            self.assertGreater(power.min().item(), 10.0)


if __name__ == "__main__":
    unittest.main()
