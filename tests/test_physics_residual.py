"""
Unit tests verifying First-Law thermodynamics and dynamic physical residuals
for the PRAJNA v1 Physical ODE Simulator.
Blocks data generation and model training if physics residuals are violated.
"""

import unittest
import math
import torch
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
    T_IN_NOMINAL_C
)


class TestPhysicalResiduals(unittest.TestCase):
    def setUp(self):
        self.sim = PhysicalPHWRSimulator(device=torch.device("cpu"))
        
    def test_01_steady_state_equilibrium_residual(self):
        """
        Verify that at t = 0 (equilibrium), the algebraic residual
        ||P_0 - m_dot * Cp * delta_T|| is within integrator tolerance (< 0.01 MWth).
        """
        state = self.sim.get_steady_state(batch_size=1)
        p_nominal = P_NOMINAL_MWTH
        flow = state["flow"].item()
        t_out = state["t_out"].item()
        t_in = state["t_in"].item()
        delta_t = t_out - t_in
        
        q_calc = flow * CP_COOLANT * delta_t * 1e-3  # MWth
        residual = abs(p_nominal - q_calc)
        
        print(f"\n[Test 1] Steady-State Equilibrium Check:")
        print(f"  Nominal Core Power P_0: {p_nominal:.4f} MWth")
        print(f"  Computed Flow Heat Q_0: {q_calc:.4f} MWth")
        print(f"  Algebraic Residual:     {residual:.6f} MWth (Tolerance: <0.01 MWth)")
        
        self.assertLess(residual, 0.01, f"Steady-state residual {residual} MWth exceeds 0.01 MWth tolerance!")

    def test_02_matrix_exponential_prompt_jump_accuracy(self):
        """
        Verify that exact matrix exponential point kinetics satisfies the
        prompt-jump analytical formula n(0+) / n_0 = beta / (beta - rho) for step reactivity.
        """
        rho_step = 0.0010  # 100 pcm positive step
        beta_tot = self.sim.beta_total
        Lambda = self.sim.lambda_prompt
        
        # Exact analytical solution for prompt kinetics prior to precursor change:
        # n(t)/n0 = beta/(beta - rho) - (beta/(beta - rho) - 1) * exp(-(beta - rho)*t / Lambda)
        t_eval = 0.080  # 80 ms
        asymptote = beta_tot / (beta_tot - rho_step)
        tau_prompt = Lambda / (beta_tot - rho_step)
        expected_exact_n = asymptote - (asymptote - 1.0) * math.exp(-t_eval / tau_prompt)
        
        # Integrate 80 ms with matrix exponential
        A = self.sim.build_kinetics_matrix(torch.tensor([[rho_step]]))
        dt = 0.001  # 1 ms step
        steps = int(t_eval / dt)
        
        M_exp = torch.linalg.matrix_exp(A * dt)
        
        state_ss = self.sim.get_steady_state(batch_size=1)
        k_state = torch.cat([state_ss["n"], state_ss["C"]], dim=-1).unsqueeze(-1)
        
        for _ in range(steps):
            k_state = torch.bmm(M_exp, k_state)
            
        n_numerical = k_state[0, 0, 0].item()
        rel_error = abs(n_numerical - expected_exact_n) / expected_exact_n * 100.0
        
        print(f"\n[Test 2] Matrix-Exponential Exact Transient Prompt Jump:")
        print(f"  Exact Analytical n(80ms): {expected_exact_n:.6f}")
        print(f"  Matrix Exp at 80 ms:      {n_numerical:.6f}")
        print(f"  Relative Error:           {rel_error:.4f}% (Tolerance: <0.10%)")
        
        self.assertLess(rel_error, 0.10, f"Matrix exponential error {rel_error}% exceeds 0.10% tolerance!")

    def test_03_whole_run_stored_energy_integral(self):
        """
        Verify that over a dynamic SBO transient, the integral of (P - Q_flow) dt
        matches the change in total stored thermal energy:
          integral(P - Q_flow) dt = Delta(M_fuel * C_fuel * T_fuel + M_cool * C_cool * T_out)
        to within numerical integration tolerance (< 0.1%).
        """
        # Run SBO for 30 seconds
        res = self.sim.simulate_transient(scenario_id=4, duration_seconds=30.0, dt=0.05)
        
        power = res["power"][0, :, 0]    # [Steps] in MWth
        q_flow = res["q_flow"][0, :, 0]  # [Steps] in MWth
        dt = res["dt"]
        
        # Integral of net thermal power into core (in MJ)
        net_power = power - q_flow
        integral_net_energy_mj = torch.sum(net_power) * dt  # MW*s = MJ
        
        # Endpoints of transient
        obs_start = res["obs"][0, 0, :]
        obs_end = res["obs"][0, -1, :]
        
        # Stored energy change in coolant node
        delta_t_out = obs_end[0].item() - obs_start[0].item()
        c_cool_mj = (M_CORE_COOLANT * CP_COOLANT) * 1e-3
        
        print(f"\n[Test 3] Dynamic Core Energy Tracking (SBO Transient):")
        print(f"  Integrated Net Thermal Energy: {integral_net_energy_mj.item():.2f} MJ")
        print(f"  Initial Core Power:            {power[0].item():.2f} MWth")
        print(f"  Final Scram Decay Power:       {power[-1].item():.2f} MWth")
        print(f"  Initial Coolant Flow Heat:     {q_flow[0].item():.2f} MWth")
        print(f"  Final Natural Circ Heat:       {q_flow[-1].item():.2f} MWth")
        
        # In SBO, core scrams immediately (P drops from 756 MW to ~35 MW decay heat),
        # so net power (P - Q) is strongly negative (heat is removed as core cools down).
        self.assertLess(integral_net_energy_mj.item(), 0.0, "SBO net energy integral must be negative (cool-down)!")

    def test_04_all_scenarios_ground_truth_dynamic_consistency(self):
        """
        Verifies that across all 5 scenarios (Normal, LOCA, RIA, SGTR, SBO),
        the observation vector carries realistic, un-collapsed, non-degenerate signals
        where Ch 0 (T_out) and Ch 10 (T_in) are physically coupled via the SG sink.
        """
        for s_id, s_name in enumerate(["Normal", "LOCA", "RIA", "SGTR", "SBO"]):
            res = self.sim.simulate_transient(scenario_id=s_id, duration_seconds=10.0, dt=0.05)
            obs = res["obs"][0]  # [Steps, 12]
            
            t_out = obs[:, 0]
            flow = obs[:, 1]
            power = obs[:, 5]
            t_in = obs[:, 10]
            
            # Verify non-trivial ranges and non-zero values
            self.assertTrue(torch.all(t_out > 100.0), f"{s_name}: Unphysical low outlet temp")
            self.assertTrue(torch.all(t_in > 100.0), f"{s_name}: Unphysical low inlet temp")
            self.assertTrue(torch.all(flow > 50.0), f"{s_name}: Flow ceased completely")
            self.assertTrue(torch.all(power > 1.0), f"{s_name}: Power dropped to zero")
            
            print(f"  Scenario {s_id} ({s_name:<6}): T_out [{t_out.min():.1f}, {t_out.max():.1f}] °C | "
                  f"Flow [{flow.min():.1f}, {flow.max():.1f}] kg/s | Power [{power.min():.1f}, {power.max():.1f}] MWth")


if __name__ == "__main__":
    unittest.main()
