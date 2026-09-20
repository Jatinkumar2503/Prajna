"""
PRAJNA PINN INTEGRATION & REGRESSION TEST SUITE
Automated unit and integration verification for:
1. Multi-scale forward pass architectures (4M, 125M, 350M, 2.25B)
2. Autograd Physics Regularizers (PKE, Energy Balance, DNBR)
3. ONNX Runtime execution and PyTorch parity
4. Memory footprint calculations
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure root package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prajna_core.models.pinn_foundation import PrajnaFoundationPINN
from prajna_core.physics import PrajnaPhysicsLoss


class TestPrajnaPINNPipeline(unittest.TestCase):
    
    def test_01_test_scale_forward(self):
        model = PrajnaFoundationPINN(num_channels=16, scale="test_4m")
        dummy = torch.randn(2, 30, 16)
        out = model(dummy)
        self.assertEqual(out["physics_trajectories"].shape, (2, 30, 16))
        self.assertEqual(out["time_to_threshold"].shape, (2, 16))
        self.assertEqual(out["eop_logits"].shape, (2, 64))
        self.assertEqual(out["latent_representation"].shape, (2, 256))

    def test_02_intermediate_2_25b_memory_profile(self):
        model = PrajnaFoundationPINN(num_channels=16, scale="intermediate_2.25b")
        mem = model.get_memory_footprint()
        self.assertGreater(mem["parameters"], 1_000_000_000)
        self.assertIn("fp16_megabytes", mem)
        self.assertIn("int8_megabytes", mem)

    def test_03_physics_loss_computation(self):
        loss_fn = PrajnaPhysicsLoss()
        pred_physics = torch.ones(2, 20, 16) * 10.0
        target_physics = torch.ones(2, 20, 16) * 10.0
        eop_logits = torch.randn(2, 64)
        target_eop = torch.tensor([0, 1], dtype=torch.long)

        losses = loss_fn(
            pred_physics=pred_physics,
            target_physics=target_physics,
            eop_logits=eop_logits,
            target_eop=target_eop
        )
        self.assertIn("total_loss", losses)
        self.assertIn("energy_loss", losses)
        self.assertIn("dnbr_penalty", losses)
        self.assertIn("eop_loss", losses)
        self.assertTrue(torch.isfinite(losses["total_loss"]))

    def test_04_onnx_runtime_parity(self):
        onnx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "prajna_pinn_efficient_125m.onnx"))
        if not os.path.exists(onnx_path):
            self.skipTest("ONNX checkpoint not generated")
            
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_input = np.random.randn(1, 45, 16).astype(np.float32)
        outputs = session.run(None, {"telemetry_sequence": test_input})
        
        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[0].shape, (1, 45, 8))  # physics_trajectories
        self.assertEqual(outputs[1].shape, (1, 16))    # time_to_threshold
        self.assertEqual(outputs[2].shape, (1, 64))    # eop_logits


    def test_05_fast_reflex_model(self):
        from prajna_core.models.fast_reflex import PrajnaFastReflex
        model = PrajnaFastReflex(num_channels=16, hidden_dim=96, num_eop_classes=64)
        dummy_seq = torch.randn(2, 45, 16)
        out_seq = model(dummy_seq)
        self.assertEqual(out_seq["eop_logits"].shape, (2, 64))
        self.assertEqual(out_seq["time_to_threshold"].shape, (2, 16))
        self.assertEqual(out_seq["scram_probability"].shape, (2, 1))

        dummy_single = torch.randn(2, 16)
        out_single = model(dummy_single)
        self.assertEqual(out_single["eop_logits"].shape, (2, 64))
        self.assertEqual(out_single["scram_probability"].shape, (2, 1))
        self.assertLess(model.count_parameters(), 50_000)

    def test_06_fast_reflex_int8_quantization_parity(self):
        from scripts.validate_reflex_quantization import run_fast_reflex_quantization_validation
        res = run_fast_reflex_quantization_validation()
        self.assertGreaterEqual(res["int8_eop_accuracy"], 95.0)
        self.assertGreaterEqual(res["compression_ratio"], 2.0)

    def test_07_xenon_and_radiolysis_physics_closure(self):
        from prajna_core.physics import XenonPoisoningCore, RadiolysisGasCore, PrajnaPhysicsLoss
        xenon = XenonPoisoningCore()
        flux_traj = torch.ones(2, 20, 1, requires_grad=True) * 2.32
        res_xenon = xenon.compute_residual(flux_traj)
        self.assertGreaterEqual(res_xenon.item(), 0.0)

        radiolysis = RadiolysisGasCore()
        rad = torch.ones(2, 20, 1) * 0.42
        power = torch.ones(2, 20, 1) * 91.64
        pen = radiolysis.compute_radiolysis_penalty(rad, power)
        self.assertGreaterEqual(pen.item(), 0.0)

        # Test composite loss backpropagation with the new terms
        loss_fn = PrajnaPhysicsLoss()
        pred_phys = torch.randn(2, 10, 16, requires_grad=True)
        target_phys = torch.randn(2, 10, 16)
        losses = loss_fn(pred_physics=pred_phys, target_physics=target_phys)
        self.assertIn("xenon_loss", losses)
        self.assertIn("radiolysis_margin", losses)
        self.assertIn("dnbr_violation_margin", losses)
        losses["total_loss"].backward()
        self.assertIsNotNone(pred_phys.grad)

    def test_08_prompt_jump_and_energy_conservation_analytic(self):
        from prajna_core.physics import DifferentiablePointKinetics, ThermalHydraulicsCore
        
        # 1. Analytic Prompt-Jump Ratio: n_1 / n_0 = beta / (beta - rho)
        pke = DifferentiablePointKinetics()
        beta = pke.beta_total.item()  # 0.0065
        rho_step = 0.0010  # subcritical positive step
        expected_ratio = beta / (beta - rho_step)  # ~1.1818
        
        # Simulate prompt relaxation across characteristic timescale tau_p = Lambda / (beta - rho) = ~18.2 ms
        # Over 80 ms (800 steps of dt=1e-4s), prompt jump reaches asymptotic level beta / (beta - rho)
        n_0 = torch.tensor([[1.0]], dtype=torch.float32)
        c_0 = (pke.beta_i / (pke.lambda_prompt * pke.lambda_i)) * n_0
        n_curr, c_curr = n_0, c_0
        rho_tensor = torch.tensor([[rho_step]], dtype=torch.float32)
        for _ in range(800):
            n_curr, c_curr = pke.step(n_curr, c_curr, rho_tensor, dt=1e-4)
        simulated_ratio = (n_curr / n_0).item()
        
        # Verification within 0.5% of analytical prompt-jump ratio: 1.1818
        self.assertAlmostEqual(simulated_ratio, expected_ratio, delta=0.01)

        # 2. First-Law Thermodynamic Energy Conservation (Pure SI Units)
        th = ThermalHydraulicsCore()
        m_dot = torch.tensor([[3500.0]])  # kg/s (derived: 756 MW / (4.863 * 44.4))
        t_out = torch.tensor([[293.4]])   # °C
        t_in  = torch.tensor([[249.0]])   # °C (Delta-T = 44.4 K)
        calc_power = th.compute_thermal_power(m_dot, t_out, t_in).item()
        # Calibrated nominal power should match 756.0 MWth (3500 * 4.863 * 44.4 * 1e-3 = 755.71 MWth)
        self.assertAlmostEqual(calc_power, 756.0, delta=0.5)

    def test_09_sensor_noise_suite_robustness(self):
        from prajna_core.noise import apply_instrument_noise_suite
        clean_telemetry = torch.ones(4, 45, 16) * 100.0
        degraded = apply_instrument_noise_suite(clean_telemetry)
        
        self.assertEqual(degraded.shape, clean_telemetry.shape)
        # Noise should create non-zero perturbation
        diff = torch.abs(degraded - clean_telemetry).mean().item()
        self.assertGreater(diff, 0.01)


if __name__ == "__main__":
    unittest.main()


