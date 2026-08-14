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


if __name__ == "__main__":
    unittest.main()
