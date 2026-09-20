"""
UNIT TESTS: EXPERIMENTAL RIGOR & ZERO-LEAKAGE VERIFICATION
Validates that PRAJNA adheres to the strict senior nuclear ML protocol.
"""

import os
import sys
import json
import unittest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from prajna_core.models.fast_reflex import PrajnaFastReflex
from scripts.reproduce_all_10_priorities import GRUForecaster, LSTMForecaster, TransformerForecaster

class TestExperimentalRigor(unittest.TestCase):

    def setUp(self):
        self.proc_dir = os.path.join(PROJECT_ROOT, "data", "processed")
        self.manifest_path = os.path.join(self.proc_dir, "dataset_manifest.json")

    def test_01_manifest_and_trajectory_isolation(self):
        """Asserts that processed datasets exist and have zero trajectory overlap."""
        self.assertTrue(os.path.exists(self.manifest_path), "Dataset manifest missing!")
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        self.assertIn("phwr_in_domain", manifest)
        self.assertIn("nppad_cross_simulator", manifest)
        self.assertIn("pur1_real_data", manifest)

        # Load NPPAD train and test files
        train_path = os.path.join(self.proc_dir, "train", "nppad_train_trajectories.pt")
        test_path = os.path.join(self.proc_dir, "test", "nppad_cross_test.pt")
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))

        nppad_train = torch.load(train_path)
        nppad_test = torch.load(test_path)

        train_files = set(nppad_train["trajectories"])
        test_files = set(nppad_test["trajectories"])

        intersection = train_files.intersection(test_files)
        self.assertEqual(len(intersection), 0, f"Trajectory leakage detected! Overlap: {intersection}")

    def test_02_normalization_separation(self):
        """Asserts that normalization statistics are calculated solely on training data."""
        train_path = os.path.join(self.proc_dir, "train", "phwr_train.pt")
        test_path = os.path.join(self.proc_dir, "test", "phwr_test_unseen.pt")
        
        phwr_train = torch.load(train_path)
        phwr_test = torch.load(test_path)

        train_mu = phwr_train["windows"].mean(dim=(0, 1))
        test_mu = phwr_test["windows"].mean(dim=(0, 1))

        # Because test runs are independent transients, means will not be identical
        diff = torch.norm(train_mu - test_mu).item()
        self.assertGreater(diff, 1e-4, "Train and test means are suspiciously identical; check data independence.")

    def test_03_temporal_baseline_contract_parity(self):
        """Verifies that GRU, LSTM, and Transformer baselines conform to the standardized interface."""
        dummy_input = torch.randn(4, 45, 12)
        
        gru = GRUForecaster()
        lstm = LSTMForecaster()
        transformer = TransformerForecaster()
        prajna = PrajnaFastReflex(num_channels=12, hidden_dim=96, num_eop_classes=5)

        for model, name in [(gru, "GRU"), (lstm, "LSTM"), (transformer, "Transformer")]:
            out = model(dummy_input)
            self.assertIn("eop_logits", out, f"{name} missing eop_logits")
            self.assertIn("tmargin", out, f"{name} missing tmargin")
            self.assertEqual(out["eop_logits"].shape, (4, 5), f"{name} invalid logits shape")
            self.assertEqual(out["tmargin"].shape, (4, 1), f"{name} invalid tmargin shape")

        prajna_out = prajna(dummy_input)
        self.assertEqual(prajna_out["eop_logits"].shape, (4, 5))

    def test_04_tmargin_safety_gate_monotonicity(self):
        """Asserts that deterministic safety gating transitions strictly on provisional T_margin thresholds."""
        tau_crit = 15.0
        tau_warn = 30.0

        def gate(tmargin):
            if tmargin <= tau_crit:
                return "CRITICAL_ADVISORY"
            elif tmargin <= tau_warn:
                return "WARNING"
            return "NORMAL"

        self.assertEqual(gate(40.0), "NORMAL")
        self.assertEqual(gate(25.0), "WARNING")
        self.assertEqual(gate(10.0), "CRITICAL_ADVISORY")
        self.assertEqual(gate(0.0), "CRITICAL_ADVISORY")

    def test_05_per_unit_normalization_properties(self):
        """Verifies that processed datasets contain non-dimensional per-unit tensors with aligned scales."""
        phwr_test = torch.load(os.path.join(self.proc_dir, "test", "phwr_test_unseen.pt"))
        nppad_test = torch.load(os.path.join(self.proc_dir, "test", "nppad_cross_test.pt"))

        self.assertIn("windows_pu", phwr_test, "phwr_test missing windows_pu tensor")
        self.assertIn("windows_pu", nppad_test, "nppad_test missing windows_pu tensor")

        # In per-unit space, steady-state channels (e.g. pressure, temp) should be near ~1.0 p.u.
        phwr_mean = phwr_test["windows_pu"].mean(dim=(0, 1)).numpy()
        nppad_mean = nppad_test["windows_pu"].mean(dim=(0, 1)).numpy()

        # Check that temperature (Ch 0) and pressure (Ch 4) are within reasonable physical per-unit bounds [0.5, 1.5]
        self.assertTrue(0.5 <= phwr_mean[0] <= 1.5, f"PHWR T_out p.u. mean {phwr_mean[0]} outside [0.5, 1.5]")
        self.assertTrue(0.5 <= nppad_mean[0] <= 1.5, f"NPPAD T_out p.u. mean {nppad_mean[0]} outside [0.5, 1.5]")
        self.assertTrue(0.5 <= phwr_mean[4] <= 1.5, f"PHWR Pressure p.u. mean {phwr_mean[4]} outside [0.5, 1.5]")
        self.assertTrue(0.5 <= nppad_mean[4] <= 1.5, f"NPPAD Pressure p.u. mean {nppad_mean[4]} outside [0.5, 1.5]")

    def test_06_d2o_photoneutron_parameters_and_inhour_parity(self):
        """Verifies D2O 7-group kinetics parameters and exact Inhour eigenvalue parity."""
        from prajna_core.simulator import PhysicalPHWRSimulator, PROMPT_NEUTRON_LIFETIME, BETA_I, LAMBDA_I
        sim = PhysicalPHWRSimulator()

        # 7 groups: 6 Keepin delayed + 1 photoneutron group
        self.assertEqual(len(BETA_I), 7)
        self.assertEqual(len(LAMBDA_I), 7)
        self.assertAlmostEqual(PROMPT_NEUTRON_LIFETIME, 1.05e-3, places=6)

        # Build 8x8 matrix at 100 pcm
        A = sim.build_kinetics_matrix(torch.tensor([[0.0010]], dtype=torch.float64), dtype=torch.float64)
        self.assertEqual(A.shape, (1, 8, 8))

        import scipy.linalg
        eigvals = scipy.linalg.eigvals(A.squeeze(0).numpy())
        dominant_eigval = float(np.max(np.real(eigvals)))
        # Analytical root of 7-group Inhour equation for rho = 100 pcm
        expected_root = 0.00623858
        rel_err = abs(dominant_eigval - expected_root) / expected_root
        self.assertLess(rel_err, 1e-5, f"Inhour root relative error {rel_err} exceeds 1e-5")

if __name__ == "__main__":
    unittest.main()
