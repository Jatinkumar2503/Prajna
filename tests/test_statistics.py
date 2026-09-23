"""
Unit tests for prajna_core/statistics.py
Verifies non-parametric bootstrap confidence intervals, paired Wilcoxon tests, and edge cases.
"""

import unittest
import numpy as np
from prajna_core.statistics import bootstrap_ci, paired_significance_test, compute_multi_seed_comparison_table


class TestStatisticsSuite(unittest.TestCase):
    def test_01_bootstrap_ci_coverage_and_bounds(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=10.0, scale=2.0, size=50)
        mean_val, ci_low, ci_high = bootstrap_ci(data, n_bootstraps=1000, ci=0.95, seed=42)

        self.assertAlmostEqual(mean_val, float(np.mean(data)), places=4)
        self.assertTrue(ci_low < mean_val < ci_high, f"Expected {ci_low} < {mean_val} < {ci_high}")
        # Standard error of mean ~ 2 / sqrt(50) = 0.28, 95% CI width ~ 1.1
        self.assertTrue(9.0 <= ci_low <= 10.5)
        self.assertTrue(10.0 <= ci_high <= 11.5)

    def test_02_bootstrap_ci_single_sample_and_empty(self):
        # Empty array
        m, l, h = bootstrap_ci(np.array([]))
        self.assertEqual((m, l, h), (0.0, 0.0, 0.0))

        # Single element
        m, l, h = bootstrap_ci(np.array([42.5]))
        self.assertEqual((m, l, h), (42.5, 42.5, 42.5))

    def test_03_paired_significance_test_significant_difference(self):
        # Model A clearly beats Model B across 10 seeds
        a = np.array([78.5, 79.0, 77.2, 80.1, 81.3, 79.5, 82.0, 78.9, 80.5, 81.0])
        b = np.array([65.2, 66.0, 64.5, 67.3, 68.0, 66.8, 69.1, 65.9, 67.4, 68.2])

        res = paired_significance_test(a, b, test_type="wilcoxon")
        self.assertTrue(res["significant"])
        self.assertLess(res["p_value"], 0.01)
        self.assertGreater(res["effect_size_cohens_d"], 5.0)
        self.assertEqual(res["n_pairs"], 10)
        self.assertTrue("Paired Wilcoxon signed-rank test" in res["test_name"])

    def test_04_paired_significance_test_null_case(self):
        # Identical models
        a = np.array([50.0, 52.0, 51.0, 49.0, 50.5])
        res = paired_significance_test(a, a, test_type="wilcoxon")
        self.assertFalse(res["significant"])
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["mean_difference"], 0.0)

    def test_05_multi_seed_comparison_table(self):
        metrics_a = {"acc": [75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0]}
        metrics_b = {"acc": [70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0]}
        comp = compute_multi_seed_comparison_table(metrics_a, metrics_b, name_a="PRAJNA", name_b="Baseline")
        self.assertIn("acc", comp)
        self.assertTrue(comp["acc"]["paired_test"]["significant"])


if __name__ == "__main__":
    unittest.main()
