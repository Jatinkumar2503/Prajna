"""
PRAJNA EMPIRICAL STATISTICS & HYPOTHESIS TESTING ENGINE
======================================================
Provides rigorous statistical methods for peer-reviewed benchmark comparisons:
1. Non-parametric empirical bootstrap resampling for confidence intervals (95% CI).
2. Paired hypothesis tests:
   - Paired Wilcoxon signed-rank test (non-parametric, distribution-free).
   - Paired Student's t-test (parametric reference).
3. Effect size estimation (Cohen's d for paired observations).
4. Trajectory-level and multi-seed bootstrap confidence intervals.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import scipy.stats as stats


def bootstrap_ci(data: np.ndarray,
                 n_bootstraps: int = 1000,
                 ci: float = 0.95,
                 seed: Optional[int] = 42,
                 statistic_fn: Callable[[np.ndarray], float] = np.mean) -> Tuple[float, float, float]:
    """
    Computes non-parametric empirical percentile bootstrap confidence interval.
    
    Args:
        data: 1D array of observed metric values across seeds or trajectories.
        n_bootstraps: Number of bootstrap resamples (default B=1,000).
        ci: Confidence interval fraction (default 0.95).
        seed: Random seed for deterministic reproducibility.
        statistic_fn: Function to compute on each bootstrap sample (default np.mean).
        
    Returns:
        (sample_stat, ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        val = float(arr[0])
        return val, val, val

    rng = np.random.default_rng(seed)
    n = len(arr)
    sample_stat = float(statistic_fn(arr))

    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    boot_samples = arr[boot_indices]
    boot_stats = np.apply_along_axis(statistic_fn, 1, boot_samples)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_stats, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))

    return sample_stat, ci_lower, ci_upper


def paired_significance_test(a: np.ndarray,
                             b: np.ndarray,
                             test_type: str = "wilcoxon",
                             alternative: str = "two-sided") -> Dict[str, Any]:
    """
    Computes paired statistical significance tests comparing model A against model B
    evaluated on identical seeds or matched trajectory test splits.
    
    Args:
        a: Array of metric observations for Model A (e.g. PRAJNA).
        b: Array of metric observations for Model B (e.g. Baseline).
        test_type: 'wilcoxon' (default non-parametric) or 'ttest' (parametric paired t-test).
        alternative: 'two-sided', 'less', or 'greater'.
        
    Returns:
        Dictionary containing test name, statistic, p-value, effect size, and significance conclusion.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Paired test requires equal sample lengths, got {len(arr_a)} vs {len(arr_b)}")

    n = len(arr_a)
    diff = arr_a - arr_b

    # Check for identical values / zero variance
    if np.allclose(diff, 0.0):
        return {
            "test_name": "Paired Wilcoxon signed-rank test" if test_type == "wilcoxon" else "Paired Student's t-test",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "effect_size_cohens_d": 0.0,
            "mean_difference": 0.0,
            "diff_ci_95": [0.0, 0.0],
            "n_pairs": n,
            "summary": "p = 1.0000 (No difference detected)"
        }

    # Cohen's d for paired samples: mean(diff) / std(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = float(np.mean(diff) / std_diff) if std_diff > 1e-12 else 0.0

    # 95% Bootstrap CI of the paired mean difference
    _, diff_ci_low, diff_ci_high = bootstrap_ci(diff, n_bootstraps=1000, ci=0.95, seed=42)

    if test_type == "wilcoxon":
        test_name = "Paired Wilcoxon signed-rank test"
        try:
            res = stats.wilcoxon(arr_a, arr_b, alternative=alternative)
            stat = float(res.statistic)
            p_val = float(res.pvalue)
        except Exception:
            # Fallback to paired t-test if zero differences prevent Wilcoxon ranking
            test_name = "Paired Student's t-test (Wilcoxon fallback)"
            res = stats.ttest_rel(arr_a, arr_b, alternative=alternative)
            stat = float(res.statistic)
            p_val = float(res.pvalue)
    else:
        test_name = "Paired Student's t-test"
        res = stats.ttest_rel(arr_a, arr_b, alternative=alternative)
        stat = float(res.statistic)
        p_val = float(res.pvalue)

    is_sig = bool(p_val < 0.05)
    p_formatted = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"

    return {
        "test_name": test_name,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 5),
        "significant": is_sig,
        "effect_size_cohens_d": round(cohens_d, 3),
        "mean_difference": round(float(np.mean(diff)), 4),
        "diff_ci_95": [round(diff_ci_low, 4), round(diff_ci_high, 4)],
        "n_pairs": n,
        "summary": f"{test_name}, p {p_formatted} ({'Statistically significant at alpha=0.05' if is_sig else 'Not significant'})"
    }


def compute_multi_seed_comparison_table(metrics_a: Dict[str, List[float]],
                                        metrics_b: Dict[str, List[float]],
                                        name_a: str = "PRAJNA",
                                        name_b: str = "Baseline") -> Dict[str, Dict[str, Any]]:
    """
    Computes comprehensive comparison between Model A and Model B across all evaluated metrics:
    Includes mean, 95% bootstrap CI, and paired Wilcoxon signed-rank tests.
    """
    report = {}
    for metric_name in metrics_a:
        if metric_name not in metrics_b:
            continue
        vals_a = np.asarray(metrics_a[metric_name], dtype=np.float64)
        vals_b = np.asarray(metrics_b[metric_name], dtype=np.float64)

        mean_a, a_low, a_high = bootstrap_ci(vals_a, seed=101)
        mean_b, b_low, b_high = bootstrap_ci(vals_b, seed=202)
        test_res = paired_significance_test(vals_a, vals_b, test_type="wilcoxon")

        report[metric_name] = {
            f"{name_a}_mean": round(mean_a, 4),
            f"{name_a}_ci95": [round(a_low, 4), round(a_high, 4)],
            f"{name_b}_mean": round(mean_b, 4),
            f"{name_b}_ci95": [round(b_low, 4), round(b_high, 4)],
            "paired_test": test_res
        }
    return report
