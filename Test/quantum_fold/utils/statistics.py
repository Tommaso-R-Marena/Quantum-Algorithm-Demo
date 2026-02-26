"""
statistics.py
Statistical analysis tools for quantum protein folding experiments.

Provides:
  • Bootstrap confidence intervals
  • Approximation ratio computation
  • Time-to-solution (TTS) estimate
  • Cohen's d effect size
  • Wilcoxon signed-rank test
  • LaTeX table generation
  • Summary statistics

References:
  [1] Efron & Tibshirani, An Introduction to the Bootstrap (1993)
  [2] Cohen, Statistical Power Analysis for the Behavioral Sciences (1988)
  [3] Rønnow et al., Science 345, 420 (2014)  — TTS metric
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : array-like
        Sample data.
    statistic : str
        "mean", "median", or "min".
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int, optional

    Returns
    -------
    point_estimate : float
    ci_lower : float
    ci_upper : float
    """
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)

    stat_fn = {"mean": np.mean, "median": np.median, "min": np.min}
    fn = stat_fn.get(statistic, np.mean)

    point = float(fn(data))

    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats[i] = fn(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point, ci_lower, ci_upper


def approximation_ratio(
    found_energy: float,
    optimal_energy: float,
) -> float:
    """
    Compute the approximation ratio r = E_found / E_optimal.

    For minimisation problems with negative optima:
      r = 1 means exact optimum found.
      r < 1 means found energy is better than optimal (shouldn't happen).
      r > 1 means found energy is worse.

    For zero or positive optima, returns 1.0 if found == optimal.
    """
    if abs(optimal_energy) < 1e-12:
        return 1.0 if abs(found_energy) < 1e-12 else float("inf")
    return found_energy / optimal_energy


def time_to_solution(
    t_single: float,
    p_success: float,
    target_probability: float = 0.99,
) -> float:
    """
    Time-to-solution metric (Rønnow et al., Science 2014).

    TTS = t_single × ⌈log(1 − p_target) / log(1 − p_success)⌉

    Parameters
    ----------
    t_single : float
        Time for a single run (seconds).
    p_success : float
        Probability of finding the ground state in one run.
    target_probability : float
        Desired overall success probability (default 0.99).

    Returns
    -------
    tts : float
        Estimated time-to-solution (seconds). Returns inf if p_success ≤ 0.
    """
    if p_success <= 0:
        return float("inf")
    if p_success >= 1.0:
        return t_single

    n_runs = np.ceil(
        np.log(1 - target_probability) / np.log(1 - p_success)
    )
    return float(t_single * n_runs)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size (pooled standard deviation).

    d = (μ₁ − μ₂) / s_pooled
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean_diff = np.mean(g1) - np.mean(g2)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)

    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if s_pooled < 1e-15:
        return 0.0

    return float(mean_diff / s_pooled)


def wilcoxon_signed_rank(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Returns (statistic, p_value). Uses normal approximation for n > 25.
    Implements the test directly to avoid scipy dependency.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x - y
    diff = diff[diff != 0]  # remove zero differences
    n = len(diff)

    if n == 0:
        return 0.0, 1.0

    abs_diff = np.abs(diff)
    ranks = np.argsort(np.argsort(abs_diff)) + 1.0  # simple ranking

    # Handle ties by averaging ranks
    for val in np.unique(abs_diff):
        mask = abs_diff == val
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])

    w_plus = np.sum(ranks[diff > 0])
    w_minus = np.sum(ranks[diff < 0])
    w = min(w_plus, w_minus)

    # Normal approximation (valid for n > ~10)
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

    if std_w < 1e-15:
        return float(w), 1.0

    z = (w - mean_w) / std_w
    # Two-tailed p-value via normal CDF approximation
    p_value = 2.0 * _norm_cdf(-abs(z))

    return float(w), float(p_value)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (approximation using error function)."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x: float) -> float:
    """
    Approximation of the error function (Abramowitz & Stegun 7.1.26).
    Maximum error: 1.5e-7.
    """
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
         - 0.284496736) * t + 0.254829592
    ) * t * np.exp(-x * x)
    return sign * y


def summary_table(
    results: Dict[str, Dict],
    exact_energy: float,
) -> str:
    """
    Generate a summary table comparing algorithm results.

    Parameters
    ----------
    results : dict
        Keys are algorithm names, values are dicts with keys:
          "energy", "time", "success_rate" (optional)
    exact_energy : float
        Known optimal energy for approximation ratio.

    Returns
    -------
    table : str
        Formatted table (plain text and LaTeX).
    """
    lines = []
    lines.append("=" * 75)
    lines.append(f"{'Algorithm':<20} {'Energy':>10} {'Ratio':>8} "
                 f"{'Time(s)':>10} {'Success%':>10}")
    lines.append("-" * 75)

    for name, r in results.items():
        e = r.get("energy", float("inf"))
        t = r.get("time", 0.0)
        sr = r.get("success_rate", None)
        ratio = approximation_ratio(e, exact_energy)

        sr_str = f"{sr*100:.1f}" if sr is not None else "N/A"
        lines.append(
            f"{name:<20} {e:>10.3f} {ratio:>8.3f} "
            f"{t:>10.3f} {sr_str:>10}"
        )

    lines.append("=" * 75)
    lines.append(f"Exact optimal energy: {exact_energy:.3f}")

    return "\n".join(lines)


def latex_table(
    results: Dict[str, Dict],
    exact_energy: float,
    caption: str = "Algorithm comparison",
) -> str:
    """Generate a LaTeX-formatted results table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Algorithm & Energy & Approx.\ Ratio & Time (s) & Success Rate \\",
        r"\midrule",
    ]

    for name, r in results.items():
        e = r.get("energy", float("inf"))
        t = r.get("time", 0.0)
        sr = r.get("success_rate", None)
        ratio = approximation_ratio(e, exact_energy)
        sr_str = f"{sr*100:.1f}\\%" if sr is not None else "N/A"
        lines.append(
            f"  {name} & {e:.3f} & {ratio:.3f} & {t:.3f} & {sr_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\label{{tab:results}}",
        r"\end{table}",
    ])

    return "\n".join(lines)
