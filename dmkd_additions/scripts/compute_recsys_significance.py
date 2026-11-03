"""
compute_recsys_significance.py - Statistical Significance Engine for Recommender Systems (DMKD Revision).

Performs rigorous statistical evaluation of recommendation models:
1. Deterministic analytical upper bounds on Standard Error via Bhatia-Davis (2000) inequality:
   SE_max = sqrt(mu * (1 - mu) / N), CI_95_max = [max(0, mu - 1.96*SE_max), min(1, mu + 1.96*SE_max)]
2. Real user-level statistical evaluation from CSV (--input-csv):
   - Non-parametric bootstrap standard errors (SE) and 95% Confidence Intervals.
   - Paired Wilcoxon signed-rank tests and paired Student's t-tests.
   - Holm-Bonferroni step-down family-wise error rate correction.
3. Utility export function export_notebook_metrics_to_user_csv for 1-line integration
   at the end of recommender_system/c_Recommendations.ipynb.
4. Export of results in JSON, CSV, and LaTeX snippet format matching Table 8 of the paper.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RecSys_Significance")

# Official Table 7 dataset sizes (number of users)
DATASET_USER_COUNTS: dict[str, int] = {
    "Zeeguu": 1574,
    "ml-100k": 942,
    "Goodreads": 39628,
    "Tomplay": 35028,
}

# Metrics reported in Table 8
METRICS_ORDER: list[str] = [
    "Recall@5",
    "Precision@5",
    "F1@5",
    "NDCG@5",
    "MRR@5",
    "MAP@5",
]

# Baseline Table 8 published values
TABLE_8_DATA: dict[str, dict[str, dict[str, float]]] = {
    "Zeeguu": {
        "ALS": {"Recall@5": 0.0626, "Precision@5": 0.0236, "F1@5": 0.0309, "NDCG@5": 0.0595, "MRR@5": 0.0774, "MAP@5": 0.0498},
        "BPR": {"Recall@5": 0.0182, "Precision@5": 0.0061, "F1@5": 0.0082, "NDCG@5": 0.0168, "MRR@5": 0.0191, "MAP@5": 0.0146},
        "LMF": {"Recall@5": 0.0251, "Precision@5": 0.0084, "F1@5": 0.0116, "NDCG@5": 0.0235, "MRR@5": 0.0291, "MAP@5": 0.0202},
        "KNN ADA": {"Recall@5": 0.0357, "Precision@5": 0.0122, "F1@5": 0.0155, "NDCG@5": 0.0290, "MRR@5": 0.0324, "MAP@5": 0.0232},
        "KNN BERT": {"Recall@5": 0.0170, "Precision@5": 0.0050, "F1@5": 0.0066, "NDCG@5": 0.0143, "MRR@5": 0.0159, "MAP@5": 0.0121},
        "LightGCN ADA": {"Recall@5": 0.0721, "Precision@5": 0.0274, "F1@5": 0.0352, "NDCG@5": 0.0666, "MRR@5": 0.0843, "MAP@5": 0.0550},
        "LightGCN Xavier ADA": {"Recall@5": 0.0630, "Precision@5": 0.0238, "F1@5": 0.0308, "NDCG@5": 0.0594, "MRR@5": 0.0759, "MAP@5": 0.0498},
        "LightGCN BERT": {"Recall@5": 0.0659, "Precision@5": 0.0249, "F1@5": 0.0323, "NDCG@5": 0.0609, "MRR@5": 0.0788, "MAP@5": 0.0501},
        "LightGCN Xavier BERT": {"Recall@5": 0.0622, "Precision@5": 0.0231, "F1@5": 0.0300, "NDCG@5": 0.0599, "MRR@5": 0.0765, "MAP@5": 0.0508},
    },
    "ml-100k": {
        "ALS": {"Recall@5": 0.0767, "Precision@5": 0.1285, "F1@5": 0.0806, "NDCG@5": 0.1414, "MRR@5": 0.2504, "MAP@5": 0.0850},
        "BPR": {"Recall@5": 0.0551, "Precision@5": 0.0975, "F1@5": 0.0580, "NDCG@5": 0.1084, "MRR@5": 0.1995, "MAP@5": 0.0634},
        "LMF": {"Recall@5": 0.0381, "Precision@5": 0.0635, "F1@5": 0.0400, "NDCG@5": 0.0704, "MRR@5": 0.1419, "MAP@5": 0.0379},
        "KNN ADA": {"Recall@5": 0.0188, "Precision@5": 0.0323, "F1@5": 0.0196, "NDCG@5": 0.0384, "MRR@5": 0.0781, "MAP@5": 0.0222},
        "KNN BERT": {"Recall@5": 0.0116, "Precision@5": 0.0189, "F1@5": 0.0117, "NDCG@5": 0.0234, "MRR@5": 0.0495, "MAP@5": 0.0135},
        "LightGCN ADA": {"Recall@5": 0.0903, "Precision@5": 0.1348, "F1@5": 0.0882, "NDCG@5": 0.1556, "MRR@5": 0.2682, "MAP@5": 0.0975},
        "LightGCN Xavier ADA": {"Recall@5": 0.0881, "Precision@5": 0.1346, "F1@5": 0.0869, "NDCG@5": 0.1548, "MRR@5": 0.2720, "MAP@5": 0.0958},
        "LightGCN BERT": {"Recall@5": 0.0855, "Precision@5": 0.1316, "F1@5": 0.0847, "NDCG@5": 0.1547, "MRR@5": 0.2761, "MAP@5": 0.0971},
        "LightGCN Xavier BERT": {"Recall@5": 0.0855, "Precision@5": 0.1323, "F1@5": 0.0847, "NDCG@5": 0.1508, "MRR@5": 0.2636, "MAP@5": 0.0922},
    },
    "Goodreads": {
        "ALS": {"Recall@5": 0.0634, "Precision@5": 0.0156, "F1@5": 0.0237, "NDCG@5": 0.0447, "MRR@5": 0.0419, "MAP@5": 0.0366},
        "BPR": {"Recall@5": 0.0447, "Precision@5": 0.0114, "F1@5": 0.0168, "NDCG@5": 0.0323, "MRR@5": 0.0323, "MAP@5": 0.0261},
        "LMF": {"Recall@5": 0.0432, "Precision@5": 0.0099, "F1@5": 0.0155, "NDCG@5": 0.0280, "MRR@5": 0.0245, "MAP@5": 0.0221},
        "KNN ADA": {"Recall@5": 0.0175, "Precision@5": 0.0050, "F1@5": 0.0070, "NDCG@5": 0.0127, "MRR@5": 0.0127, "MAP@5": 0.0101},
        "KNN BERT": {"Recall@5": 0.0100, "Precision@5": 0.0029, "F1@5": 0.0040, "NDCG@5": 0.0075, "MRR@5": 0.0078, "MAP@5": 0.0060},
        "LightGCN ADA": {"Recall@5": 0.0732, "Precision@5": 0.0183, "F1@5": 0.0275, "NDCG@5": 0.0513, "MRR@5": 0.0481, "MAP@5": 0.0415},
        "LightGCN Xavier ADA": {"Recall@5": 0.0603, "Precision@5": 0.0155, "F1@5": 0.0231, "NDCG@5": 0.0432, "MRR@5": 0.0415, "MAP@5": 0.0352},
        "LightGCN BERT": {"Recall@5": 0.0673, "Precision@5": 0.0170, "F1@5": 0.0254, "NDCG@5": 0.0473, "MRR@5": 0.0449, "MAP@5": 0.0382},
        "LightGCN Xavier BERT": {"Recall@5": 0.0570, "Precision@5": 0.0149, "F1@5": 0.0220, "NDCG@5": 0.0416, "MRR@5": 0.0405, "MAP@5": 0.0340},
    },
    "Tomplay": {
        "ALS": {"Recall@5": 0.0875, "Precision@5": 0.0598, "F1@5": 0.0650, "NDCG@5": 0.0838, "MRR@5": 0.1408, "MAP@5": 0.0528},
        "BPR": {"Recall@5": 0.0295, "Precision@5": 0.0236, "F1@5": 0.0246, "NDCG@5": 0.0306, "MRR@5": 0.0583, "MAP@5": 0.0178},
        "LMF": {"Recall@5": 0.0327, "Precision@5": 0.0249, "F1@5": 0.0263, "NDCG@5": 0.0323, "MRR@5": 0.0598, "MAP@5": 0.0187},
        "KNN ADA": {"Recall@5": 0.0269, "Precision@5": 0.0170, "F1@5": 0.0188, "NDCG@5": 0.0242, "MRR@5": 0.0379, "MAP@5": 0.0156},
        "KNN BERT": {"Recall@5": 0.0323, "Precision@5": 0.0209, "F1@5": 0.0229, "NDCG@5": 0.0314, "MRR@5": 0.0514, "MAP@5": 0.0210},
        "LightGCN ADA": {"Recall@5": 0.0958, "Precision@5": 0.0646, "F1@5": 0.0706, "NDCG@5": 0.0924, "MRR@5": 0.1541, "MAP@5": 0.0595},
        "LightGCN Xavier ADA": {"Recall@5": 0.0943, "Precision@5": 0.0643, "F1@5": 0.0699, "NDCG@5": 0.0910, "MRR@5": 0.1528, "MAP@5": 0.0581},
        "LightGCN BERT": {"Recall@5": 0.0984, "Precision@5": 0.0660, "F1@5": 0.0722, "NDCG@5": 0.0947, "MRR@5": 0.1582, "MAP@5": 0.0608},
        "LightGCN Xavier BERT": {"Recall@5": 0.0966, "Precision@5": 0.0656, "F1@5": 0.0715, "NDCG@5": 0.0936, "MRR@5": 0.1566, "MAP@5": 0.0601},
    },
}

MODELS_ORDER: list[str] = [
    "ALS",
    "BPR",
    "LMF",
    "KNN ADA",
    "KNN BERT",
    "LightGCN ADA",
    "LightGCN Xavier ADA",
    "LightGCN BERT",
    "LightGCN Xavier BERT",
]


def compute_bhatia_davis_se_bound(
    mean_score: float,
    n_users: int,
) -> tuple[float, tuple[float, float]]:
    """
    Compute deterministic analytical upper bound on Standard Error and 95% CI
    via the Bhatia-Davis (2000) inequality for bounded random variables in [0, 1].

    For any metric X_u in [0, 1] with mean mu in [0, 1] over N users:
        Var(X_u) <= mu * (1 - mu)
        SE_max = sqrt(mu * (1 - mu) / N)
        CI_95_max = [max(0.0, mu - 1.96 * SE_max), min(1.0, mu + 1.96 * SE_max)]

    Args:
        mean_score: Metric mean score in [0.0, 1.0].
        n_users: Number of users N (must be >= 1).

    Returns:
        tuple[float, tuple[float, float]]:
            - se_max: Maximum possible standard error sqrt(mu * (1 - mu) / N).
            - (ci_low, ci_high): 95% Confidence Interval upper bounds clipped to [0, 1].

    Raises:
        ValueError: If n_users <= 0 or mean_score is not in [0.0, 1.0] or not finite.
    """
    if n_users <= 0:
        raise ValueError(f"n_users must be positive, got {n_users}")
    if not np.isfinite(mean_score):
        raise ValueError(f"mean_score must be a finite number, got {mean_score}")
    if mean_score < 0.0 or mean_score > 1.0:
        raise ValueError(f"mean_score must be in [0, 1], got {mean_score}")

    if mean_score == 0.0 or mean_score == 1.0:
        return 0.0, (float(mean_score), float(mean_score))

    se_max = float(np.sqrt(mean_score * (1.0 - mean_score) / n_users))
    ci_low = float(max(0.0, mean_score - 1.96 * se_max))
    ci_high = float(min(1.0, mean_score + 1.96 * se_max))
    return se_max, (ci_low, ci_high)


def compute_bootstrap_se(
    user_scores: Sequence[float] | np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Compute mean, bootstrap standard error (SE), and 95% confidence interval for user-level scores.

    Args:
        user_scores: 1D array or sequence of per-user metric scores (e.g., NDCG@5, Recall@5).
        n_boot: Number of bootstrap iterations (default: 1000).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        tuple[float, float, float, float]:
            - mean: Sample mean of the scores.
            - se: Bootstrap standard error (std of bootstrap means).
            - ci_low: 2.5th percentile of bootstrap distribution.
            - ci_high: 97.5th percentile of bootstrap distribution.

    Raises:
        ValueError: If user_scores is empty or contains non-finite values.
    """
    arr = np.asarray(user_scores, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute bootstrap SE on empty user_scores.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("user_scores contains NaN or Inf values.")

    mean_val = float(np.mean(arr))
    n = arr.size

    # Handle edge cases: single element or constant array
    if n == 1 or np.all(arr == arr[0]):
        return mean_val, 0.0, mean_val, mean_val

    rng = np.random.default_rng(seed)
    # Generate bootstrap samples: shape (n_boot, n)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(arr[indices], axis=1)

    se = float(np.std(boot_means, ddof=1))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return mean_val, se, ci_low, ci_high


def compute_paired_significance(
    scores_model: Sequence[float] | np.ndarray,
    scores_baseline: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """
    Compute paired non-parametric Wilcoxon signed-rank test and paired Student's t-test.

    Args:
        scores_model: Array of per-user scores for the candidate/best model.
        scores_baseline: Array of per-user scores for the baseline model.

    Returns:
        dict[str, Any] containing:
            - n_users: Number of paired observations.
            - diff_mean: Mean difference (model - baseline).
            - wilcoxon_stat: Wilcoxon signed-rank test statistic W.
            - wilcoxon_p: Two-sided p-value for Wilcoxon test.
            - ttest_stat: Paired t-test statistic.
            - ttest_p: Two-sided p-value for paired t-test.

    Raises:
        ValueError: If array lengths mismatch or are empty.
    """
    m_arr = np.asarray(scores_model, dtype=float)
    b_arr = np.asarray(scores_baseline, dtype=float)

    if m_arr.size == 0 or b_arr.size == 0:
        raise ValueError("Input score arrays cannot be empty.")
    if m_arr.size != b_arr.size:
        raise ValueError(f"Array length mismatch: {m_arr.size} vs {b_arr.size}")
    if not np.all(np.isfinite(m_arr)) or not np.all(np.isfinite(b_arr)):
        raise ValueError("Input arrays contain NaN or Inf values.")

    n_users = int(m_arr.size)
    diff = m_arr - b_arr
    diff_mean = float(np.mean(diff))

    # All differences are strictly identical
    if np.all(diff == 0.0):
        return {
            "n_users": n_users,
            "diff_mean": 0.0,
            "wilcoxon_stat": 0.0,
            "wilcoxon_p": 1.0,
            "ttest_stat": 0.0,
            "ttest_p": 1.0,
        }

    # Paired Wilcoxon signed-rank test
    try:
        w_res = stats.wilcoxon(m_arr, b_arr, zero_method="wilcox", alternative="two-sided")
        w_stat = float(w_res.statistic)
        w_p = float(w_res.pvalue)
    except Exception as e:
        logger.warning("Wilcoxon calculation encountered exception: %s", e)
        w_stat, w_p = 0.0, 1.0

    # Paired Student t-test
    try:
        t_res = stats.ttest_rel(m_arr, b_arr)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)
    except Exception as e:
        logger.warning("Paired t-test encountered exception: %s", e)
        t_stat, t_p = 0.0, 1.0

    return {
        "n_users": n_users,
        "diff_mean": diff_mean,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "ttest_stat": t_stat,
        "ttest_p": t_p,
    }


def apply_holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Apply Holm-Bonferroni step-down correction for multiple hypothesis testing.

    Ensures strong control of the family-wise error rate (FWER) under arbitrary dependence.
    Assigns standard publication LaTeX markers:
        - p_adj < 0.001 -> '^{***}'
        - p_adj < 0.01  -> '^{**}'
        - p_adj < 0.05  -> '^{*}'
        - otherwise     -> 'ns'

    Args:
        p_values: Sequence of raw p-values.
        alpha: Family-wise significance level (default: 0.05).

    Returns:
        list[dict[str, Any]]: Ordered list corresponding to the original p_values sequence.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Pair with original index and sort by p-value ascending
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * m
    running_max = 0.0

    for rank_idx, (orig_idx, p_val) in enumerate(sorted_pairs):
        k = rank_idx + 1  # 1-indexed
        raw_p = float(p_val)
        current_adj = min(1.0, (m - k + 1) * raw_p)
        running_max = max(running_max, current_adj)
        adj_p = min(1.0, running_max)
        adjusted[rank_idx] = (orig_idx, raw_p, adj_p, k)

    results: list[dict[str, Any]] = [{} for _ in range(m)]  # type: ignore
    for orig_idx, raw_p, adj_p, rank in adjusted:
        rejected = bool(adj_p < alpha)

        if adj_p < 0.001:
            marker = "^{***}"
        elif adj_p < 0.01:
            marker = "^{**}"
        elif adj_p < 0.05:
            marker = "^{*}"
        else:
            marker = "ns"

        results[orig_idx] = {
            "index": orig_idx,
            "p_raw": raw_p,
            "p_adj": adj_p,
            "rejected": rejected,
            "rank": rank,
            "latex_marker": marker,
        }

    return results


def format_metric_with_se_and_sig(
    mean: float,
    se: float,
    marker: str = "",
    bold: bool = False,
    decimals: int = 4,
) -> str:
    r"""
    Format metric as $\mu \pm \text{SE}$ with optional LaTeX bolding and significance marker.
    """
    val_str = f"{mean:.{decimals}f}"
    se_str = f"{se:.{decimals}f}"

    if bold:
        core = f"\\textbf{{{val_str} \\pm {se_str}}}"
    else:
        core = f"{val_str} \\pm {se_str}"

    if marker and marker != "ns":
        return f"{core}{marker}"
    elif marker == "ns":
        return f"{core}^{{\\text{{ns}}}}"
    return core


def compute_bhatia_davis_bounds_table(
    table_data: dict[str, dict[str, dict[str, float]]] = TABLE_8_DATA,
    user_counts: dict[str, int] = DATASET_USER_COUNTS,
) -> dict[str, Any]:
    """
    Calculate deterministic Bhatia-Davis analytical upper bounds on SE and 95% CI
    for published Table 8 metrics across all datasets and models.

    Because user-level metrics X_u in [0, 1], the population variance is bounded by:
        Var(X_u) <= mu * (1 - mu)   [Bhatia-Davis, 2000]
    giving deterministic SE upper bound:
        SE_max = sqrt(mu * (1 - mu) / N_users)
    """
    results: dict[str, Any] = {
        "mode": "bhatia_davis_analytical_upper_bound",
        "description": "Deterministic Bhatia-Davis analytical upper bounds on standard error (SE_max = sqrt(mu*(1-mu)/N)) and 95% CI.",
        "datasets": {},
        "summary": {
            "datasets_analyzed": list(table_data.keys()),
            "inequality": "Bhatia-Davis (2000): Var(X) <= mu * (1 - mu)",
        },
    }

    for dataset_name, models_dict in table_data.items():
        n_users = user_counts.get(dataset_name, 1000)
        ds_results: dict[str, Any] = {"n_users": n_users, "models": {}}

        for model_name, metrics in models_dict.items():
            ds_results["models"][model_name] = {}
            for metric_name in METRICS_ORDER:
                mean_val = metrics.get(metric_name, 0.0)
                se_max, (ci_low, ci_high) = compute_bhatia_davis_se_bound(mean_val, n_users)
                ds_results["models"][model_name][metric_name] = {
                    "mean": mean_val,
                    "se_max": se_max,
                    "se": se_max,  # alias
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bound_type": "bhatia_davis_upper_bound",
                }

        results["datasets"][dataset_name] = ds_results

    return results


def evaluate_table_significance(
    table_data: dict[str, dict[str, dict[str, float]]] = TABLE_8_DATA,
    user_counts: dict[str, int] = DATASET_USER_COUNTS,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Deterministic evaluation for Table 8 data using Bhatia-Davis analytical bounds.
    Maintains compatibility with test suites and callers without synthetic score simulation.
    """
    return compute_bhatia_davis_bounds_table(table_data=table_data, user_counts=user_counts)


def evaluate_from_user_csv(
    df_user: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Perform rigorous user-level statistical evaluation from a real per-user evaluation DataFrame.

    Expects columns:
        - 'dataset': dataset identifier (e.g., 'Zeeguu', 'ml-100k')
        - 'model': model name (e.g., 'LightGCN ADA', 'LightGCN Xavier ADA')
        - 'user_id': user identifier for pairing
        - metric columns: e.g., 'Recall@5', 'Precision@5', 'F1@5', 'NDCG@5', 'MRR@5', 'MAP@5'

    Computes:
        1. Bootstrap SE and 95% CI per model and metric.
        2. Paired Wilcoxon signed-rank and paired t-tests between models aligned on user_id.
        3. Holm-Bonferroni step-down correction on paired tests.
    """
    required_cols = {"dataset", "model", "user_id"}
    missing = required_cols - set(df_user.columns)
    if missing:
        raise ValueError(f"df_user is missing required columns: {sorted(missing)}")

    metric_cols = [c for c in METRICS_ORDER if c in df_user.columns]
    if not metric_cols:
        metric_cols = [c for c in df_user.columns if c not in required_cols and pd.api.types.is_numeric_dtype(df_user[c])]
    if not metric_cols:
        raise ValueError("No numeric metric columns found in df_user.")

    results: dict[str, Any] = {
        "mode": "user_level_evaluation",
        "datasets": {},
        "summary": {
            "n_datasets": int(df_user["dataset"].nunique()),
            "total_rows": int(len(df_user)),
            "metric_columns": metric_cols,
        },
    }

    for ds_name, df_ds in df_user.groupby("dataset"):
        ds_name_str = str(ds_name)
        ds_results: dict[str, Any] = {"models": {}, "comparisons": {}}

        models = list(df_ds["model"].unique())
        for model_name in models:
            df_m = df_ds[df_ds["model"] == model_name]
            ds_results["models"][model_name] = {}
            for metric in metric_cols:
                scores = df_m[metric].dropna().values.astype(float)
                if len(scores) == 0:
                    continue
                mean_val, se, ci_l, ci_h = compute_bootstrap_se(scores, n_boot=n_boot, seed=seed)
                ds_results["models"][model_name][metric] = {
                    "mean": mean_val,
                    "se": se,
                    "ci95_low": ci_l,
                    "ci95_high": ci_h,
                    "n_users": int(len(scores)),
                }

        # Comparisons plan
        comparisons_plan = [
            ("LightGCN ADA", "LightGCN Xavier ADA"),
            ("LightGCN BERT", "LightGCN Xavier BERT"),
            ("LightGCN ADA", "ALS"),
            ("LightGCN BERT", "ALS"),
        ]
        active_pairs = [(c, b) for c, b in comparisons_plan if c in models and b in models]
        if not active_pairs and len(models) >= 2:
            # Pair each subsequent model against the baseline model (index 0)
            active_pairs = [(models[i], models[0]) for i in range(1, len(models))]

        comp_results: dict[str, Any] = {}
        for cand, base in active_pairs:
            pair_key = f"{cand} vs {base}"
            comp_results[pair_key] = {}
            df_cand = df_ds[df_ds["model"] == cand]
            df_base = df_ds[df_ds["model"] == base]

            raw_p_values = []
            valid_metrics = []
            stats_list = []

            for metric in metric_cols:
                m_cand = df_cand[["user_id", metric]].dropna()
                m_base = df_base[["user_id", metric]].dropna()
                merged = pd.merge(m_cand, m_base, on="user_id", suffixes=("_cand", "_base"))
                if len(merged) < 2:
                    continue
                s_cand = merged[f"{metric}_cand"].values.astype(float)
                s_base = merged[f"{metric}_base"].values.astype(float)
                sig = compute_paired_significance(s_cand, s_base)
                raw_p_values.append(sig["wilcoxon_p"])
                valid_metrics.append(metric)
                stats_list.append(sig)

            if raw_p_values:
                adj_results = apply_holm_bonferroni(raw_p_values, alpha=alpha)
                for m_name, stat_item, adj_item in zip(valid_metrics, stats_list, adj_results):
                    comp_results[pair_key][m_name] = {
                        **stat_item,
                        **adj_item,
                    }

        ds_results["comparisons"] = comp_results
        results["datasets"][ds_name_str] = ds_results

    return results


def export_notebook_metrics_to_user_csv(
    evaluation_dict: dict[str, Any],
    output_csv_path: str = "",
    dataset: str = "Zeeguu",
    model: str = "LightGCN ADA",
    user_ids: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """
    Utility function to export per-user metric lists from recommender_system/c_Recommendations.ipynb
    (Cell 13 output: 'ndcgs', 'rrs', 'aps', 'recalls', 'precisions', 'f1s')
    to the standardized per-user CSV format expected by evaluate_from_user_csv and --input-csv.

    Args:
        evaluation_dict: Dictionary returned by Cell 13 of c_Recommendations.ipynb.
        output_csv_path: Destination CSV file path. If empty, returns DataFrame without writing.
        dataset: Dataset label (default: 'Zeeguu').
        model: Model label (default: 'LightGCN ADA').
        user_ids: Optional sequence of user IDs. If None, 0..N-1 is used.

    Returns:
        pd.DataFrame: Exported DataFrame with columns:
            dataset, model, user_id, Recall@5, Precision@5, F1@5, NDCG@5, MRR@5, MAP@5.
    """
    # Check if evaluation_dict is multi-model: {model_name: eval_dict}
    if evaluation_dict:
        first_val = next(iter(evaluation_dict.values()))
        if isinstance(first_val, dict) and any(k in first_val for k in ("ndcgs", "recalls", "rrs")):
            frames = []
            for m_name, sub_dict in evaluation_dict.items():
                f = export_notebook_metrics_to_user_csv(
                    sub_dict, output_csv_path="", dataset=dataset, model=m_name, user_ids=user_ids
                )
                frames.append(f)
            merged_df = pd.concat(frames, ignore_index=True)
            if output_csv_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
                merged_df.to_csv(output_csv_path, index=False)
                logger.info("Exported multi-model user CSV to %s (%d rows)", output_csv_path, len(merged_df))
            return merged_df

    key_mapping = {
        "Recall@5": "recalls",
        "Precision@5": "precisions",
        "F1@5": "f1s",
        "NDCG@5": "ndcgs",
        "MRR@5": "rrs",
        "MAP@5": "aps",
    }

    extracted_cols = {}
    n_users = 0
    for col_name, dict_key in key_mapping.items():
        if dict_key in evaluation_dict and evaluation_dict[dict_key] is not None:
            vals = list(evaluation_dict[dict_key])
            extracted_cols[col_name] = vals
            n_users = max(n_users, len(vals))

    if n_users == 0:
        raise ValueError("evaluation_dict does not contain any recognizable per-user metric lists (recalls, ndcgs, etc.).")

    # Align metrics to common length
    for col_name in key_mapping:
        if col_name not in extracted_cols:
            extracted_cols[col_name] = [0.0] * n_users
        elif len(extracted_cols[col_name]) < n_users:
            extracted_cols[col_name].extend([0.0] * (n_users - len(extracted_cols[col_name])))

    if user_ids is None:
        user_ids = list(range(n_users))
    elif len(user_ids) < n_users:
        user_ids = list(user_ids) + list(range(len(user_ids), n_users))

    data = {
        "dataset": [dataset] * n_users,
        "model": [model] * n_users,
        "user_id": list(user_ids)[:n_users],
        **extracted_cols,
    }
    df = pd.DataFrame(data)

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logger.info("Exported notebook metrics to user CSV: %s (%d rows)", output_csv_path, len(df))

    return df


def export_results_to_csv(
    evaluation_results: dict[str, Any],
    csv_path: str,
) -> pd.DataFrame:
    """
    Export flattened statistical results table to CSV.
    Supports both Bhatia-Davis analytical upper bounds and user-level evaluation modes.
    """
    mode = evaluation_results.get("mode", "bhatia_davis_analytical_upper_bound")
    rows = []

    for ds_name, ds_info in evaluation_results["datasets"].items():
        n_users = ds_info.get("n_users", 0)
        models_info = ds_info.get("models", {})
        comparisons = ds_info.get("comparisons", {})

        for model_name, metrics_dict in models_info.items():
            for metric_name, m_stats in metrics_dict.items():
                if mode == "bhatia_davis_analytical_upper_bound":
                    rows.append({
                        "dataset": ds_name,
                        "model": model_name,
                        "metric": metric_name,
                        "mean": m_stats["mean"],
                        "n_users": n_users,
                        "se_max_bhatia_davis": m_stats.get("se_max", m_stats.get("se", 0.0)),
                        "ci95_low": m_stats["ci95_low"],
                        "ci95_high": m_stats["ci95_high"],
                        "bound_type": m_stats.get("bound_type", "bhatia_davis_upper_bound"),
                    })
                else:
                    # User-level evaluation mode
                    marker = ""
                    p_adj = np.nan
                    p_raw = np.nan
                    if model_name == "LightGCN ADA":
                        comp = comparisons.get("LightGCN ADA vs LightGCN Xavier ADA", {}).get(metric_name, {})
                        marker = comp.get("latex_marker", "")
                        p_adj = comp.get("p_adj", np.nan)
                        p_raw = comp.get("p_raw", np.nan)
                    elif model_name == "LightGCN BERT":
                        comp = comparisons.get("LightGCN BERT vs LightGCN Xavier BERT", {}).get(metric_name, {})
                        marker = comp.get("latex_marker", "")
                        p_adj = comp.get("p_adj", np.nan)
                        p_raw = comp.get("p_raw", np.nan)

                    rows.append({
                        "dataset": ds_name,
                        "model": model_name,
                        "metric": metric_name,
                        "mean": m_stats["mean"],
                        "se": m_stats["se"],
                        "ci95_low": m_stats["ci95_low"],
                        "ci95_high": m_stats["ci95_high"],
                        "p_raw_vs_xavier": p_raw,
                        "p_adj_holm": p_adj,
                        "significance_marker": marker,
                    })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Exported CSV results to %s", csv_path)
    return df


def build_latex_table_snippet(
    table_data: dict[str, dict[str, dict[str, float]]],
    evaluation_results: dict[str, Any],
    decimals: int = 4,
) -> str:
    r"""
    Generate LaTeX snippet for Table 8 matching publication style.
    """
    mode = evaluation_results.get("mode", "bhatia_davis_analytical_upper_bound")
    lines: list[str] = []

    if mode == "bhatia_davis_analytical_upper_bound":
        lines.append("% Table 8: Evaluation results with deterministic Bhatia-Davis analytical upper bounds")
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append(r"\caption{Evaluation results ($\mu \pm \text{SE}_{\max}$ over $N$ users, with deterministic Bhatia--Davis analytical upper bound $\text{SE}_{\max} = \sqrt{\hat{\mu}(1 - \hat{\mu})/N}$).}")
    else:
        lines.append("% Statistical Evaluation of Recommendation Models with Bootstrap SE & Holm-Bonferroni")
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append(r"\caption{Evaluation results ($\mu \pm \text{SE}$ over $N$ users, with bootstrap $B=1000$). Asterisks indicate statistical significance over the respective Xavier baseline via paired Wilcoxon signed-rank test with Holm-Bonferroni FWER control ($^{*}p_{\text{adj}}<0.05$, $^{**}p_{\text{adj}}<0.01$, $^{***}p_{\text{adj}}<0.001$).}")

    lines.append("\\label{tab:evaluation_results_significance}")
    lines.append("\\tiny")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l l cccccc}")
    lines.append("\\toprule")
    lines.append("\\rotatebox{45}{\\textbf{Dataset}} & \\rotatebox{45}{\\textbf{Model}} & "
                 "\\rotatebox{45}{\\textbf{Recall@5}} & \\rotatebox{45}{\\textbf{Precision@5}} & "
                 "\\rotatebox{45}{\\textbf{F1@5}} & \\rotatebox{45}{\\textbf{NDCG@5}} & "
                 "\\rotatebox{45}{\\textbf{MRR@5}} & \\rotatebox{45}{\\textbf{MAP@5}} \\\\")
    lines.append("\\midrule")

    dataset_names = list(table_data.keys())
    for d_idx, ds_name in enumerate(dataset_names):
        models_dict = table_data[ds_name]
        ds_stats = evaluation_results["datasets"].get(ds_name, {})
        models_stats = ds_stats.get("models", {})
        comparisons = ds_stats.get("comparisons", {})

        n_models = len(models_dict)
        lines.append(f"% Results for {ds_name}")

        best_vals = {}
        for m_name in METRICS_ORDER:
            vals = [models_dict[m].get(m_name, 0.0) for m in models_dict]
            best_vals[m_name] = max(vals) if vals else 0.0

        for m_idx, model_name in enumerate(models_dict.keys()):
            row_prefix = f"\\multirow{{{n_models}}}{{*}}{{\\rotatebox{{45}}{{{ds_name}}}}} & " if m_idx == 0 else " & "
            formatted_cells = []

            for met_name in METRICS_ORDER:
                mean_val = models_dict[model_name].get(met_name, 0.0)
                m_info = models_stats.get(model_name, {}).get(met_name, {})
                se_val = m_info.get("se_max", m_info.get("se", 0.0))
                is_best = (abs(mean_val - best_vals[met_name]) < 1e-6)

                marker = ""
                if mode != "bhatia_davis_analytical_upper_bound":
                    if model_name == "LightGCN ADA":
                        comp_ada = comparisons.get("LightGCN ADA vs LightGCN Xavier ADA", {}).get(met_name, {})
                        marker = comp_ada.get("latex_marker", "")
                    elif model_name == "LightGCN BERT":
                        comp_bert = comparisons.get("LightGCN BERT vs LightGCN Xavier BERT", {}).get(met_name, {})
                        marker = comp_bert.get("latex_marker", "")

                formatted_cell = format_metric_with_se_and_sig(
                    mean_val, se_val, marker=marker, bold=is_best, decimals=decimals
                )
                formatted_cells.append(formatted_cell)

            line_content = f"{row_prefix}{model_name:<20} & " + " & ".join(formatted_cells) + " \\\\"
            lines.append(line_content)

        if d_idx < len(dataset_names) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Statistical Significance or Bhatia-Davis Analytical Bounds for Recommender Systems (DMKD Revision)."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to real per-user evaluation CSV containing columns: user_id, model, dataset, and metrics.",
    )
    parser.add_argument(
        "--analytical-bounds",
        action="store_true",
        default=False,
        help="Compute deterministic Bhatia-Davis analytical SE upper bounds on Table 8 data.",
    )
    parser.add_argument(
        "--export-latex",
        type=str,
        default=None,
        help="Optional output path for LaTeX table snippet.",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Output path for JSON results.",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Output path for CSV results.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for SE computation when evaluating per-user CSV (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Initializing RecSys Statistical Significance Engine...")

    if args.input_csv and os.path.exists(args.input_csv):
        logger.info("Operating in USER-LEVEL EVALUATION mode with real CSV: %s", args.input_csv)
        df_user = pd.read_csv(args.input_csv)
        logger.info("Loaded CSV with %d rows and columns: %s", len(df_user), list(df_user.columns))

        json_out = args.export_json or "dmkd_additions/results/recsys_significance_results.json"
        csv_out = args.export_csv or "dmkd_additions/results/recsys_significance_results.csv"

        evaluation_results = evaluate_from_user_csv(
            df_user=df_user,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    else:
        logger.info("Operating in BHATIA-DAVIS DETERMINISTIC ANALYTICAL BOUNDS mode...")
        json_out = args.export_json or "dmkd_additions/results/recsys_bhatia_davis_bounds.json"
        csv_out = args.export_csv or "dmkd_additions/results/recsys_bhatia_davis_bounds.csv"

        evaluation_results = compute_bhatia_davis_bounds_table(
            table_data=TABLE_8_DATA,
            user_counts=DATASET_USER_COUNTS,
        )

    os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    logger.info("Saved JSON results to %s", json_out)

    export_results_to_csv(evaluation_results, csv_out)

    if args.export_latex:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_latex)), exist_ok=True)
        latex_snippet = build_latex_table_snippet(TABLE_8_DATA, evaluation_results)
        with open(args.export_latex, "w", encoding="utf-8") as f:
            f.write(latex_snippet)
        logger.info("Saved LaTeX table snippet to %s", args.export_latex)

    print("\n" + "=" * 60)
    print("RecSys Significance Computation Completed Successfully!")
    print(f"Mode:  {evaluation_results.get('mode')}")
    print(f"JSON:  {json_out}")
    print(f"CSV:   {csv_out}")
    if args.export_latex:
        print(f"LaTeX: {args.export_latex}")
    print("=" * 60)


if __name__ == "__main__":
    main()
