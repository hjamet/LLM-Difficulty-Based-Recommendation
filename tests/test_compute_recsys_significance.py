"""
tests/test_compute_recsys_significance.py - Unit test suite for RecSys Statistical Significance Engine.

Tests:
1. Exact mathematical properties of compute_bootstrap_se:
   - Determinism given random seed
   - 95% Confidence Interval bounds (ci_low <= mean <= ci_high)
   - Degenerate cases: constant array, single element, empty array handling
2. Paired statistical significance tests (compute_paired_significance):
   - Wilcoxon signed-rank and paired t-test on known paired vectors
   - Zero-difference identity case
   - Input validation and length mismatch errors
3. Holm-Bonferroni FWER step-down correction (apply_holm_bonferroni):
   - Monotonicity of adjusted p-values (p_adj[k] >= p_adj[k-1] in sorted order)
   - Proper attribution of LaTeX markers: ^{***}, ^{**}, ^{*}, and ns
   - Invariant preservation of original index ordering
4. LaTeX formatting (format_metric_with_se_and_sig):
   - Value and SE decimal precision
   - Optional bolding and significance markers
"""

import os
import sys
import math
import pytest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dmkd_additions.scripts.compute_recsys_significance import (
    compute_bootstrap_se,
    compute_paired_significance,
    apply_holm_bonferroni,
    format_metric_with_se_and_sig,
    simulate_user_scores_from_summary,
    evaluate_table_significance,
    TABLE_8_DATA,
)


class TestComputeBootstrapSE:
    """Tests for compute_bootstrap_se."""

    def test_determinism_with_fixed_seed(self):
        scores = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.19])
        mean1, se1, low1, high1 = compute_bootstrap_se(scores, n_boot=500, seed=123)
        mean2, se2, low2, high2 = compute_bootstrap_se(scores, n_boot=500, seed=123)
        assert mean1 == mean2
        assert se1 == se2
        assert low1 == low2
        assert high1 == high2

    def test_ci_bounds_encompass_mean(self):
        rng = np.random.default_rng(42)
        scores = rng.normal(loc=0.5, scale=0.1, size=200)
        mean, se, ci_low, ci_high = compute_bootstrap_se(scores, n_boot=1000, seed=42)
        assert ci_low <= mean <= ci_high
        assert se > 0.0
        # For sample mean of normal, bootstrap SE should match analytical SE = std / sqrt(N)
        analytical_se = np.std(scores, ddof=1) / math.sqrt(len(scores))
        assert abs(se - analytical_se) < 0.01

    def test_constant_array_degenerate_case(self):
        scores = np.array([0.42, 0.42, 0.42, 0.42, 0.42])
        mean, se, ci_low, ci_high = compute_bootstrap_se(scores, n_boot=200, seed=42)
        assert mean == pytest.approx(0.42)
        assert se == 0.0
        assert ci_low == pytest.approx(0.42)
        assert ci_high == pytest.approx(0.42)

    def test_single_element_array(self):
        scores = np.array([0.75])
        mean, se, ci_low, ci_high = compute_bootstrap_se(scores, n_boot=100, seed=42)
        assert mean == pytest.approx(0.75)
        assert se == 0.0
        assert ci_low == pytest.approx(0.75)
        assert ci_high == pytest.approx(0.75)

    def test_empty_array_raises_value_error(self):
        with pytest.raises(ValueError, match="empty user_scores"):
            compute_bootstrap_se(np.array([]))

    def test_nan_array_raises_value_error(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_bootstrap_se(np.array([0.1, np.nan, 0.3]))


class TestComputePairedSignificance:
    """Tests for compute_paired_significance."""

    def test_strictly_superior_model(self):
        # Model consistently scores higher than baseline
        model = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.65, 0.75, 0.85, 0.7, 0.8])
        baseline = np.array([0.2, 0.3, 0.35, 0.4, 0.45, 0.3, 0.4, 0.5, 0.35, 0.45])
        res = compute_paired_significance(model, baseline)

        assert res["n_users"] == 10
        assert res["diff_mean"] > 0.0
        assert res["wilcoxon_p"] < 0.01
        assert res["ttest_p"] < 0.01

    def test_identical_models_zero_diff(self):
        scores = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        res = compute_paired_significance(scores, scores)
        assert res["diff_mean"] == 0.0
        assert res["wilcoxon_stat"] == 0.0
        assert res["wilcoxon_p"] == 1.0
        assert res["ttest_p"] == 1.0

    def test_length_mismatch_raises_error(self):
        m = np.array([0.1, 0.2, 0.3])
        b = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="Array length mismatch"):
            compute_paired_significance(m, b)

    def test_empty_array_raises_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_paired_significance(np.array([]), np.array([]))


class TestApplyHolmBonferroni:
    """Tests for apply_holm_bonferroni."""

    def test_monotonicity_of_adjusted_p_values(self):
        raw_p = [0.0001, 0.008, 0.02, 0.04, 0.15, 0.80]
        results = apply_holm_bonferroni(raw_p, alpha=0.05)

        assert len(results) == len(raw_p)
        # Check sorted by raw p
        sorted_res = sorted(results, key=lambda x: x["p_raw"])
        for i in range(1, len(sorted_res)):
            assert sorted_res[i]["p_adj"] >= sorted_res[i - 1]["p_adj"]

    def test_markers_attribution(self):
        raw_p = [0.00001, 0.002, 0.01, 0.25]
        results = apply_holm_bonferroni(raw_p, alpha=0.05)

        # 0.00001 * 4 = 0.00004 < 0.001 -> ^{***}
        assert results[0]["latex_marker"] == "^{***}"
        assert results[0]["rejected"] is True

        # 0.002 * 3 = 0.006 < 0.01 -> ^{**}
        assert results[1]["latex_marker"] == "^{**}"
        assert results[1]["rejected"] is True

        # 0.01 * 2 = 0.02 < 0.05 -> ^{*}
        assert results[2]["latex_marker"] == "^{*}"
        assert results[2]["rejected"] is True

        # 0.25 * 1 = 0.25 >= 0.05 -> ns
        assert results[3]["latex_marker"] == "ns"
        assert results[3]["rejected"] is False

    def test_preserves_original_indexing(self):
        raw_p = [0.50, 0.001, 0.05]
        results = apply_holm_bonferroni(raw_p)
        assert results[0]["index"] == 0
        assert results[1]["index"] == 1
        assert results[2]["index"] == 2
        assert results[1]["p_raw"] == 0.001

    def test_empty_list(self):
        assert apply_holm_bonferroni([]) == []


class TestFormatMetricWithSEAndSig:
    """Tests for format_metric_with_se_and_sig."""

    def test_standard_formatting(self):
        res = format_metric_with_se_and_sig(0.0626, 0.0030)
        assert res == "0.0626 \\pm 0.0030"

    def test_bold_formatting(self):
        res = format_metric_with_se_and_sig(0.0721, 0.0032, bold=True)
        assert res == "\\textbf{0.0721 \\pm 0.0032}"

    def test_significance_marker_asterisk(self):
        res = format_metric_with_se_and_sig(0.0721, 0.0032, marker="^{*}", bold=True)
        assert res == "\\textbf{0.0721 \\pm 0.0032}^{*}"

    def test_ns_marker(self):
        res = format_metric_with_se_and_sig(0.0659, 0.0030, marker="ns")
        assert res == "0.0659 \\pm 0.0030^{\\text{ns}}"


class TestIntegrationTable8:
    """End-to-end integration test with Table 8 evaluation."""

    def test_evaluate_table_significance_structure(self):
        # Subset with 1 dataset for fast testing
        sub_data = {"Zeeguu": TABLE_8_DATA["Zeeguu"]}
        user_counts = {"Zeeguu": 100}
        res = evaluate_table_significance(sub_data, user_counts=user_counts, n_boot=50, seed=42)

        assert "Zeeguu" in res["datasets"]
        zeeguu_models = res["datasets"]["Zeeguu"]["models"]
        assert "LightGCN ADA" in zeeguu_models
        assert "Recall@5" in zeeguu_models["LightGCN ADA"]
        stats_recall = zeeguu_models["LightGCN ADA"]["Recall@5"]
        assert stats_recall["ci95_low"] <= stats_recall["mean"] <= stats_recall["ci95_high"]
