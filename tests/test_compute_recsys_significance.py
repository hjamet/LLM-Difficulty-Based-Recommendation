"""
tests/test_compute_recsys_significance.py - Unit test suite for RecSys Statistical Significance Engine.

Tests:
1. Exact mathematical properties of compute_bhatia_davis_se_bound:
   - Exact analytical formula: SE_max = sqrt(mu * (1 - mu) / N)
   - Boundary values at mu = 0 and mu = 1 (SE_max = 0, degenerate CI)
   - Monotonic decrease in cohort size N
   - Truncation of 95% Confidence Interval in [0, 1]
   - Input validation (mean not in [0, 1], non-positive N)
2. Real user-level statistical evaluation from CSV (evaluate_from_user_csv):
   - Preservation and use of real per-user scores
   - Computation of empirical means, bootstrap standard errors, and 95% CIs
   - Paired Wilcoxon signed-rank and paired t-tests with Holm-Bonferroni FWER control
   - Validation of required columns and error handling
3. Cell 13 notebook metrics export (export_notebook_metrics_to_user_csv):
   - Conforms to exact output format of c_Recommendations.ipynb Cell 13
   - Produces correct schema: dataset, model, user_id, Recall@5, Precision@5, F1@5, NDCG@5, MRR@5, MAP@5
   - Multi-model nested dictionary export support
4. Exact mathematical properties of compute_bootstrap_se:
   - Determinism given random seed
   - 95% Confidence Interval bounds (ci_low <= mean <= ci_high)
   - Degenerate cases: constant array, single element, empty array handling
5. Paired statistical significance tests (compute_paired_significance):
   - Wilcoxon signed-rank and paired t-test on known paired vectors
   - Zero-difference identity case
   - Input validation and length mismatch errors
6. Holm-Bonferroni FWER step-down correction (apply_holm_bonferroni):
   - Monotonicity of adjusted p-values (p_adj[k] >= p_adj[k-1] in sorted order)
   - Proper attribution of LaTeX markers: ^{***}, ^{**}, ^{*}, and ns
   - Invariant preservation of original index ordering
7. LaTeX formatting (format_metric_with_se_and_sig):
   - Value and SE decimal precision
   - Optional bolding and significance markers
"""

import os
import sys
import math
import pytest
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dmkd_additions.scripts.compute_recsys_significance import (
    compute_bhatia_davis_se_bound,
    compute_bhatia_davis_bounds_table,
    evaluate_from_user_csv,
    export_notebook_metrics_to_user_csv,
    compute_bootstrap_se,
    compute_paired_significance,
    apply_holm_bonferroni,
    format_metric_with_se_and_sig,
    evaluate_table_significance,
    TABLE_8_DATA,
    DATASET_USER_COUNTS,
)


class TestComputeBhatiaDavisSEBound:
    """Tests for compute_bhatia_davis_se_bound."""

    def test_exact_value_formula(self):
        # mu = 0.5, N = 100 -> SE_max = sqrt(0.25 / 100) = 0.05
        se_max, (ci_low, ci_high) = compute_bhatia_davis_se_bound(0.5, 100)
        assert se_max == pytest.approx(0.05, abs=1e-9)
        assert ci_low == pytest.approx(0.5 - 1.96 * 0.05, abs=1e-5)
        assert ci_high == pytest.approx(0.5 + 1.96 * 0.05, abs=1e-5)

    def test_boundary_zero_and_one(self):
        # At boundary mu = 0.0, variance is 0, SE_max = 0, CI = [0, 0]
        se_zero, (low_zero, high_zero) = compute_bhatia_davis_se_bound(0.0, 1000)
        assert se_zero == 0.0
        assert low_zero == 0.0
        assert high_zero == 0.0

        # At boundary mu = 1.0, variance is 0, SE_max = 0, CI = [1, 1]
        se_one, (low_one, high_one) = compute_bhatia_davis_se_bound(1.0, 1000)
        assert se_one == 0.0
        assert low_one == 1.0
        assert high_one == 1.0

    def test_monotonicity_in_cohort_size_n(self):
        # As N increases for fixed mu, SE_max must strictly decrease
        mu = 0.0721
        se_100, _ = compute_bhatia_davis_se_bound(mu, 100)
        se_1000, _ = compute_bhatia_davis_se_bound(mu, 1000)
        se_10000, _ = compute_bhatia_davis_se_bound(mu, 10000)

        assert se_100 > se_1000 > se_10000
        # Ratio should scale exactly as 1 / sqrt(10)
        assert se_1000 == pytest.approx(se_100 / math.sqrt(10), rel=1e-5)

    def test_ci_clipping_to_unit_interval(self):
        # Small mu and small N: mu - 1.96 * SE_max < 0 -> must clip to 0.0
        mu = 0.01
        n = 10
        se_max, (ci_low, ci_high) = compute_bhatia_davis_se_bound(mu, n)
        assert ci_low == 0.0
        assert ci_high <= 1.0
        assert ci_high > mu

        # Large mu: mu + 1.96 * SE_max > 1 -> must clip to 1.0
        mu_large = 0.99
        se_max_l, (ci_low_l, ci_high_l) = compute_bhatia_davis_se_bound(mu_large, n)
        assert ci_low_l >= 0.0
        assert ci_high_l == 1.0

    def test_invalid_inputs_raise_value_error(self):
        with pytest.raises(ValueError, match="n_users must be positive"):
            compute_bhatia_davis_se_bound(0.5, 0)

        with pytest.raises(ValueError, match="n_users must be positive"):
            compute_bhatia_davis_se_bound(0.5, -10)

        with pytest.raises(ValueError, match="mean_score must be in"):
            compute_bhatia_davis_se_bound(-0.1, 100)

        with pytest.raises(ValueError, match="mean_score must be in"):
            compute_bhatia_davis_se_bound(1.05, 100)


class TestEvaluateFromUserCSV:
    """Tests for evaluate_from_user_csv."""

    def test_real_dataframe_evaluation_and_significance(self):
        # Create a deterministic mock DataFrame with 30 users and 2 models
        n_users = 30
        users = [f"u_{i}" for i in range(n_users)]

        # Candidate model has higher scores than baseline
        scores_cand = [0.15 + 0.01 * (i % 5) for i in range(n_users)]
        scores_base = [0.08 + 0.005 * (i % 5) for i in range(n_users)]

        rows = []
        for u, s_c in zip(users, scores_cand):
            rows.append({
                "dataset": "Zeeguu",
                "model": "LightGCN ADA",
                "user_id": u,
                "Recall@5": s_c,
                "Precision@5": s_c * 0.4,
                "F1@5": s_c * 0.5,
                "NDCG@5": s_c * 0.9,
                "MRR@5": s_c * 1.1,
                "MAP@5": s_c * 0.8,
            })
        for u, s_b in zip(users, scores_base):
            rows.append({
                "dataset": "Zeeguu",
                "model": "LightGCN Xavier ADA",
                "user_id": u,
                "Recall@5": s_b,
                "Precision@5": s_b * 0.4,
                "F1@5": s_b * 0.5,
                "NDCG@5": s_b * 0.9,
                "MRR@5": s_b * 1.1,
                "MAP@5": s_b * 0.8,
            })

        df_user = pd.DataFrame(rows)
        results = evaluate_from_user_csv(df_user, n_boot=200, seed=42)

        assert results["mode"] == "user_level_evaluation"
        assert "Zeeguu" in results["datasets"]
        zeeguu = results["datasets"]["Zeeguu"]

        # Check empirical mean preservation
        expected_mean = float(np.mean(scores_cand))
        actual_mean = zeeguu["models"]["LightGCN ADA"]["Recall@5"]["mean"]
        assert actual_mean == pytest.approx(expected_mean, abs=1e-6)

        # Check paired comparisons
        assert "LightGCN ADA vs LightGCN Xavier ADA" in zeeguu["comparisons"]
        comp = zeeguu["comparisons"]["LightGCN ADA vs LightGCN Xavier ADA"]["Recall@5"]

        assert comp["n_users"] == n_users
        assert comp["diff_mean"] > 0.0
        assert comp["wilcoxon_p"] < 0.001
        assert comp["p_adj"] < 0.001
        assert comp["rejected"] is True
        assert comp["latex_marker"] == "^{***}"

    def test_missing_required_columns_raises_error(self):
        df_invalid = pd.DataFrame({"dataset": ["Zeeguu"], "model": ["ALS"]})
        with pytest.raises(ValueError, match="missing required columns"):
            evaluate_from_user_csv(df_invalid)

    def test_no_metrics_raises_error(self):
        df_no_metrics = pd.DataFrame({"dataset": ["Zeeguu"], "model": ["ALS"], "user_id": ["u1"]})
        with pytest.raises(ValueError, match="No numeric metric columns"):
            evaluate_from_user_csv(df_no_metrics)


class TestExportNotebookMetricsToUserCSV:
    """Tests for export_notebook_metrics_to_user_csv."""

    def test_cell_13_structure_export(self, tmp_path):
        # Exact structure from Cell 13 of c_Recommendations.ipynb
        cell_13_output = {
            "mndcg": 0.0666,
            "mrr": 0.0843,
            "map": 0.0550,
            "mrecall": 0.0721,
            "mprecision": 0.0274,
            "mf1": 0.0352,
            "ndcgs": [0.05, 0.08, 0.12, 0.02, 0.06],
            "rrs": [0.07, 0.10, 0.15, 0.03, 0.07],
            "aps": [0.04, 0.07, 0.11, 0.01, 0.05],
            "recalls": [0.06, 0.09, 0.13, 0.02, 0.06],
            "precisions": [0.02, 0.03, 0.05, 0.01, 0.02],
            "f1s": [0.03, 0.04, 0.07, 0.01, 0.03],
            "all_recommendations": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            "all_really_in_test": [[1], [3], [5], [7], [9]],
        }

        csv_file = str(tmp_path / "exported_user_metrics.csv")
        df = export_notebook_metrics_to_user_csv(
            evaluation_dict=cell_13_output,
            output_csv_path=csv_file,
            dataset="Zeeguu",
            model="LightGCN ADA",
        )

        assert len(df) == 5
        expected_cols = [
            "dataset",
            "model",
            "user_id",
            "Recall@5",
            "Precision@5",
            "F1@5",
            "NDCG@5",
            "MRR@5",
            "MAP@5",
        ]
        assert list(df.columns) == expected_cols
        assert df["NDCG@5"].tolist() == [0.05, 0.08, 0.12, 0.02, 0.06]
        assert df["Recall@5"].tolist() == [0.06, 0.09, 0.13, 0.02, 0.06]
        assert df["user_id"].tolist() == [0, 1, 2, 3, 4]

        # Verify disk write and re-reading
        re_read = pd.read_csv(csv_file)
        assert len(re_read) == 5
        assert re_read["dataset"].iloc[0] == "Zeeguu"

    def test_multi_model_nested_dictionary(self, tmp_path):
        multi_dict = {
            "LightGCN ADA": {
                "ndcgs": [0.1, 0.2],
                "rrs": [0.15, 0.25],
                "aps": [0.08, 0.18],
                "recalls": [0.05, 0.15],
                "precisions": [0.02, 0.04],
                "f1s": [0.03, 0.06],
            },
            "LightGCN Xavier ADA": {
                "ndcgs": [0.08, 0.16],
                "rrs": [0.12, 0.20],
                "aps": [0.06, 0.14],
                "recalls": [0.04, 0.12],
                "precisions": [0.015, 0.03],
                "f1s": [0.025, 0.05],
            },
        }

        csv_file = str(tmp_path / "multi_model.csv")
        df = export_notebook_metrics_to_user_csv(multi_dict, csv_file, dataset="ml-100k")
        assert len(df) == 4
        assert set(df["model"].unique()) == {"LightGCN ADA", "LightGCN Xavier ADA"}


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
        sorted_res = sorted(results, key=lambda x: x["p_raw"])
        for i in range(1, len(sorted_res)):
            assert sorted_res[i]["p_adj"] >= sorted_res[i - 1]["p_adj"]

    def test_markers_attribution(self):
        raw_p = [0.00001, 0.002, 0.01, 0.25]
        results = apply_holm_bonferroni(raw_p, alpha=0.05)

        assert results[0]["latex_marker"] == "^{***}"
        assert results[0]["rejected"] is True

        assert results[1]["latex_marker"] == "^{**}"
        assert results[1]["rejected"] is True

        assert results[2]["latex_marker"] == "^{*}"
        assert results[2]["rejected"] is True

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
    """Integration test with Table 8 deterministic analytical evaluation."""

    def test_evaluate_table_significance_structure(self):
        res = evaluate_table_significance(TABLE_8_DATA, user_counts=DATASET_USER_COUNTS)

        assert res["mode"] == "bhatia_davis_analytical_upper_bound"
        assert "Zeeguu" in res["datasets"]
        zeeguu_models = res["datasets"]["Zeeguu"]["models"]
        assert "LightGCN ADA" in zeeguu_models
        assert "Recall@5" in zeeguu_models["LightGCN ADA"]
        stats_recall = zeeguu_models["LightGCN ADA"]["Recall@5"]
        assert stats_recall["ci95_low"] <= stats_recall["mean"] <= stats_recall["ci95_high"]
        assert stats_recall["bound_type"] == "bhatia_davis_upper_bound"
