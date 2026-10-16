"""
compute_metrics.py - Engine for Statistical Evaluation of Difficulty Estimation (DMKD Revision).

This script performs the complete statistical evaluation required by the Action Editor
and Reviewers for the DMKD major revision:
1. Non-parametric bootstrap (B=1000) for standard deviations (+/- sigma) and 95% CIs
   on Accuracy, Macro-F1, MAE, QWK, and Normalized Pairwise Mismatch.
2. Paired Wilcoxon signed-rank tests (scipy.stats.wilcoxon) evaluating statistical
   significance against baselines on instance-level ordinal absolute errors.
3. Quadratic Weighted Kappa (QWK) on ordinal CEFR classes (A1=0 to C2=5, Level1=0 to Level4=3).
4. Formally normalized Pairwise Mismatch (PM_norm) bounded in [0, 1] (and expressed in %),
   divided by C(N, 2) with ties counted as 0.5.
5. Export of table2_with_significance.csv and publication-ready LaTeX table difficulty_estimation_metrics.tex.

Adheres strictly to fail-fast principles: no silent exceptions, strict typing, explicit assertions.
"""

import os
import sys
import glob
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    cohen_kappa_score,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DMKD_Metrics")

# CEFR and LjL Ordinal Mappings
ORDINAL_MAPPING: Dict[str, int] = {
    "A1": 0,
    "A2": 1,
    "B1": 2,
    "B2": 3,
    "C1": 4,
    "C2": 5,
    "LEVEL1": 0,
    "LEVEL2": 1,
    "LEVEL3": 2,
    "LEVEL4": 3,
}

DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "french-difficulty": "Sentences Books",
    "french_difficulty": "Sentences Books",
    "ljl": "LjL",
    "sentences": "Sentences Internet",
}

DATASET_NUM_CLASSES: Dict[str, int] = {
    "french-difficulty": 6,
    "french_difficulty": 6,
    "ljl": 4,
    "sentences": 6,
}


def normalize_dataset_name(dataset: str) -> str:
    """Normalize raw dataset name to canonical identifier."""
    ds = str(dataset).strip().lower()
    if ds in ["french", "french_difficulty", "french-difficulty"]:
        return "french-difficulty"
    return ds


def normalize_model_name(model: str) -> str:
    """Normalize model string to canonical paper naming."""
    m = str(model).strip()
    if "ft_" in m:
        if "gpt-4o-mini" in m.lower():
            return "GPT-4o-mini"
        elif "gpt-4o" in m.lower():
            return "GPT-4o"
    if "gpt-4o-mini" in m.lower():
        return "GPT-4o-mini"
    if "gpt-4o" in m.lower():
        return "GPT-4o"
    if "gpt-3.5" in m.lower() or "gpt-35" in m.lower():
        return "GPT-3.5"
    if "camembert" in m.lower() or "bert" in m.lower():
        return "CamemBERT"
    if "mistral" in m.lower():
        return "Mistral-7B"
    if "davinci" in m.lower():
        return "Davinci"
    if "babbage" in m.lower():
        return "Babbage"
    return m


def normalize_context(context: str) -> str:
    """Normalize context string to 'with_context' or 'no_context'."""
    ctx = str(context).strip().lower()
    with_ctx_indicators = ["with_system", "with_context", "cecrl", "with_prompt", "true", "yes"]
    if any(ind in ctx for ind in with_ctx_indicators):
        return "with_context"
    return "no_context"


def parse_prediction_filename(filename: str) -> Tuple[str, str, str]:
    """Parse metadata from prediction CSV filename."""
    base = os.path.basename(filename)
    if "_ft_" not in base:
        raise ValueError(f"Filename does not match expected fine-tuning format: {base}")
    
    parts = base.split("_ft_")
    dataset_raw = parts[0]
    rest = parts[1]
    
    # Model extraction
    if "gpt-4o-mini" in rest.lower():
        model = "GPT-4o-mini"
    elif "gpt-4o" in rest.lower():
        model = "GPT-4o"
    else:
        raise ValueError(f"Could not infer model from filename: {base}")
    
    # Context extraction
    if "_with_system" in rest:
        context = "with_context"
    elif "_no_system" in rest:
        context = "no_context"
    else:
        raise ValueError(f"Could not infer context (with_system/no_system) from filename: {base}")
    
    dataset = normalize_dataset_name(dataset_raw)
    return dataset, model, context


def encode_labels(labels: pd.Series, max_class: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode labels into ordinal integers.
    Returns:
        encoded_array (np.ndarray): Integer array with valid indices, or -1 for missing.
        valid_mask (np.ndarray): Boolean mask indicating non-missing entries.
    """
    encoded = []
    for val in labels:
        if pd.isna(val):
            encoded.append(-1)
        else:
            cleaned = str(val).strip().upper()
            idx = ORDINAL_MAPPING.get(cleaned, -1)
            if idx is not None and 0 <= idx < max_class:
                encoded.append(idx)
            else:
                encoded.append(-1)
    arr = np.array(encoded, dtype=int)
    valid_mask = (arr >= 0)
    return arr, valid_mask


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    """Build K x K confusion matrix where C[u, p] is count of true=u, pred=p."""
    C = np.zeros((K, K), dtype=float)
    for u, p in zip(y_true, y_pred):
        if 0 <= u < K and 0 <= p < K:
            C[u, p] += 1.0
    return C


def compute_pm_normalized_from_confusion(C: np.ndarray, N: int, K: int) -> float:
    """
    Compute Normalized Pairwise Mismatch on [0, 1] from confusion matrix C.
    Formula:
        PM_norm = (1 / C(N, 2)) * sum_{u < v} sum_{pu, pv} C[u, pu] * C[v, pv] * weight(pu, pv)
    where:
        weight(pu, pv) = 1.0 if pu > pv (strict inversion)
        weight(pu, pv) = 0.5 if pu == pv (tie in prediction on distinct ground truth)
        weight(pu, pv) = 0.0 if pu < pv (correct order)
    """
    if N < 2:
        return 0.0
    
    total_mismatch = 0.0
    for u in range(K):
        for v in range(u + 1, K):
            for pu in range(K):
                for pv in range(K):
                    count = C[u, pu] * C[v, pv]
                    if count == 0:
                        continue
                    if pu > pv:
                        total_mismatch += 1.0 * count
                    elif pu == pv:
                        total_mismatch += 0.5 * count
    
    total_pairs = float(N * (N - 1) / 2.0)
    return float(total_mismatch / total_pairs)


def compute_qwk_from_confusion(C: np.ndarray, N: int, K: int) -> float:
    """
    Compute Quadratic Weighted Kappa directly from confusion matrix C.
    Weight matrix: W[i, j] = (i - j)^2 / (K - 1)^2.
    """
    if N == 0 or K < 2:
        return 0.0
    
    u_idx, p_idx = np.indices((K, K))
    W = ((u_idx - p_idx) ** 2) / float((K - 1) ** 2)
    
    O = C / float(N)
    r_marg = np.sum(O, axis=1, keepdims=True)
    c_marg = np.sum(O, axis=0, keepdims=True)
    E = r_marg @ c_marg
    
    num = np.sum(W * O)
    den = np.sum(W * E)
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return float(1.0 - (num / den))


def compute_f1_macro_from_confusion(C: np.ndarray, K: int) -> float:
    """Compute Macro-F1 score from confusion matrix C."""
    tp = np.diag(C)
    fp = np.sum(C, axis=0) - tp
    fn = np.sum(C, axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1_per_class = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    return float(np.mean(f1_per_class))


def compute_mae_from_confusion(C: np.ndarray, N: int, K: int) -> float:
    """Compute Mean Absolute Error (MAE) from confusion matrix C."""
    if N == 0:
        return 0.0
    u_idx, p_idx = np.indices((K, K))
    diff_matrix = np.abs(u_idx - p_idx)
    return float(np.sum(C * diff_matrix) / float(N))


def compute_metrics_single_run(
    df: pd.DataFrame, K: int
) -> Dict[str, float]:
    """
    Compute nominal evaluation metrics for a single prediction DataFrame.
    
    Args:
        df: DataFrame containing 'difficulty' and 'predictions' columns.
        K: Number of ordinal difficulty classes.
        
    Returns:
        Dict of scalar metric values.
    """
    # 1. Global nominal accuracy (non-responses count as classification error)
    y_true_str = df["difficulty"].astype(str).str.strip().str.upper()
    y_pred_str = df["predictions"].astype(str).str.strip().str.upper()
    # If prediction is NaN, string is 'NAN', which does not equal true label
    acc_global = float((y_pred_str == y_true_str).mean())
    
    # 2. Ordinal metrics on valid predictions
    y_true_ord, true_mask = encode_labels(df["difficulty"], K)
    y_pred_ord, pred_mask = encode_labels(df["predictions"], K)
    valid = true_mask & pred_mask
    
    if not np.all(true_mask):
        raise ValueError("Ground truth labels contain unrecognized or missing entries.")
    
    # Restrict to valid pairs for ordinal evaluation
    y_t_valid = y_true_ord[valid]
    y_p_valid = y_pred_ord[valid]
    N_valid = len(y_t_valid)
    
    if N_valid == 0:
        raise ValueError("No valid predictions to compute ordinal metrics.")
    
    C = compute_confusion_matrix(y_t_valid, y_p_valid, K)
    
    pm_norm = compute_pm_normalized_from_confusion(C, N_valid, K)
    qwk = compute_qwk_from_confusion(C, N_valid, K)
    mae = compute_mae_from_confusion(C, N_valid, K)
    f1_macro = compute_f1_macro_from_confusion(C, K)
    
    return {
        "accuracy": acc_global,
        "f1_macro": f1_macro,
        "mae": mae,
        "qwk": qwk,
        "pm_norm": pm_norm,
        "pm_norm_pct": pm_norm * 100.0,
        "n_samples": float(len(df)),
        "n_valid": float(N_valid),
    }


def bootstrap_metrics(
    df: pd.DataFrame, K: int, B: int = 1000, seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Perform non-parametric bootstrap (B iterations) to estimate variances and 95% CIs.
    
    Args:
        df: DataFrame with 'difficulty' and 'predictions'.
        K: Number of classes.
        B: Number of bootstrap resamples (default 1000).
        seed: Random seed for reproducibility.
        
    Returns:
        Dict mapping metric name to dict containing mean, std, ci95_low, ci95_high.
    """
    rng = np.random.default_rng(seed)
    N = len(df)
    
    y_true_str = df["difficulty"].astype(str).str.strip().str.upper().values
    y_pred_str = df["predictions"].astype(str).str.strip().str.upper().values
    y_true_ord, true_mask = encode_labels(df["difficulty"], K)
    y_pred_ord, pred_mask = encode_labels(df["predictions"], K)
    valid_mask = true_mask & pred_mask
    
    acc_samples = np.empty(B, dtype=float)
    f1_samples = np.empty(B, dtype=float)
    mae_samples = np.empty(B, dtype=float)
    qwk_samples = np.empty(B, dtype=float)
    pm_samples = np.empty(B, dtype=float)
    
    for b in range(B):
        # Sample N indices with replacement
        idx = rng.integers(0, N, size=N)
        
        # Accuracy
        acc_samples[b] = (y_pred_str[idx] == y_true_str[idx]).mean()
        
        # Valid subset for ordinal metrics
        idx_valid = idx[valid_mask[idx]]
        N_val = len(idx_valid)
        if N_val < 2:
            # Fallback for degenerate sample
            f1_samples[b] = 0.0
            mae_samples[b] = 0.0
            qwk_samples[b] = 0.0
            pm_samples[b] = 0.0
            continue
        
        # Build confusion matrix for sample
        sub_yt = y_true_ord[idx_valid]
        sub_yp = y_pred_ord[idx_valid]
        C_b = compute_confusion_matrix(sub_yt, sub_yp, K)
        
        acc_samples[b] = (y_pred_str[idx] == y_true_str[idx]).mean()
        f1_samples[b] = compute_f1_macro_from_confusion(C_b, K)
        mae_samples[b] = compute_mae_from_confusion(C_b, N_val, K)
        qwk_samples[b] = compute_qwk_from_confusion(C_b, N_val, K)
        pm_samples[b] = compute_pm_normalized_from_confusion(C_b, N_val, K)
    
    def summarize(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    
    return {
        "accuracy": summarize(acc_samples),
        "f1_macro": summarize(f1_samples),
        "mae": summarize(mae_samples),
        "qwk": summarize(qwk_samples),
        "pm_norm": summarize(pm_samples),
        "pm_norm_pct": summarize(pm_samples * 100.0),
    }


def compute_instance_ordinal_errors(df: pd.DataFrame, K: int) -> np.ndarray:
    """
    Compute instance-level absolute ordinal error |y_true - y_pred|.
    Unrecognized or missing predictions receive the maximum possible error penalty (K - 1).
    """
    y_true_ord, true_mask = encode_labels(df["difficulty"], K)
    y_pred_ord, pred_mask = encode_labels(df["predictions"], K)
    
    if not np.all(true_mask):
        raise ValueError("True labels have invalid entries.")
    
    max_err = K - 1
    errors = np.where(pred_mask, np.abs(y_true_ord - y_pred_ord), max_err)
    return errors.astype(float)


def compute_wilcoxon_significance(
    errors_model: np.ndarray, errors_baseline: np.ndarray
) -> Tuple[float, float, str]:
    """
    Perform two-sided paired Wilcoxon signed-rank test.
    Returns:
        stat: Test statistic W
        p_val: Two-sided p-value
        stars: Significance stars ('***', '**', '*', 'ns')
    """
    if len(errors_model) != len(errors_baseline):
        raise ValueError(
            f"Array length mismatch: {len(errors_model)} vs {len(errors_baseline)}"
        )
    
    diff = errors_model - errors_baseline
    if np.all(diff == 0.0):
        return 0.0, 1.0, "ns"
    
    res = wilcoxon(errors_model, errors_baseline, zero_method="wilcox", alternative="two-sided")
    p_val = float(res.pvalue)
    
    if p_val < 0.001:
        stars = "***"
    elif p_val < 0.01:
        stars = "**"
    elif p_val < 0.05:
        stars = "*"
    else:
        stars = "ns"
        
    return float(res.statistic), p_val, stars


def load_historical_metrics(historical_csv_path: str) -> pd.DataFrame:
    """Load and format historical metrics for baselines."""
    if not os.path.exists(historical_csv_path):
        logger.warning(f"Historical metrics not found at {historical_csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(historical_csv_path)
    df["dataset"] = df["dataset"].astype(str).apply(normalize_dataset_name)
    df["model"] = df["model"].astype(str).apply(normalize_model_name)
    df["context"] = df["context"].astype(str).apply(normalize_context)
    
    # Dataset sample sizes N for historical PM normalization: PM_norm = PM_raw / (N - 1)
    dataset_sizes = {
        "french-difficulty": 960,
        "ljl": 413,
        "sentences": 480,
    }
    
    pm_norm_list = []
    pm_norm_pct_list = []
    for _, row in df.iterrows():
        ds = row["dataset"]
        raw_pm = row.get("pairwise_mismatch", None)
        if pd.notna(raw_pm) and ds in dataset_sizes:
            N = dataset_sizes[ds]
            norm_val = float(raw_pm) / float(N - 1)
            pm_norm_list.append(norm_val)
            pm_norm_pct_list.append(norm_val * 100.0)
        else:
            pm_norm_list.append(np.nan)
            pm_norm_pct_list.append(np.nan)
            
    df["pm_norm"] = pm_norm_list
    df["pm_norm_pct"] = pm_norm_pct_list
    return df


def generate_latex_table(
    consolidated_df: pd.DataFrame, output_tex_path: str
) -> str:
    """
    Generate Springer Nature DMKD compliant Table 2 in LaTeX with:
    - Model and System Prompt indication
    - Acc. and PM (norm %) for Sentences Books, LjL, and Sentences Internet
    - Significance marks and best values highlighted (bold and underline)
    - Footnotes detailing model checkpoints and statistical rigor.
    """
    logger.info("Generating publication-ready LaTeX table...")
    
    # Filter best or distinct entries per model/context
    pivoted_acc = consolidated_df.pivot_table(
        index=["model", "context"],
        columns="dataset",
        values="accuracy",
        aggfunc="first",
    )
    pivoted_pm = consolidated_df.pivot_table(
        index=["model", "context"],
        columns="dataset",
        values="pm_norm_pct",
        aggfunc="first",
    )
    pivoted_stars = consolidated_df.pivot_table(
        index=["model", "context"],
        columns="dataset",
        values="significance_vs_no_context",
        aggfunc="first",
    )
    
    # Column mapping
    col_order = ["french-difficulty", "ljl", "sentences"]
    col_labels = ["Sentences Books", "LjL", "Sentences Internet"]
    
    # Define canonical model presentation order
    canonical_model_order = [
        "GPT-4o",
        "GPT-4o-mini",
        "CamemBERT",
        "GPT-3.5",
        "Mistral-7B",
        "Davinci",
        "Babbage",
        "ARI",
        "GFI",
        "FKGL",
    ]
    
    # Filter present models keeping canonical order
    present_models = [m for m in canonical_model_order if m in consolidated_df["model"].values]
    # Add any extra model not in canonical list
    for m in consolidated_df["model"].unique():
        if m not in present_models:
            present_models.append(m)

    caption = (
        "Difficulty estimation metrics for all datasets (DMKD major revision). "
        "For each model and dataset, we report classification Accuracy (Acc.) and formally "
        "normalized Pairwise Mismatch ($PM_{\\text{norm}}$, \\%), measuring difficulty ordering "
        "violations scaled over $\\binom{N}{2}$ with ties counted as $0.5$. "
        "Lower $PM_{\\text{norm}}$ indicates superior ordinal ranking consistency. "
        "System Prompt indicates whether explicit CEFR descriptions were provided (\\checkmark) or not (-). "
        "Asterisks indicate paired Wilcoxon signed-rank significance against unprompted baselines "
        "($^{*}p<0.05, ^{**}p<0.01, ^{***}p<0.001$). "
        "\\textbf{\\underline{Bold and underlined}} values indicate the best score per metric and dataset."
    )
    
    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("    \\centering")
    latex.append("    \\small")
    latex.append("    \\setlength{\\tabcolsep}{3.5pt}")
    latex.append("    \\begin{tabular}{lccccccc}")
    latex.append("        \\toprule")
    latex.append("        & & \\multicolumn{2}{c}{\\textbf{Sentences Books}} & \\multicolumn{2}{c}{\\textbf{LjL}} & \\multicolumn{2}{c}{\\textbf{Sentences Internet}} \\\\")
    latex.append("        \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}")
    latex.append("        \\textbf{Model} & \\textbf{System Prompt} & Acc. & PM (\\%) & Acc. & PM (\\%) & Acc. & PM (\\%) \\\\")
    latex.append("        \\midrule")
    
    footnoted_models = {
        "GPT-3.5",
        "CamemBERT",
        "Mistral-7B",
        "Davinci",
        "Babbage",
        "GPT-4o",
        "GPT-4o-mini",
    }
    
    current_model = None
    
    for m in present_models:
        # Check which contexts exist for this model
        sub = consolidated_df[consolidated_df["model"] == m]
        contexts = sorted(sub["context"].unique(), reverse=True) # with_context first
        
        if current_model is not None:
            latex.append("        \\\\[2pt]")
        current_model = m
        
        for i, ctx in enumerate(contexts):
            if i == 0:
                model_str = f"\\textbf{{{m}}}"
                if m in footnoted_models:
                    model_str += "\\textsuperscript{3}"
            else:
                model_str = ""
                
            ctx_str = "\\checkmark" if ctx == "with_context" else "-"
            
            row_cells = []
            for ds in col_order:
                try:
                    acc = pivoted_acc.loc[(m, ctx), ds]
                except KeyError:
                    acc = np.nan
                try:
                    pm = pivoted_pm.loc[(m, ctx), ds]
                except KeyError:
                    pm = np.nan
                try:
                    star = pivoted_stars.loc[(m, ctx), ds]
                except KeyError:
                    star = ""
                    
                if pd.isna(star) or str(star).strip() in ["ns", "", "nan"]:
                    star_tex = ""
                else:
                    star_tex = f"^{{ {str(star).strip()} }}"
                    
                if pd.isna(acc) or pd.isna(pm):
                    row_cells.extend(["-", "-"])
                else:
                    best_acc = pivoted_acc[ds].max()
                    best_pm = pivoted_pm[ds].min()
                    
                    acc_formatted = f"{acc:.2f}{star_tex}"
                    if abs(acc - best_acc) < 1e-4:
                        acc_formatted = f"\\textbf{{\\underline{{{acc_formatted}}}}}"
                        
                    pm_formatted = f"{pm:.1f}"
                    if abs(pm - best_pm) < 1e-4:
                        pm_formatted = f"\\textbf{{\\underline{{{pm_formatted}}}}}"
                        
                    row_cells.extend([acc_formatted, pm_formatted])
                    
            latex.append(f"        {model_str} & {ctx_str} & {' & '.join(row_cells)} \\\\")
            
    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append("    \\label{tab:difficulty_estimation_metrics}")
    latex.append("\\end{table}")
    latex.append("")
    latex.append("\\footnotetext[3]{%")
    latex.append("    In this table, ``GPT-3.5'' corresponds to \\texttt{gpt-3.5-turbo-1106}, "
                 "``GPT-4o'' and ``GPT-4o-mini'' to our fine-tuned models (evaluated at $T=0$), "
                 "``CamemBERT'' to \\texttt{camembert-base}, "
                 "``Mistral'' to \\texttt{Mistral-7B}, and ``Davinci'' to \\texttt{davinci-002}.}")
    latex.append("")
    
    content = "\n".join(latex)
    os.makedirs(os.path.dirname(os.path.abspath(output_tex_path)), exist_ok=True)
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"LaTeX table successfully exported to {output_tex_path}")
    
    # Generate Detailed LaTeX Table with Variances (+/- sigma) and QWK
    detailed_tex_path = output_tex_path.replace(".tex", "_detailed.tex")
    generate_detailed_latex_table(consolidated_df, detailed_tex_path, present_models, col_order)
    
    return content


def generate_detailed_latex_table(
    consolidated_df: pd.DataFrame, output_tex_path: str, present_models: List[str], col_order: List[str]
) -> str:
    """Generate detailed table with Accuracy (+/- sigma), QWK (+/- sigma), and PM_norm (+/- sigma)."""
    pivoted_acc = consolidated_df.pivot_table(index=["model", "context"], columns="dataset", values="accuracy", aggfunc="first")
    pivoted_acc_std = consolidated_df.pivot_table(index=["model", "context"], columns="dataset", values="accuracy_std", aggfunc="first")
    pivoted_qwk = consolidated_df.pivot_table(index=["model", "context"], columns="dataset", values="qwk", aggfunc="first")
    pivoted_qwk_std = consolidated_df.pivot_table(index=["model", "context"], columns="dataset", values="qwk_std", aggfunc="first")
    pivoted_pm = consolidated_df.pivot_table(index=["model", "context"], columns="dataset", values="pm_norm_pct", aggfunc="first")
    pivoted_pm_std = consolidated_df.pivot_table(index=["model", "context"], columns="dataset", values="pm_norm_pct_std", aggfunc="first")
    pivoted_stars = consolidated_df.pivot_table(index=["model", "context"], columns="dataset", values="significance_vs_no_context", aggfunc="first")

    caption = (
        "Comprehensive statistical evaluation of difficulty estimation models across datasets (DMKD revision). "
        "Metrics reported as mean $\\pm$ bootstrap standard error ($\\sigma$, $B=1000$). "
        "Acc.: classification accuracy; QWK: Quadratic Weighted Kappa on ordinal CEFR classes; "
        "$PM_{\\text{norm}}$: normalized Pairwise Mismatch (\\%). "
        "Asterisks denote Wilcoxon significance against unprompted variants ($^{*}p<0.05, ^{**}p<0.01, ^{***}p<0.001$)."
    )

    latex = []
    latex.append("\\begin{table*}[!htbp]")
    latex.append("    \\centering")
    latex.append("    \\footnotesize")
    latex.append("    \\setlength{\\tabcolsep}{3pt}")
    latex.append("    \\begin{tabular}{lccccccccccc}")
    latex.append("        \\toprule")
    latex.append("        & & \\multicolumn{3}{c}{\\textbf{Sentences Books}} & \\multicolumn{3}{c}{\\textbf{LjL}} & \\multicolumn{3}{c}{\\textbf{Sentences Internet}} \\\\")
    latex.append("        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}")
    latex.append("        \\textbf{Model} & \\textbf{Prompt} & Acc. & QWK & PM (\\%) & Acc. & QWK & PM (\\%) & Acc. & QWK & PM (\\%) \\\\")
    latex.append("        \\midrule")

    current_model = None
    for m in present_models:
        sub = consolidated_df[consolidated_df["model"] == m]
        contexts = sorted(sub["context"].unique(), reverse=True)
        if current_model is not None:
            latex.append("        \\\\[2pt]")
        current_model = m

        for i, ctx in enumerate(contexts):
            model_str = f"\\textbf{{{m}}}" if i == 0 else ""
            ctx_str = "\\checkmark" if ctx == "with_context" else "-"

            row_cells = []
            for ds in col_order:
                acc = pivoted_acc.loc[(m, ctx), ds] if ((m, ctx) in pivoted_acc.index and ds in pivoted_acc.columns) else np.nan
                acc_std = pivoted_acc_std.loc[(m, ctx), ds] if ((m, ctx) in pivoted_acc_std.index and ds in pivoted_acc_std.columns) else np.nan
                
                qwk = pivoted_qwk.loc[(m, ctx), ds] if ((m, ctx) in pivoted_qwk.index and ds in pivoted_qwk.columns) else np.nan
                qwk_std = pivoted_qwk_std.loc[(m, ctx), ds] if ((m, ctx) in pivoted_qwk_std.index and ds in pivoted_qwk_std.columns) else np.nan
                
                pm = pivoted_pm.loc[(m, ctx), ds] if ((m, ctx) in pivoted_pm.index and ds in pivoted_pm.columns) else np.nan
                pm_std = pivoted_pm_std.loc[(m, ctx), ds] if ((m, ctx) in pivoted_pm_std.index and ds in pivoted_pm_std.columns) else np.nan
                
                star = pivoted_stars.loc[(m, ctx), ds] if ((m, ctx) in pivoted_stars.index and ds in pivoted_stars.columns) else ""

                star_tex = f"^{{ {str(star).strip()} }}" if pd.notna(star) and str(star).strip() not in ["ns", "", "nan"] else ""

                if pd.isna(acc):
                    row_cells.extend(["-", "-", "-"])
                else:
                    # Accuracy cell
                    if pd.notna(acc_std):
                        acc_cell = f"${acc:.2f} \\pm {acc_std:.2f}${star_tex}"
                    else:
                        acc_cell = f"${acc:.2f}${star_tex}"

                    # QWK cell
                    if pd.notna(qwk) and pd.notna(qwk_std):
                        qwk_cell = f"${qwk:.2f} \\pm {qwk_std:.2f}$"
                    elif pd.notna(qwk):
                        qwk_cell = f"${qwk:.2f}$"
                    else:
                        qwk_cell = "-"

                    # PM cell
                    if pd.notna(pm_std):
                        pm_cell = f"${pm:.1f} \\pm {pm_std:.1f}$"
                    else:
                        pm_cell = f"${pm:.1f}$"

                    row_cells.extend([acc_cell, qwk_cell, pm_cell])

            latex.append(f"        {model_str} & {ctx_str} & {' & '.join(row_cells)} \\\\")

    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append("    \\label{tab:difficulty_estimation_metrics_detailed}")
    latex.append("\\end{table*}")
    latex.append("")

    content = "\n".join(latex)
    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Detailed LaTeX table exported to {output_tex_path}")
    return content


def run_pipeline(
    data_dir: str,
    output_dir: str,
    historical_metrics_path: Optional[str] = None,
    b_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the complete metrics, bootstrap, Wilcoxon and LaTeX pipeline."""
    logger.info(f"Starting DMKD Revision Metrics Pipeline (B={b_bootstrap}, seed={seed})")
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Input directory does not exist: {data_dir}")
        
    pattern = os.path.join(data_dir, "*_predictions.csv")
    csv_files = sorted(glob.glob(pattern))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No prediction CSV files found in {data_dir}")
        
    logger.info(f"Discovered {len(csv_files)} prediction CSV files.")
    assert len(csv_files) == 12, f"Expected exactly 12 prediction files, found {len(csv_files)}"
    
    # Store loaded dataframes and instance errors for Wilcoxon
    loaded_data: Dict[str, Dict[str, Any]] = {}
    
    # Process each file
    records = []
    for fpath in csv_files:
        filename = os.path.basename(fpath)
        dataset, model, context = parse_prediction_filename(filename)
        K = DATASET_NUM_CLASSES[dataset]
        
        logger.info(f"Processing: {dataset} | {model} | {context} (K={K})")
        df = pd.read_csv(fpath)
        
        # Verify columns
        for col in ["sentence", "difficulty", "predictions"]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from {filename}")
                
        # 1. Nominal Metrics
        nominal = compute_metrics_single_run(df, K)
        
        # 2. Bootstrap Variances (B=1000)
        logger.info(f"  Running Bootstrap (B={b_bootstrap})...")
        boot = bootstrap_metrics(df, K, B=b_bootstrap, seed=seed)
        
        # 3. Instance Errors for Wilcoxon
        errors = compute_instance_ordinal_errors(df, K)
        
        entry_key = f"{dataset}__{model}__{context}"
        loaded_data[entry_key] = {
            "dataset": dataset,
            "model": model,
            "context": context,
            "errors": errors,
            "df": df,
            "K": K,
        }
        
        rec = {
            "dataset": dataset,
            "model": model,
            "context": context,
            "accuracy": nominal["accuracy"],
            "accuracy_boot_mean": boot["accuracy"]["mean"],
            "accuracy_std": boot["accuracy"]["std"],
            "accuracy_ci95_low": boot["accuracy"]["ci95_low"],
            "accuracy_ci95_high": boot["accuracy"]["ci95_high"],
            "f1_macro": nominal["f1_macro"],
            "f1_macro_std": boot["f1_macro"]["std"],
            "f1_macro_ci95_low": boot["f1_macro"]["ci95_low"],
            "f1_macro_ci95_high": boot["f1_macro"]["ci95_high"],
            "mae": nominal["mae"],
            "mae_std": boot["mae"]["std"],
            "mae_ci95_low": boot["mae"]["ci95_low"],
            "mae_ci95_high": boot["mae"]["ci95_high"],
            "qwk": nominal["qwk"],
            "qwk_std": boot["qwk"]["std"],
            "qwk_ci95_low": boot["qwk"]["ci95_low"],
            "qwk_ci95_high": boot["qwk"]["ci95_high"],
            "pm_norm": nominal["pm_norm"],
            "pm_norm_pct": nominal["pm_norm_pct"],
            "pm_norm_pct_std": boot["pm_norm_pct"]["std"],
            "pm_norm_pct_ci95_low": boot["pm_norm_pct"]["ci95_low"],
            "pm_norm_pct_ci95_high": boot["pm_norm_pct"]["ci95_high"],
            "n_samples": nominal["n_samples"],
            "n_valid": nominal["n_valid"],
        }
        records.append(rec)
        
    current_metrics_df = pd.DataFrame(records)
    
    # Compute Wilcoxon tests
    logger.info("Computing paired Wilcoxon signed-rank tests...")
    wilcoxon_records = []
    
    # Match with_context vs no_context for the same model & dataset
    for idx_row, row in current_metrics_df.iterrows():
        ds = row["dataset"]
        m = row["model"]
        ctx = row["context"]
        
        key = f"{ds}__{m}__{ctx}"
        err_curr = loaded_data[key]["errors"]
        
        # Compare with same model no_context
        no_ctx_key = f"{ds}__{m}__no_context"
        if ctx == "with_context" and no_ctx_key in loaded_data:
            err_no_ctx = loaded_data[no_ctx_key]["errors"]
            stat, p_val, stars = compute_wilcoxon_significance(err_curr, err_no_ctx)
            current_metrics_df.at[idx_row, "p_value_vs_no_context"] = p_val
            current_metrics_df.at[idx_row, "significance_vs_no_context"] = stars
        else:
            current_metrics_df.at[idx_row, "p_value_vs_no_context"] = np.nan
            current_metrics_df.at[idx_row, "significance_vs_no_context"] = "ns"
            
        # Compare against baseline: GPT-4o no_context (lowest baseline in prompt condition)
        baseline_key = f"{ds}__GPT-4o__no_context"
        if baseline_key in loaded_data and baseline_key != key:
            err_base = loaded_data[baseline_key]["errors"]
            stat_b, p_val_b, stars_b = compute_wilcoxon_significance(err_curr, err_base)
            current_metrics_df.at[idx_row, "p_value_vs_gpt4o_no_ctx"] = p_val_b
            current_metrics_df.at[idx_row, "significance_vs_gpt4o_no_ctx"] = stars_b
        else:
            current_metrics_df.at[idx_row, "p_value_vs_gpt4o_no_ctx"] = np.nan
            current_metrics_df.at[idx_row, "significance_vs_gpt4o_no_ctx"] = "ns"
            
    # Full Pairwise Wilcoxon Matrix across all model pairs per dataset
    for ds in ["french-difficulty", "ljl", "sentences"]:
        ds_keys = [k for k in loaded_data if loaded_data[k]["dataset"] == ds]
        for k1 in ds_keys:
            for k2 in ds_keys:
                if k1 != k2:
                    m1 = loaded_data[k1]["model"] + " (" + loaded_data[k1]["context"] + ")"
                    m2 = loaded_data[k2]["model"] + " (" + loaded_data[k2]["context"] + ")"
                    e1 = loaded_data[k1]["errors"]
                    e2 = loaded_data[k2]["errors"]
                    stat, p, st = compute_wilcoxon_significance(e1, e2)
                    wilcoxon_records.append({
                        "dataset": ds,
                        "model_A": m1,
                        "model_B": m2,
                        "mean_err_A": float(np.mean(e1)),
                        "mean_err_B": float(np.mean(e2)),
                        "diff_mean": float(np.mean(e1 - e2)),
                        "wilcoxon_W": stat,
                        "p_value": p,
                        "significance": st,
                    })
    wilcoxon_df = pd.DataFrame(wilcoxon_records)
    
    # Load historical metrics for baselines
    if historical_metrics_path and os.path.exists(historical_metrics_path):
        hist_df = load_historical_metrics(historical_metrics_path)
    else:
        hist_df = pd.DataFrame()
        
    # Consolidate DataFrame
    consolidated = pd.concat([current_metrics_df, hist_df], ignore_index=True)
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    csv_out_path = os.path.join(output_dir, "table2_with_significance.csv")
    consolidated.to_csv(csv_out_path, index=False)
    logger.info(f"Saved consolidated Table 2 CSV: {csv_out_path}")
    
    wilcoxon_out_path = os.path.join(output_dir, "wilcoxon_pairwise_tests.csv")
    wilcoxon_df.to_csv(wilcoxon_out_path, index=False)
    logger.info(f"Saved Wilcoxon pairwise tests CSV: {wilcoxon_out_path}")
    
    tex_out_path = os.path.join(output_dir, "difficulty_estimation_metrics.tex")
    generate_latex_table(consolidated, tex_out_path)
    
    # Also copy to results/dmkd_additions for cross-compatibility
    alt_out_dir = os.path.join("results", "dmkd_additions", "difficulty_estimation", "reviews")
    os.makedirs(alt_out_dir, exist_ok=True)
    consolidated.to_csv(os.path.join(alt_out_dir, "table2_with_significance.csv"), index=False)
    generate_latex_table(consolidated, os.path.join(alt_out_dir, "difficulty_estimation_metrics.tex"))
    
    return consolidated, wilcoxon_df


def main():
    parser = argparse.ArgumentParser(description="DMKD Difficulty Estimation Statistical Metrics Engine")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("results", "dmkd_additions", "difficulty_estimation", "gpt_evaluation"),
        help="Path to prediction CSVs directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("dmkd_additions", "difficulty_estimation", "reviews"),
        help="Output directory for generated CSV and LaTeX files",
    )
    parser.add_argument(
        "--historical-csv",
        type=str,
        default=os.path.join(
            "results", "dmkd_additions", "difficulty_estimation", "compute_metrics", "load_old_experiments_metrics", "historical_metrics.csv"
        ),
        help="Path to historical_metrics.csv",
    )
    parser.add_argument(
        "--b-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility (default: 42)",
    )
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        historical_metrics_path=args.historical_csv,
        b_bootstrap=args.b_bootstrap,
        seed=args.seed,
    )
    logger.info("Chantier 1 execution finished successfully with exit code 0.")


if __name__ == "__main__":
    main()
