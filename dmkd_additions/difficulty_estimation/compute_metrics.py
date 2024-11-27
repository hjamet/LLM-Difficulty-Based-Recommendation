import os
import glob
import pandas as pd
import logging
from typing import Dict, List, Tuple
from src.utilitaries.Logger import Logger
from sklearn.preprocessing import LabelEncoder

# Initialize logger
logger = Logger(__name__)


def _normalize_model_name(model: str) -> str:
    """Normalize model names to a consistent format.

    Args:
        model (str): Raw model name

    Returns:
        str: Normalized model name
    """
    # Handle fine-tuned models
    if "ft_" in model:
        if "gpt-4o-mini" in model:
            return "GPT-4o-mini"
        elif "gpt-4o" in model:
            return "GPT-4o"

    # Normalize OpenAI model names
    model_mapping = {
        "gpt-3.5-turbo-1106": "GPT-3.5",
        "davinci-002": "Davinci",
        "babbage-002": "Babbage",
    }

    return model_mapping.get(model, model)


def _normalize_dataset_name(dataset: str) -> str:
    """Normalize dataset names to a consistent format.

    Args:
        dataset (str): Raw dataset name

    Returns:
        str: Normalized dataset name
    """
    # Convert to string if needed
    dataset = str(dataset)

    # Normalize french difficulty variations
    if dataset.lower() in ["french", "french_difficulty", "french-difficulty"]:
        return "french-difficulty"
    return dataset.lower()


def _normalize_context(context: str) -> str:
    """Normalize context names to a consistent format.

    Args:
        context (str): Raw context name (CECRL, empty, with_system, no_system)

    Returns:
        str: Normalized context name (with_context, no_context)
    """
    # Convert to string and lowercase for consistent comparison
    context = str(context).lower()

    # Map various context indicators to standardized values
    with_context_indicators = ["with_system", "with_context", "cecrl", "with_prompt"]

    # Log the context normalization
    logger.debug(f"Normalizing context: {context}")

    # Return normalized value
    normalized = (
        "with_context"
        if any(indicator in context for indicator in with_context_indicators)
        else "no_context"
    )
    logger.debug(f"Normalized context: {normalized}")

    return normalized


def compute_metrics() -> pd.DataFrame:
    """Compute accuracy metrics for all prediction files.

    This function:
    1. Scans the results directory for prediction files
    2. Loads each file and computes accuracy
    3. Returns a DataFrame with model performance metrics

    Returns:
        pd.DataFrame: Contains columns:
            - dataset: Name of the dataset (str)
            - model: Name of the model (str)
            - context: Whether system prompt was used (str)
            - accuracy: Prediction accuracy (float)
    """
    logger.info("Starting metrics computation")

    # Define results directory
    results_dir = os.path.join(
        "results", "dmkd_additions", "difficulty_estimation", "gpt_evaluation"
    )

    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    # Find all prediction files
    pattern = os.path.join(results_dir, "*_predictions.csv")
    prediction_files = glob.glob(pattern)

    if not prediction_files:
        logger.warning("No prediction files found")
        return pd.DataFrame(
            columns=["dataset", "model", "context", "accuracy", "pairwise_mismatch"]
        )

    metrics_list = []

    # Process each prediction file
    for file_path in prediction_files:
        try:
            # Parse filename to get metadata
            filename = os.path.basename(file_path)
            dataset, model, context = _parse_filename(filename)

            # Load predictions
            logger.debug(f"Processing file: {filename}")
            df = pd.read_csv(file_path)

            # Compute accuracy
            if "predictions" not in df.columns or "difficulty" not in df.columns:
                logger.warning(f"Invalid file format: {filename}")
                continue

            # Normalize labels to uppercase for comparison
            true_labels = df["difficulty"].str.upper()
            pred_labels = df["predictions"].str.upper()

            accuracy = (pred_labels == true_labels).mean()

            # Compute pairwise mismatch
            pw_mismatch = pairwise_mismatch(pred_labels, true_labels)

            metrics_list.append(
                {
                    "dataset": _normalize_dataset_name(dataset),
                    "model": _normalize_model_name(model),
                    "context": _normalize_context(context),
                    "accuracy": round(accuracy, 4),
                    "pairwise_mismatch": round(pw_mismatch, 4),
                }
            )

            logger.debug(f"Accuracy for {model} on {dataset}: {accuracy:.4f}")
            logger.debug(
                f"Pairwise mismatch for {model} on {dataset}: {pw_mismatch:.4f}"
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    # Create DataFrame from results
    metrics_df = pd.DataFrame(metrics_list)

    # Sort by dataset and accuracy
    metrics_df = metrics_df.sort_values(
        by=["dataset", "accuracy"], ascending=[True, False]
    )

    # Save results
    output_dir = os.path.join(
        "results",
        "dmkd_additions",
        "difficulty_estimation",
        "compute_metrics",
        "compute_metrics",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "current_metrics.csv")
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Saved current metrics to {output_path}")

    logger.info("Metrics computation completed")
    return metrics_df


def _parse_filename(filename: str) -> Tuple[str, str, str]:
    """Parse prediction filename to extract metadata."""
    logger.debug(f"Parsing filename: {filename}")

    # Extract dataset (first part before _ft_)
    dataset = filename.split("_ft_")[0]

    # Extract model name
    if "_ft_" in filename:
        # Pour les modèles fine-tunés
        model_part = filename.split("_ft_")[1].split("_university")[0]
        # Extraire la partie avant la date
        model = model_part.split("-20")[0]
        # Standardiser le nom
        if model.upper().startswith("GPT-4O"):
            if model.upper().endswith("MINI"):
                model = "GPT-4o-mini"
            else:
                model = "GPT-4o"
    else:
        # Pour les autres modèles (cas non fine-tuné)
        model = filename.split("_")[1]

    # Extract context
    if "_with_system" in filename:
        context = "with_context"
    elif "_no_system" in filename:
        context = "no_context"
    else:
        context = "no_context"

    logger.debug(f"Parsed: dataset={dataset}, model={model}, context={context}")
    return dataset, model, context


def load_old_experiments_metrics() -> pd.DataFrame:
    """Load and format historical experiment results."""
    logger.info("Loading historical experiment results")

    # Define paths
    base_dir = "results/difficulty_estimation"
    openai_path = os.path.join(base_dir, "OpenAiEvaluation/metrics.csv")
    bert_path = os.path.join(base_dir, "OpenSourceModelsEvaluation/bert_metrics.csv")
    mistral_path = os.path.join(
        base_dir, "OpenSourceModelsEvaluation/mistral_metrics.csv"
    )
    readability_path = os.path.join(
        base_dir, "PairwiseMismatch/readability_index_classification_metrics.csv"
    )

    # Define pairwise mismatch paths
    openai_pw_path = os.path.join(
        base_dir, "PairwiseMismatch/openai_pairwise_mismatch.csv"
    )
    bert_pw_path = os.path.join(base_dir, "PairwiseMismatch/bert_pairwise_mismatch.csv")
    mistral_pw_path = os.path.join(
        base_dir, "PairwiseMismatch/mistral_pairwise_mismatch.csv"
    )
    readability_pw_path = os.path.join(
        base_dir, "PairwiseMismatch/readability_index_pairwise_mismatch.csv"
    )

    metrics_list = []

    try:
        # Load and normalize pairwise mismatch results
        if os.path.exists(openai_pw_path):
            openai_pw = pd.read_csv(openai_pw_path)
            openai_pw["dataset"] = (
                openai_pw["dataset"].astype(str).apply(_normalize_dataset_name)
            )
            openai_pw["model"] = (
                openai_pw["model"].astype(str).apply(_normalize_model_name)
            )
            openai_pw["context"] = (
                openai_pw["context"].astype(str).apply(_normalize_context)
            )
        else:
            logger.warning(f"OpenAI pairwise mismatch file not found: {openai_pw_path}")
            openai_pw = None

        if os.path.exists(bert_pw_path):
            bert_pw = pd.read_csv(bert_pw_path, index_col=0)
            bert_pw.index = bert_pw.index.astype(str).map(_normalize_dataset_name)
        else:
            logger.warning(f"BERT pairwise mismatch file not found: {bert_pw_path}")
            bert_pw = None

        if os.path.exists(mistral_pw_path):
            mistral_pw = pd.read_csv(mistral_pw_path, index_col=0)
            # Créer un nouveau DataFrame avec les index normalisés
            new_index = []
            for idx in mistral_pw.index:
                parts = idx.split("_")
                dataset = _normalize_dataset_name(parts[0])
                # Si le dataset est french_difficulty, on prend le contexte de la dernière partie
                if dataset == "french-difficulty":
                    context = parts[-1]
                else:
                    context = parts[1] if len(parts) > 1 else "no-context"
                new_index.append(f"{dataset}_{context}")
            mistral_pw.index = new_index
        else:
            logger.warning(
                f"Mistral pairwise mismatch file not found: {mistral_pw_path}"
            )
            mistral_pw = None

        if os.path.exists(readability_pw_path):
            readability_pw = pd.read_csv(readability_pw_path)
            readability_pw["dataset"] = (
                readability_pw["dataset"].astype(str).apply(_normalize_dataset_name)
            )
            readability_pw["model"] = readability_pw["model"].apply(
                lambda x: str.upper(str(x))
            )
        else:
            logger.warning(
                f"Readability pairwise mismatch file not found: {readability_pw_path}"
            )
            readability_pw = None

        # Load OpenAI results
        if os.path.exists(openai_path):
            openai_df = pd.read_csv(openai_path)
            for _, row in openai_df.iterrows():
                dataset = _normalize_dataset_name(row["dataset"])
                model = _normalize_model_name(row["model"])
                context = _normalize_context(row["context"])

                # Get pairwise mismatch if available
                pw_mismatch = None
                if openai_pw is not None:
                    pw_row = openai_pw[
                        (openai_pw["dataset"] == dataset)
                        & (openai_pw["model"] == model)
                        & (openai_pw["context"] == context)
                    ]
                    if not pw_row.empty:
                        pw_mismatch = pw_row["pairwise_mismatch"].iloc[0]

                metrics_list.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "context": context,
                        "accuracy": round(row["accuracy"], 4),
                        "pairwise_mismatch": pw_mismatch,
                    }
                )

        # Load CamemBERT results
        if os.path.exists(bert_path):
            bert_df = pd.read_csv(bert_path)
            for _, row in bert_df.iterrows():
                dataset = _normalize_dataset_name(row["Unnamed: 0"])

                # Get pairwise mismatch if available
                pw_mismatch = None
                if bert_pw is not None:
                    pw_row = bert_pw[bert_pw.index == dataset]
                    if not pw_row.empty:
                        pw_mismatch = pw_row["Pairwise mismatch"].iloc[0]

                metrics_list.append(
                    {
                        "dataset": dataset,
                        "model": "CamemBERT",
                        "context": "no_context",
                        "accuracy": round(row["accuracy"], 4),
                        "pairwise_mismatch": pw_mismatch,
                    }
                )

        # Load Mistral results
        if os.path.exists(mistral_path):
            mistral_df = pd.read_csv(mistral_path)
            for _, row in mistral_df.iterrows():
                dataset = _normalize_dataset_name(row["dataset"])
                context = "with_context" if row["context"] == "CECRL" else "no_context"

                # Get pairwise mismatch if available
                pw_mismatch = None
                if mistral_pw is not None:
                    key = f"{dataset}_{row['context']}"
                    logger.debug(f"Looking for Mistral match with key: {key}")
                    logger.debug(f"Available Mistral keys: {mistral_pw.index.tolist()}")
                    pw_row = mistral_pw[mistral_pw.index == key]
                    if not pw_row.empty:
                        pw_mismatch = pw_row["Pairwise mismatch"].iloc[0]
                        logger.debug(
                            f"Found Mistral match with pairwise_mismatch: {pw_mismatch}"
                        )
                    else:
                        logger.debug("No Mistral match found")

                metrics_list.append(
                    {
                        "dataset": dataset,
                        "model": "Mistral-7B",
                        "context": context,
                        "accuracy": round(row["accuracy"], 4),
                        "pairwise_mismatch": pw_mismatch,
                    }
                )

        # Load readability indices results
        if os.path.exists(readability_path):
            readability_df = pd.read_csv(readability_path)
            for _, row in readability_df.iterrows():
                dataset = _normalize_dataset_name(row["dataset"])
                model = str.upper(str(row["model"]))

                # Get pairwise mismatch if available
                pw_mismatch = None
                if readability_pw is not None:
                    pw_row = readability_pw[
                        (readability_pw["dataset"] == dataset)
                        & (readability_pw["model"] == model)
                    ]
                    if not pw_row.empty:
                        pw_mismatch = pw_row["pairwise_mismatch"].iloc[0]

                metrics_list.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "context": "no_context",
                        "accuracy": round(row["accuracy"], 4),
                        "pairwise_mismatch": pw_mismatch,
                    }
                )

        logger.info("Historical results loaded successfully")

    except Exception as e:
        logger.error(f"Error loading historical results: {str(e)}")
        raise

    # Create DataFrame and sort results
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.sort_values(
        by=["dataset", "accuracy"], ascending=[True, False]
    )

    # Save results
    output_dir = os.path.join(
        "results",
        "dmkd_additions",
        "difficulty_estimation",
        "compute_metrics",
        "load_old_experiments_metrics",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "historical_metrics.csv")
    metrics_df.to_csv(output_path, index=False)
    logger.info(f"Saved historical metrics to {output_path}")

    return metrics_df


def pairwise_mismatch(y_pred: pd.Series, y_real: pd.Series) -> float:
    """Compute pairwise mismatch score between predictions and ground truth.

    Args:
        y_pred (pd.Series): Predicted labels
        y_real (pd.Series): True labels

    Returns:
        float: Pairwise mismatch score
    """
    predictions = pd.DataFrame({"pred": y_pred, "real": y_real})

    # Sort alphabetically
    predictions = predictions.sort_values(by=["real", "pred"])

    # Transform labels to integers while conserving the alphabetical order
    predictions["pred"] = predictions["pred"].astype("category")
    predictions["real"] = predictions["real"].astype("category")

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode labels
    unique_labels = pd.unique(predictions[["pred", "real"]].values.ravel("K"))
    label_encoder.fit(unique_labels)
    predictions["pred"] = label_encoder.transform(predictions["pred"])
    predictions["real"] = label_encoder.transform(predictions["real"])

    # Compute the pairwise mismatch
    mismatch = (
        predictions["pred"].expanding().apply(lambda s: (s.iloc[-1] - s < 0).sum())
        + predictions["pred"][::-1]
        .expanding()
        .apply(lambda s: (s.iloc[-1] - s > 0).sum())[::-1]
    )

    return mismatch.mean()


def create_latex_table() -> Dict[str, str]:
    """Create LaTeX table comparing performance metrics across all models."""
    logger.info("Creating LaTeX table")

    try:
        # Load all results
        historical_df = load_old_experiments_metrics()
        current_df = compute_metrics()

        # Debug logs pour voir les données avant fusion
        logger.debug("Historical data:")
        logger.debug(
            historical_df[
                ["model", "context", "dataset", "accuracy", "pairwise_mismatch"]
            ]
        )
        logger.debug("\nCurrent data:")
        logger.debug(
            current_df[["model", "context", "dataset", "accuracy", "pairwise_mismatch"]]
        )

        # Standardiser les noms de modèles avant la fusion
        def standardize_model_name(x):
            if "GPT-4O-MINI" in x.upper():
                return "GPT-4o-mini"
            elif "GPT-4O" in x.upper():
                return "GPT-4o"
            return x

        historical_df["model"] = historical_df["model"].apply(standardize_model_name)
        current_df["model"] = current_df["model"].apply(standardize_model_name)

        # Combine results and remove duplicates
        all_results = pd.concat([historical_df, current_df], ignore_index=True)

        # Garder la meilleure performance pour chaque combinaison unique de dataset/model/context
        best_results = []
        for (dataset, model, context), group in all_results.groupby(
            ["dataset", "model", "context"]
        ):
            # Pour chaque groupe, prendre la ligne avec la meilleure accuracy
            best_row = group.loc[group["accuracy"].idxmax()]
            best_results.append(best_row)

        all_results = pd.DataFrame(best_results)

        # Debug log après regroupement
        logger.debug("\nAll results after grouping:")
        logger.debug(
            all_results[
                ["model", "context", "dataset", "accuracy", "pairwise_mismatch"]
            ]
        )

        # Debug: afficher les valeurs uniques de contexte avant standardisation
        logger.debug("Unique context values before mapping:")
        logger.debug(all_results["context"].unique())

        # Standardize context values before pivot
        context_mapping = {
            "with_context": "with_context",
            "with_system": "with_context",
            "CECRL": "with_context",
            "no_context": "no_context",
            "no_system": "no_context",
            "-": "no_context",
            None: "no_context",  # Ajout pour gérer les valeurs None
            "": "no_context",  # Ajout pour gérer les chaînes vides
        }

        # Appliquer le mapping avec un message de debug pour les valeurs non mappées
        def map_context(x):
            if x not in context_mapping:
                logger.warning(f"Unmapped context value: '{x}'")
                return "no_context"
            return context_mapping[x]

        all_results["context"] = all_results["context"].apply(map_context)

        # Debug: afficher les valeurs uniques après standardisation
        logger.debug("Unique context values after mapping:")
        logger.debug(all_results["context"].unique())

        # Debug: afficher les lignes pour GPT-4o et GPT-4o-mini
        logger.debug("GPT-4o and GPT-4o-mini rows:")
        logger.debug(all_results[all_results["model"].isin(["GPT-4o", "GPT-4o-mini"])])

        # Create one table with all datasets
        pivoted_acc = all_results.pivot_table(
            index=["model", "context"],
            columns="dataset",
            values="accuracy",
            aggfunc="first",
        ).round(2)

        pivoted_pm = all_results.pivot_table(
            index=["model", "context"],
            columns="dataset",
            values="pairwise_mismatch",
            aggfunc="first",
        ).round(1)

        # Rename columns
        column_mapping = {
            "ljl": "LjL",
            "sentences": "Sentences Internet",
            "french-difficulty": "Sentences Books",
        }
        pivoted_acc.columns = pivoted_acc.columns.map(column_mapping)
        pivoted_pm.columns = pivoted_pm.columns.map(column_mapping)

        # Sort by average pairwise mismatch
        avg_mismatch = pivoted_pm.mean(axis=1)
        sort_order = avg_mismatch.sort_values().index

        # Create detailed caption
        caption = (
            "Difficulty estimation metrics for all datasets. "
            "For each model and dataset, we report two metrics: "
            "accuracy (Acc.) and pairwise mismatch (PM). "
            "Accuracy ranges from 0 to 1, higher is better. "
            "PM measures the violation of the difficulty ordering between pairs of sentences "
            "(e.g., an A1 sentence predicted as more difficult than a C2 sentence). "
            "A PM score of 0 indicates perfect ordering, while higher scores indicate more ordering errors. "
            "System Prompt indicates whether the model was given explicit CEFR level descriptions (\\checkmark) "
            "or not (-). "
            "Models are sorted by their average PM score across all datasets. "
            "\\textbf{\\underline{Bold and underlined}} values indicate the best score for each metric and dataset."
        )

        # Create LaTeX table
        latex = "\\begin{table}[!h]\n"
        latex += "    \\centering\n"
        latex += "    \\small\n"
        latex += "    \\setlength{\\tabcolsep}{4pt}\n"
        latex += "    \\begin{tabular}{lccccccc}\n"
        latex += "        \\toprule\n"
        latex += "        & & \\multicolumn{2}{c}{\\textbf{Sentences Books}} & \\multicolumn{2}{c}{\\textbf{LjL}} & \\multicolumn{2}{c}{\\textbf{Sentences Internet}} \\\\\n"
        latex += "        \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n"
        latex += "        \\textbf{Model} & \\textbf{System Prompt} & Acc. & PM & Acc. & PM & Acc. & PM \\\\\n"
        latex += "        \\midrule\n"

        # Content
        current_model = None
        footnoted_models = {
            "GPT-3.5",
            "CamemBERT",
            "Mistral-7B",
            "Davinci",
            "Babbage",
            "GPT-4o",
            "GPT-4o-mini",
        }

        # Group rows by model to handle spacing
        model_rows = {}
        for model, context in sort_order:
            if model not in model_rows:
                model_rows[model] = []
            model_rows[model].append(context)

        # Debug log des modèles et leurs contextes
        logger.debug("\nModel rows:")
        for model, contexts in model_rows.items():
            logger.debug(f"{model}: {contexts}")

        # Generate table content
        for model in model_rows:
            # Add spacing before new model (except first)
            if current_model is not None:
                latex += "        \\\\[2pt]\n"
            current_model = model

            # Debug log pour chaque modèle
            logger.debug(f"\nProcessing model: {model}")
            logger.debug(f"Contexts: {model_rows[model]}")

            # Add each context variant for the model
            for i, context in enumerate(sorted(model_rows[model], reverse=True)):
                # Model name only on first row
                if i == 0:
                    model_str = f"\\textbf{{{model}}}"
                    if model in footnoted_models:
                        model_str += "\\textsuperscript{3}"
                else:
                    model_str = ""

                # Context
                context_str = "\\checkmark" if context == "with_context" else "-"

                # Values
                values = []
                for dataset in ["Sentences Books", "LjL", "Sentences Internet"]:
                    acc = pivoted_acc.loc[(model, context), dataset]
                    pm = pivoted_pm.loc[(model, context), dataset]

                    if pd.isna(acc):
                        values.extend(["-", "-"])
                    else:
                        # Get best values for this dataset
                        best_acc = pivoted_acc[dataset].max()
                        best_pm = pivoted_pm[dataset].min()  # Lower is better for PM

                        # Accuracy cell - bleu, plus foncé = meilleur (valeur haute)
                        intensity = acc
                        r = int(255 * (1 - intensity))
                        g = int(255 * (1 - intensity * 0.8))
                        b = int(255 * (1 - intensity * 0.4))
                        html_color = f"{r:02x}{g:02x}{b:02x}"
                        text_color = "white" if intensity > 0.5 else "black"

                        # Add bold and underline for best accuracy
                        acc_str = f"{acc:.2f}"
                        if (
                            abs(acc - best_acc) < 0.0001
                        ):  # Using small epsilon for float comparison
                            acc_str = f"\\textbf{{\\underline{{{acc_str}}}}}"
                        acc_cell = f"\\cellcolor[HTML]{{{html_color}}}\\textcolor{{{text_color}}}{{{acc_str}}}"

                        # PM cell - même bleu, plus foncé = meilleur (valeur basse)
                        pm_min = pivoted_pm.min().min()
                        pm_max = pivoted_pm.max().max()
                        pm_normalized = (pm - pm_min) / (pm_max - pm_min)
                        intensity = 1 - pm_normalized

                        r = int(255 * (1 - intensity))
                        g = int(255 * (1 - intensity * 0.8))
                        b = int(255 * (1 - intensity * 0.4))
                        html_color = f"{r:02x}{g:02x}{b:02x}"
                        text_color = "white" if intensity > 0.5 else "black"

                        # Add bold and underline for best PM
                        pm_str = f"{pm:.1f}"
                        if (
                            abs(pm - best_pm) < 0.0001
                        ):  # Using small epsilon for float comparison
                            pm_str = f"\\textbf{{\\underline{{{pm_str}}}}}"
                        pm_cell = f"\\cellcolor[HTML]{{{html_color}}}\\textcolor{{{text_color}}}{{{pm_str}}}"

                        values.extend([acc_cell, pm_cell])

                # Add row
                latex += (
                    f"        {model_str} & {context_str} & {' & '.join(values)} \\\\\n"
                )

        latex += "        \\bottomrule\n"
        latex += "    \\end{tabular}\n"
        latex += f"    \\caption{{{caption}}}\n"
        latex += "    \\label{tab:difficulty_estimation_metrics}\n"
        latex += "\\end{table}\n\n"

        # Add footnote
        latex += "\\footnotetext[3]{%\n"
        latex += "    In this figure, ``GPT-3.5'' corresponds to \\texttt{gpt-3.5-turbo-1106}, "
        latex += "``GPT-4o'' and ``GPT-4o-mini'' to our fine-tuned models, "
        latex += "``BERT'' to \\texttt{camembert-base}, "
        latex += "``Mistral'' to \\texttt{Mistral-7B}, "
        latex += "and ``Davinci'' to \\texttt{davinci-002}."
        latex += "}\n"

        # Save table
        output_dir = os.path.join(
            "results",
            "dmkd_additions",
            "difficulty_estimation",
            "compute_metrics",
            "create_latex_table",
        )
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "difficulty_estimation_metrics.tex")
        with open(output_path, "w") as f:
            f.write(latex)
        logger.info(f"Saved LaTeX table to {output_path}")

        return {"combined": latex}

    except Exception as e:
        logger.error(f"Error creating LaTeX table: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Create LaTeX tables
        logger.info("Creating LaTeX tables")
        tables = create_latex_table()

        logger.info("All tables saved successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
