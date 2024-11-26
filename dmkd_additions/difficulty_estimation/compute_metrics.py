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
    if context in ["CECRL", "with_system"]:
        return "with_context"
    return "no_context"


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

    logger.info("Metrics computation completed")
    return metrics_df


def _parse_filename(filename: str) -> Tuple[str, str, str]:
    """Parse prediction filename to extract metadata.

    Args:
        filename (str): Format: {dataset}_{model}_[context]_predictions.csv

    Returns:
        Tuple[str, str, str]: (dataset, model, context)
    """
    # Remove _predictions.csv suffix
    base = filename.replace("_predictions.csv", "")

    # Split remaining parts
    parts = base.split("_")

    # Extract dataset (first part)
    dataset = parts[0]

    # Check for context
    context = "with_context" if "with_system" in filename else "no_context"

    # Extract model name (everything between dataset and context/predictions)
    model_start = len(dataset) + 1
    model_end = filename.find(
        "_with_system" if context == "with_context" else "_no_system"
    )
    if model_end == -1:
        model_end = filename.find("_predictions.csv")
    model = filename[model_start:model_end]

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


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Load historical results
        logger.info("Loading historical metrics")
        historical_df = load_old_experiments_metrics()

        # Compute current metrics
        logger.info("Computing current metrics")
        current_df = compute_metrics()

        # Combine results
        all_results = pd.concat([historical_df, current_df], ignore_index=True)
        all_results = all_results.sort_values(
            by=["dataset", "accuracy"], ascending=[True, False]
        )

        # Display results
        if not all_results.empty:
            print("\nResults by dataset:")
            for dataset in all_results["dataset"].unique():
                print(f"\n{dataset.upper()}:")
                dataset_results = all_results[all_results["dataset"] == dataset]
                print(dataset_results.to_string(index=False))
        else:
            print("No results found")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
