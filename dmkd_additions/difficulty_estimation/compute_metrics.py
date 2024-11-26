import os
import glob
import pandas as pd
import logging
from typing import Dict, List, Tuple
from src.utilitaries.Logger import Logger

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
    # Normalize french difficulty variations
    if dataset.lower() in ["french", "french_difficulty", "french-difficulty"]:
        return "french-difficulty"
    return dataset.lower()


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
        return pd.DataFrame(columns=["dataset", "model", "context", "accuracy"])

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

            metrics_list.append(
                {
                    "dataset": _normalize_dataset_name(dataset),
                    "model": _normalize_model_name(model),
                    "context": context,
                    "accuracy": round(accuracy, 4),
                }
            )

            logger.debug(f"Accuracy for {model} on {dataset}: {accuracy:.4f}")

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
    """Load and format historical experiment results.

    Extracts metrics from:
    - OpenAI models (GPT-3.5, Davinci, Babbage)
    - Open source models (CamemBERT, Mistral)
    - Readability indices (ARI, FKGL, GFI)

    Returns:
        pd.DataFrame: Contains columns:
            - dataset: Name of the dataset (str)
            - model: Name of the model (str)
            - context: Whether system prompt was used (str)
            - accuracy: Prediction accuracy (float)
    """
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

    metrics_list = []

    try:
        # Load OpenAI results
        if os.path.exists(openai_path):
            openai_df = pd.read_csv(openai_path)
            for _, row in openai_df.iterrows():
                metrics_list.append(
                    {
                        "dataset": _normalize_dataset_name(row["dataset"]),
                        "model": _normalize_model_name(row["model"]),
                        "context": (
                            "with_context"
                            if row["context"] == "CECRL"
                            else "no_context"
                        ),
                        "accuracy": round(row["accuracy"], 4),
                    }
                )

        # Load CamemBERT results
        if os.path.exists(bert_path):
            bert_df = pd.read_csv(bert_path)
            for _, row in bert_df.iterrows():
                metrics_list.append(
                    {
                        "dataset": _normalize_dataset_name(row["Unnamed: 0"]),
                        "model": "CamemBERT",
                        "context": "no_context",
                        "accuracy": round(row["accuracy"], 4),
                    }
                )

        # Load Mistral results
        if os.path.exists(mistral_path):
            mistral_df = pd.read_csv(mistral_path)
            for _, row in mistral_df.iterrows():
                metrics_list.append(
                    {
                        "dataset": _normalize_dataset_name(row["dataset"]),
                        "model": "Mistral-7B",
                        "context": (
                            "with_context"
                            if row["context"] == "CECRL"
                            else "no_context"
                        ),
                        "accuracy": round(row["accuracy"], 4),
                    }
                )

        # Load readability indices results
        if os.path.exists(readability_path):
            readability_df = pd.read_csv(readability_path)
            for _, row in readability_df.iterrows():
                metrics_list.append(
                    {
                        "dataset": _normalize_dataset_name(row["dataset"]),
                        "model": row["model"].upper(),  # ARI, FKGL, GFI
                        "context": "no_context",
                        "accuracy": round(row["accuracy"], 4),
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
