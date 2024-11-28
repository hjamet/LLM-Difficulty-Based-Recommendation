import os
import dill
import pandas as pd
import numpy as np
import torch
from typing import Dict
from transformers import CamembertForSequenceClassification, AutoTokenizer
from torch.nn.functional import cosine_similarity
from src.utilitaries.Logger import Logger
from huggingface_hub import snapshot_download
from tqdm import tqdm

logger = Logger(__name__)

# Constants
W1_BERT = 0.8  # Fixed weight as discussed
RESULTS_DIR = os.path.join("results", "dmkd_additions", "text_simplification")
DATASET = "french_difficulty"  # Dataset name for BERT model


def compute_metrics() -> pd.DataFrame:
    """Compute metrics for text simplification predictions."""
    logger.info("Starting metrics computation")

    try:
        # Download BERT model if needed
        pwd = os.getcwd()
        if not os.path.exists(os.path.join(pwd, DATASET)):
            logger.info("Downloading BERT model...")
            snapshot_download(
                repo_id=f"OloriBern/Lingorank_Bert_{DATASET}",
                local_dir=os.path.join(pwd, DATASET),
                revision="main",
                repo_type="model",
            )

        # Load tokenizer and label encoder
        logger.info("Loading tokenizer and label encoder")
        with open(
            os.path.join(
                os.path.join(pwd, DATASET),
                "train_camembert_tokenizer_label_encoder.pkl",
            ),
            "rb",
        ) as f:
            tokenizer, label_encoder = dill.load(f)

        # Load model
        logger.info("Loading BERT model")
        model = CamembertForSequenceClassification.from_pretrained(
            os.path.join(pwd, DATASET)
        )
        model.eval()

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Get predictions directory
        pred_dir = os.path.join(RESULTS_DIR, "gpt_evaluation")
        if not os.path.exists(pred_dir):
            raise FileNotFoundError(f"Predictions directory not found: {pred_dir}")

        # Load all prediction files
        prediction_files = [
            f for f in os.listdir(pred_dir) if f.endswith("_predictions.csv")
        ]
        if not prediction_files:
            raise FileNotFoundError("No prediction files found")

        # Process files with progress bar
        results = {}
        for file in tqdm(prediction_files, desc="Processing models"):
            logger.info(f"Processing {file}")

            # Load predictions
            df = pd.read_csv(os.path.join(pred_dir, file))

            # Calculate metrics
            accuracy = __compute_accuracy(df, model, tokenizer, device)
            similarity = __compute_similarity(df, model, tokenizer, device)

            # Combine metrics using weighted formula
            score = __compute_weighted_score(accuracy, similarity, W1_BERT)

            # Store results
            model_name = file.split("_predictions.csv")[0]
            results[model_name] = {
                "Accuracy": accuracy,
                "Similarity": similarity,
                "Score": score,
            }

        # Convert to DataFrame and save
        results_df = pd.DataFrame.from_dict(results, orient="index")
        output_dir = os.path.join(RESULTS_DIR, "compute_metrics")
        os.makedirs(output_dir, exist_ok=True)

        # Save as CSV
        csv_path = os.path.join(output_dir, "simplification_metrics.csv")
        results_df.to_csv(csv_path)
        logger.info(f"Saved metrics to {csv_path}")

        logger.info("Metrics computation completed")
        return results_df

    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise


def __compute_accuracy(
    df: pd.DataFrame,
    model: CamembertForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Calculate accuracy of simplification levels using BERT."""
    logger.debug("Computing accuracy")

    batch_size = 8  # Réduit la taille des lots pour éviter les OOM
    all_results = []

    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]

        # Get predictions for batch
        all_texts = pd.concat([batch_df["Original"], batch_df["simplified"]])
        inputs = tokenizer(
            all_texts.tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,  # Limite la longueur des séquences
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probas = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

        # Split predictions for this batch
        n = len(batch_df)
        comparison = {
            "Original": probas[:n, 1:],  # Remove A1 for original
            "Simplified": np.cumsum(
                probas[n:, :-1], axis=1
            ),  # Remove C2 and cumsum for simplified
        }

        # Calculate accuracy for batch
        results = (comparison["Original"] * comparison["Simplified"]).sum(axis=1)
        all_results.extend(results)

        # Clear CUDA cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.mean(all_results)


def __compute_similarity(
    df: pd.DataFrame,
    model: CamembertForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Calculate cosine similarity between original and simplified texts."""
    logger.debug("Computing similarity")

    batch_size = 8  # Réduit la taille des lots
    similarities = []

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]

        for _, row in batch_df.iterrows():
            # Get embeddings from logits
            inputs = tokenizer(
                [row["Original"], row["simplified"]],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,  # Limite la longueur des séquences
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.logits

            # Calculate cosine similarity
            sim = cosine_similarity(
                embeddings[0].unsqueeze(0).cpu(), embeddings[1].unsqueeze(0).cpu()
            )
            similarities.append(sim.item())

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return np.mean(similarities)


def __compute_weighted_score(accuracy: float, similarity: float, w1: float) -> float:
    """Compute weighted score combining accuracy and similarity."""
    return (2 * w1 * accuracy * (1 - w1) * similarity) / (
        w1 * accuracy + (1 - w1) * similarity
    )


def load_old_experiments_metrics() -> pd.DataFrame:
    """Load metrics from previous experiments (hardcoded from paper table).

    Returns:
        pd.DataFrame: DataFrame containing metrics for each model with columns:
            - Accuracy: Simplification accuracy score
            - Similarity: Semantic similarity score
            - Score: Combined weighted score (w1=0.8)
            - Fine_tuned: Whether the model was fine-tuned
    """
    # Hardcoded values from the paper's LaTeX table (without scores)
    data = {
        "Mistral_ft": {
            "Accuracy": 0.59,
            "Similarity": 0.91,
            "Fine_tuned": True,
        },
        "GPT-3.5_ft": {
            "Accuracy": 0.57,
            "Similarity": 0.91,
            "Fine_tuned": True,
        },
        "GPT-3.5": {
            "Accuracy": 0.53,
            "Similarity": 0.93,
            "Fine_tuned": False,
        },
        "GPT-4": {
            "Accuracy": 0.51,
            "Similarity": 0.93,
            "Fine_tuned": False,
        },
        "Mistral": {
            "Accuracy": 0.47,
            "Similarity": 0.93,
            "Fine_tuned": False,
        },
        "GPT-4o": {
            "Accuracy": 0.46,
            "Similarity": 0.89,
            "Fine_tuned": False,
        },
        "Davinci_ft": {
            "Accuracy": 0.44,
            "Similarity": 0.83,
            "Fine_tuned": True,
        },
    }

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")

    # Calculate scores with w1=0.8
    for idx in df.index:
        df.loc[idx, "Score"] = __compute_weighted_score(
            df.loc[idx, "Accuracy"], df.loc[idx, "Similarity"], W1_BERT
        )

    return df


def create_latex_table() -> Dict[str, str]:
    """Create LaTeX table comparing performance metrics across all models."""
    logger.info("Creating LaTeX table")

    # Load all results
    historical_df = load_old_experiments_metrics()
    current_df = compute_metrics()

    try:
        # Standardize model names
        def standardize_model_name(name):
            if isinstance(name, str):
                if "GPT-4O-MINI" in name.upper():
                    return "GPT-4o-Mini"
                elif "GPT-4O" in name.upper():
                    return "GPT-4o"
                name = name.replace("test_set_ft_", "")
                name = name.split("_")[0]  # Remove date and other suffixes
            return name

        current_df.index = current_df.index.map(standardize_model_name)

        # Set Fine_tuned to True for current results (they are all fine-tuned)
        current_df["Fine_tuned"] = True

        # Remove duplicates, keeping the row with the highest Score
        all_results = pd.concat([historical_df, current_df])
        all_results = all_results.loc[all_results.groupby(level=0)["Score"].idxmax()]
        all_results = all_results.sort_values("Score", ascending=False)

        # Debug log
        logger.debug("All results after deduplication:")
        logger.debug(all_results)

        # Create detailed caption
        caption = (
            "Text simplification metrics for all models. "
            "For each model, we report three metrics: "
            "Simplification Accuracy (Acc.), Semantic Similarity (Sim.), and their Weighted Score (Score). "
            "All metrics range from 0 to 1, higher is better. "
            "Fine-tuned (FT) indicates whether the model was fine-tuned on our dataset (\\checkmark) "
            "or used as-is (-). "
            "Models are sorted by their weighted score. "
            "\\textbf{\\underline{Bold and underlined}} values indicate the best score for each metric."
        )

        # Create LaTeX table
        latex = "\\begin{table}[!h]\n"
        latex += "    \\centering\n"
        latex += "    \\small\n"
        latex += "    \\begin{tabular}{lcrrr}\n"  # Changed to right-align numbers
        latex += "        \\toprule\n"
        latex += "        \\textbf{Model} & \\textbf{FT} & \\textbf{Acc.} & \\textbf{Sim.} & \\textbf{Score} \\\\\n"
        latex += "        \\midrule\n"

        # Get best values for highlighting
        best_acc = all_results["Accuracy"].max()
        best_sim = all_results["Similarity"].max()
        best_score = all_results["Score"].max()

        # Add rows
        current_model = None
        footnoted_models = {
            "GPT-3.5",
            "GPT-4",
            "GPT-4o",
            "GPT-4o-Mini",
            "Mistral",
            "Davinci",
        }

        for model in all_results.index:
            # Add spacing before new model (except first)
            if current_model is not None:
                latex += "        \\\\[2pt]\n"
            current_model = model

            row = all_results.loc[model]

            # Format model name
            model_str = str(model).replace("_", "\\_")  # Escape underscores
            model_str = f"\\textbf{{{model_str}}}"
            if str(model) in footnoted_models:  # Convert to string for comparison
                model_str += "$^3$"  # Changed to math mode

            # Format fine-tuning indicator (handle NaN as False)
            ft_value = row["Fine_tuned"]
            if isinstance(ft_value, pd.Series):
                ft_value = ft_value.iloc[0]  # Take first value if Series
            ft_str = "\\checkmark" if pd.notna(ft_value) and ft_value else "-"

            # Format metrics with colors
            metrics = []
            for value, best_value, col in [
                (row["Accuracy"], best_acc, "Accuracy"),
                (row["Similarity"], best_sim, "Similarity"),
                (row["Score"], best_score, "Score"),
            ]:
                # Handle Series
                if isinstance(value, pd.Series):
                    value = value.iloc[0]

                # Calculate color intensity (blue scale)
                if col == "Score":
                    # Pour Score, plus foncé = meilleur (valeur haute)
                    intensity = value
                else:
                    # Pour les autres métriques, normaliser entre min et max
                    col_min = all_results[col].min()
                    col_max = all_results[col].max()
                    intensity = (value - col_min) / (col_max - col_min)

                r = int(255 * (1 - intensity))
                g = int(255 * (1 - intensity * 0.8))
                b = int(255 * (1 - intensity * 0.4))
                html_color = f"{r:02x}{g:02x}{b:02x}"
                text_color = "white" if intensity > 0.5 else "black"

                # Format value
                value_str = f"{value:.2f}"
                if abs(value - best_value) < 0.0001:
                    value_str = f"\\textbf{{\\underline{{{value_str}}}}}"

                cell = f"\\cellcolor[HTML]{{{html_color}}}\\textcolor{{{text_color}}}{{{value_str}}}"
                metrics.append(cell)

            # Add row
            latex += f"        {model_str} & {ft_str} & {' & '.join(metrics)} \\\\\n"

        latex += "        \\bottomrule\n"
        latex += "    \\end{tabular}\n"
        latex += f"    \\caption{{{caption}}}\n"
        latex += "    \\label{tab:text_simplification_metrics}\n"
        latex += "\\end{table}\n\n"

        # Add footnote
        latex += "\\footnotetext[3]{%\n"
        latex += "    In this figure, ``GPT-3.5'' corresponds to \\texttt{gpt-3.5-turbo-1106}, "
        latex += "``GPT-4o'' and ``GPT-4o-Mini'' to our fine-tuned models, "
        latex += "``Mistral'' to \\texttt{Mistral-7B}, "
        latex += "and ``Davinci'' to \\texttt{davinci-002}.}\n"

        # Save table
        output_dir = os.path.join(RESULTS_DIR, "compute_metrics", "latex")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "text_simplification_metrics.tex")
        with open(output_path, "w") as f:
            f.write(latex)
            logger.info(f"Saved LaTeX table to {output_path}")

        return {"combined": latex}

    except Exception as e:
        logger.error(f"Error creating LaTeX table: {str(e)}")
        raise


if __name__ == "__main__":
    logger.info("Starting metrics computation")

    try:
        # Create and save LaTeX table
        logger.info("\nCreating LaTeX table...")
        latex_table = create_latex_table()
        logger.info("LaTeX table created successfully")

    except Exception as e:
        logger.error(f"Error in metrics computation: {str(e)}")
        raise

    logger.info("Metrics computation completed")
