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
W1_BERT = 0.5  # Fixed weight as discussed
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
        output_dir = os.path.join(RESULTS_DIR, "metrics")
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, "metrics.csv"))

        logger.info("Metrics computation completed")
        return results_df

    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise


def __get_cefr_level(outputs) -> str:
    """Convert model outputs to CEFR level.

    Uses the same approach as the original BERT model:
    1. Get probabilities for each level
    2. Return the most likely level
    """
    # Get probabilities from logits
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get predicted level index
    pred_idx = torch.argmax(predictions, dim=1)

    # Map index to CEFR level
    level_map = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C1", 5: "C2"}
    return level_map[pred_idx.item()]


def __compute_accuracy(
    df: pd.DataFrame,
    model: CamembertForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Calculate accuracy of simplification levels using BERT.

    Uses the approach from d_Metrics.ipynb:
    1. Get predictions for original and simplified texts
    2. Compare probability distributions
    """
    logger.debug("Computing accuracy")

    # Get predictions for all texts
    all_texts = pd.concat([df["Original"], df["simplified"]])
    inputs = tokenizer(
        all_texts.tolist(), padding=True, truncation=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probas = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Split predictions
    n = len(df)
    orig_probas = probas[:n, 1:]  # Remove A1 for original
    simp_probas = probas[n:, :-1]  # Remove C2 for simplified

    # Calculate accuracy using probability overlap
    accuracy = torch.sum(orig_probas * simp_probas, dim=1).mean().item()

    return accuracy


def __compute_similarity(
    df: pd.DataFrame,
    model: CamembertForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Calculate cosine similarity between original and simplified texts.

    Uses logits as sentence embeddings since we're using a classifier model.
    """
    logger.debug("Computing similarity")

    similarities = []
    for _, row in df.iterrows():
        # Get embeddings from logits
        inputs = tokenizer(
            [row["Original"], row["simplified"]],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use logits as embeddings
            embeddings = outputs.logits  # Shape: [2, num_classes]

        # Calculate cosine similarity between logits
        sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        similarities.append(sim.item())

    return np.mean(similarities)


def __compute_weighted_score(accuracy: float, similarity: float, w1: float) -> float:
    """Compute weighted score combining accuracy and similarity."""
    return (2 * w1 * accuracy * (1 - w1) * similarity) / (
        w1 * accuracy + (1 - w1) * similarity
    )


def load_old_experiments_metrics() -> pd.DataFrame:
    raise NotImplementedError("Not implemented")


def create_latex_table() -> Dict[str, str]:
    raise NotImplementedError("Not implemented")


if __name__ == "__main__":
    from src.utilitaries.openai_utils import setup_openai_token

    # Setup OpenAI token
    setup_openai_token()
    logger.info("Starting metrics computation")

    try:
        # Calculate metrics
        results_df = compute_metrics()

        # Display results
        logger.info("\nResults:")
        print(results_df)

    except Exception as e:
        logger.error(f"Error in metrics computation: {str(e)}")
        raise

    logger.info("Metrics computation completed")
