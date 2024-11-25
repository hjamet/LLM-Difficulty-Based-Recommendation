import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List, Optional
from src.utilitaries.Logger import Logger
from dmkd_additions.difficulty_estimation.gpt_fine_tuning import DEFAULT_SYSTEM_PROMPT
import logging

# Disable OpenAI logging
logging.getLogger("openai").setLevel(logging.WARNING)

# Initialize logger
logger = Logger(__name__)


def validate_prediction(pred: str, dataset_name: str) -> Optional[str]:
    """
    Validate prediction based on dataset type.

    Args:
        pred (str): The prediction to validate
        dataset_name (str): Name of the dataset being evaluated

    Returns:
        Optional[str]: Validated prediction or None if invalid
    """
    if not pred:
        return None

    pred = pred.strip().upper()

    # Define valid levels for each dataset
    VALID_LEVELS: Dict[str, set] = {
        "ljl": {f"LEVEL{i}" for i in range(1, 5)},
        "french_difficulty": {"A1", "A2", "B1", "B2", "C1", "C2"},
        "sentences": {"A1", "A2", "B1", "B2", "C1", "C2"},
    }

    if dataset_name not in VALID_LEVELS:
        logger.error(f"Unknown dataset: {dataset_name}")
        return None

    return pred if pred in VALID_LEVELS[dataset_name] else None


def generate_predictions(
    model_id: str,
    dataset: pd.DataFrame,
    dataset_name: str,
    system_prompt: bool = False,
) -> pd.DataFrame:
    """
    Generate predictions for a fine-tuned model on a test dataset.
    Loads existing predictions if available and only predicts missing values.
    Saves results after each prediction for safety.

    Args:
        model_id (str): The ID of the fine-tuned model
        dataset (pd.DataFrame): DataFrame containing test data with 'sentence' column
        dataset_name (str): Name of the dataset being evaluated
        system_prompt (bool): Whether to use system prompt in predictions

    Returns:
        pd.DataFrame: Original dataset with added 'predictions' column
    """
    logger.info(
        f"Starting predictions generation for model {model_id} on {dataset_name}"
    )
    logger.info(f"System prompt: {'enabled' if system_prompt else 'disabled'}")

    # Prepare output path
    system_prompt_suffix = "_with_system" if system_prompt else "_no_system"
    sanitized_model_id = model_id.replace(":", "_")
    output_dir = os.path.join(
        "results", "dmkd_additions", "difficulty_estimation", "gpt_evaluation"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}_{sanitized_model_id}{system_prompt_suffix}_predictions.csv",
    )

    # Initialize result DataFrame
    result_df = dataset.copy()

    # Try to load existing predictions
    if os.path.exists(output_file):
        logger.info(f"Loading existing predictions from {output_file}")
        try:
            existing_predictions = pd.read_csv(output_file)
            if len(existing_predictions) == len(dataset):
                result_df["predictions"] = existing_predictions["predictions"]
                logger.info(f"Loaded {len(existing_predictions)} existing predictions")
            else:
                logger.warning(
                    "Existing predictions file has different length, starting fresh"
                )
                result_df["predictions"] = None
        except pd.errors.EmptyDataError:
            logger.warning("Existing predictions file is empty, starting fresh")
            result_df["predictions"] = None
    else:
        result_df["predictions"] = None
        logger.info("No existing predictions found, starting from scratch")

    # Get indices of missing or empty predictions
    missing_indices = result_df[
        result_df["predictions"].isna() | (result_df["predictions"] == "")
    ].index

    if len(missing_indices) == 0:
        logger.info("All predictions already exist, no new predictions needed")
        return result_df

    logger.info(f"Generating predictions for {len(missing_indices)} missing entries")

    # Initialize OpenAI client
    client = OpenAI()

    # Process predictions one by one with progress bar
    for idx in tqdm(missing_indices, desc="Generating predictions"):
        text = result_df.loc[idx, "sentence"]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": text})

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=10,
                n=1,
                presence_penalty=0,
                frequency_penalty=0,
            )

            pred = response.choices[0].message.content
            validated_pred = validate_prediction(pred, dataset_name)

            if validated_pred is None:
                logger.warning(
                    f"Invalid prediction format: '{pred}' for dataset {dataset_name}"
                )
            else:
                logger.debug(f"Prediction for text '{text[:50]}...': {validated_pred}")
                # Update single prediction only if valid
                result_df.loc[idx, "predictions"] = validated_pred
                # Save after each valid prediction
                result_df.to_csv(output_file, index=False)
                logger.debug(f"Saved progress to {output_file}")

        except Exception as e:
            logger.error(f"Error predicting for text: {text[:50]}... Error: {str(e)}")
            continue

    # Final statistics
    total_predictions = len(result_df)
    valid_predictions = result_df["predictions"].notna().sum()
    success_rate = (valid_predictions / total_predictions) * 100

    logger.info(
        f"Predictions completed: {valid_predictions}/{total_predictions} ({success_rate:.2f}%)"
    )

    return result_df


if __name__ == "__main__":
    import json
    from src.utilitaries.openai_utils import setup_openai_token
    from dmkd_additions.difficulty_estimation.download_data import download_data

    # Setup OpenAI token
    setup_openai_token()
    logger.info("Starting evaluation script")

    try:
        # Load fine-tuned models information
        results_dir = os.path.join(
            "results", "dmkd_additions", "difficulty_estimation", "gpt_fine_tuning"
        )

        # Test datasets
        data = download_data()
        test_cases = {
            "french_difficulty": data.french_difficulty.test.content,
            "ljl": data.ljl.test.content,
            "sentences": data.sentences.test.content,
        }

        # Find all job files
        job_files = [f for f in os.listdir(results_dir) if f.endswith("_job.json")]

        for job_file in job_files:
            with open(os.path.join(results_dir, job_file), "r") as f:
                job_details = json.load(f)

            model_id = job_details.get("fine_tuned_model")
            if not model_id:
                logger.warning(f"No model ID found in {job_file}")
                continue

            # Extract dataset name from job_details instead of filename
            dataset_name = job_details.get("dataset_name")
            if not dataset_name or dataset_name not in test_cases:
                logger.warning(f"Invalid dataset name in {job_file}: {dataset_name}")
                continue

            system_prompt = "_with_system" in job_file

            logger.info(f"\nTesting model: {model_id}")
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"System prompt: {system_prompt}")

            try:
                # Generate predictions for one example
                result_df = generate_predictions(
                    model_id=model_id,
                    dataset=test_cases[dataset_name],
                    dataset_name=dataset_name,
                    system_prompt=system_prompt,
                )

                # Display results
                logger.info("\nTest Results:")
                logger.info(f"Original text: {result_df['sentence'].iloc[0]}")
                logger.info(f"True difficulty: {result_df['difficulty'].iloc[0]}")
                logger.info(f"Predicted difficulty: {result_df['predictions'].iloc[0]}")

            except Exception as e:
                logger.error(f"Error testing model {model_id}: {str(e)}")
                logger.error(
                    f"Full error details: {str(e.__class__.__name__)}: {str(e)}"
                )
                continue

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

    logger.info("Evaluation script completed")
