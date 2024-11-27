import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from src.utilitaries.Logger import Logger
from dmkd_additions.text_simplification.gpt_fine_tuning import DEFAULT_SYSTEM_PROMPT
import logging

# Disable OpenAI logging
logging.getLogger("openai").setLevel(logging.WARNING)

# Initialize logger
logger = Logger(__name__)


def generate_predictions(
    model_id: str,
    dataset: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Generate text simplifications using a GPT model.

    Args:
        model_id (str): The ID of the model to use
        dataset (pd.DataFrame): DataFrame containing sentences to simplify
        dataset_name (str): Name of the dataset (for logging and saving)

    Returns:
        pd.DataFrame: Original dataset with added 'simplified' column
    """
    logger.info(
        f"Starting predictions generation for model {model_id} on {dataset_name}"
    )

    # Prepare output path
    output_dir = os.path.join(
        "results", "dmkd_additions", "text_simplification", "gpt_evaluation"
    )
    os.makedirs(output_dir, exist_ok=True)

    sanitized_model_id = model_id.replace(":", "_")
    output_file = os.path.join(
        output_dir,
        f"{dataset_name}_{sanitized_model_id}_predictions.csv",
    )

    # Initialize result DataFrame
    result_df = dataset.copy()
    result_df["simplified"] = None

    # Try to load existing predictions
    if os.path.exists(output_file):
        logger.info(f"Loading existing predictions from {output_file}")
        try:
            existing_predictions = pd.read_csv(output_file)
            if len(existing_predictions) == len(dataset):
                result_df["simplified"] = existing_predictions["simplified"]
                logger.info(f"Loaded {len(existing_predictions)} existing predictions")
            else:
                logger.warning(
                    "Existing predictions file has different length, starting fresh"
                )
        except Exception as e:
            logger.warning(f"Error loading existing predictions: {str(e)}")

    # Get indices of missing predictions
    missing_indices = result_df[result_df["simplified"].isna()].index

    if len(missing_indices) == 0:
        logger.info("All predictions already exist")
        return result_df

    logger.info(f"Generating predictions for {len(missing_indices)} missing entries")

    # Initialize OpenAI client
    client = OpenAI()

    # Process predictions with progress bar
    for idx in tqdm(missing_indices, desc="Generating predictions"):
        text = result_df.loc[idx, "Original"]
        difficulty = result_df.loc[idx, "Difficulty"]

        # Determine target difficulty level
        level_map = {"C2": "C1", "C1": "B2", "B2": "B1", "B1": "A2", "A2": "A1"}
        target_level = level_map.get(difficulty)

        if not target_level:
            logger.warning(f"Invalid difficulty level: {difficulty}")
            continue

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0,
                max_tokens=len(text) * 2,  # Allow enough tokens for response
                n=1,
            )

            simplified = response.choices[0].message.content.strip()

            # Update and save after each prediction
            result_df.loc[idx, "simplified"] = simplified
            result_df.to_csv(output_file, index=False)

            logger.debug(f"Original ({difficulty}->{target_level}): {text[:50]}...")
            logger.debug(f"Simplified: {simplified[:50]}...")

        except Exception as e:
            logger.error(f"Error predicting for text: {text[:50]}... Error: {str(e)}")
            continue

    # Log completion statistics
    total = len(result_df)
    completed = result_df["simplified"].notna().sum()
    logger.info(
        f"Predictions completed: {completed}/{total} ({(completed/total)*100:.2f}%)"
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
        # Load test data
        data = download_data()
        test_df = pd.concat(
            [data.french_difficulty.test.content, data.sentences.test.content]
        )
        # Remove A1 level and keep 100 sentences per level
        test_df = test_df[test_df["difficulty"] != "A1"].groupby("difficulty").head(100)
        test_df.columns = ["Original", "Difficulty"]  # Standardize column names

        # Load fine-tuned models information
        results_dir = os.path.join(
            "results", "dmkd_additions", "text_simplification", "gpt_fine_tuning"
        )

        # Find all job files
        job_files = [f for f in os.listdir(results_dir) if f.endswith("_job.json")]

        for job_file in job_files:
            with open(os.path.join(results_dir, job_file), "r") as f:
                job_details = json.load(f)

            model_id = job_details.get("fine_tuned_model")
            if not model_id:
                logger.warning(f"No model ID found in {job_file}")
                continue

            logger.info(f"\nTesting model: {model_id}")

            # Generate predictions
            result_df = generate_predictions(
                model_id=model_id, dataset=test_df, dataset_name="test_set"
            )

            # Display sample results
            logger.info("\nSample Results:")
            sample = result_df.sample(n=3)
            for _, row in sample.iterrows():
                logger.info(f"\nOriginal ({row['Difficulty']}): {row['Original']}")
                logger.info(f"Simplified: {row['simplified']}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

    logger.info("Evaluation script completed")
