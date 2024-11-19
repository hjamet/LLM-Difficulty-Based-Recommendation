import os
import json
import signal
import pandas as pd
from openai import OpenAI
from typing import Dict, Any
from tqdm import tqdm
from src.utilitaries.Logger import Logger
from src.utilitaries.openai_utils import setup_openai_token
from dmkd_additions.difficulty_estimation.download_data import download_data
import time

# Initialize logger
logger = Logger(__name__)

# Global variable to store OpenAI client for signal handler
global_client = None


def signal_handler(signum, frame):
    """
    Handler for system signals (Ctrl+C, system shutdown, etc.).
    Ensures all fine-tuning jobs are cancelled before exit.
    """
    logger.warning("Received interrupt signal. Cleaning up...")
    if global_client:
        __cancel_all_jobs(global_client)
    logger.info("Cleanup complete. Exiting...")
    exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # System shutdown


def fine_tune_gpt(
    model_name: str, dataset_name: str, system_prompt: bool = False
) -> Dict[str, Any]:
    """
    Fine-tunes a GPT model on a dataset.

    Args:
        model_name (str): The name of the GPT model to fine-tune.
        dataset_name (str): The name of the dataset to fine-tune on.
        system_prompt (bool): Whether to use a system prompt for the fine-tuning.

    Returns:
        dict: A json file describing the fine-tuned model.
    """
    global global_client

    # Confirmation for the user
    if (
        not input(
            f"Are you sure you want to fine-tune {model_name} on {dataset_name}? (y/n): "
        ).lower()
        == "y"
    ):
        logger.info("Fine-tuning cancelled by user")
        return

    logger.info(
        f"Starting fine-tuning process for model {model_name} on dataset {dataset_name}"
    )

    try:
        # Initialize OpenAI client
        client = OpenAI()
        global_client = client  # Store client globally for signal handler

        data = download_data()
        dataset = getattr(data, dataset_name)

        # Prepare training data
        train_conversations = __prepare_training_data(
            dataset.train.content, system_prompt
        )

        # Save training file
        results_dir = "results/dmkd_additions/difficulty_estimation/gpt_fine_tuning"
        train_file_path = os.path.join(
            results_dir, f"{model_name}_{dataset_name}_train.jsonl"
        )
        train_file_path = __save_training_file(train_conversations, train_file_path)

        # Upload file to OpenAI
        logger.info("Uploading training file to OpenAI")
        with open(train_file_path, "rb") as file:
            training_file = client.files.create(file=file, purpose="fine-tune")

        # Create fine-tuning job
        logger.info("Creating fine-tuning job")
        job = client.fine_tuning.jobs.create(
            training_file=training_file.id, model=model_name
        )

        # Wait for job completion with progress bar
        logger.info("Waiting for fine-tuning job to complete...")
        logger.info(f"Job ID: {job.id}")

        # Initialize progress bar
        pbar = tqdm(desc="Fine-tuning progress", unit="tokens")
        last_token_count = 0
        start_time = time.time()

        while True:
            try:
                # Get the latest status and events
                job_status = client.fine_tuning.jobs.retrieve(job.id)
                events = client.fine_tuning.jobs.list_events(job.id, limit=1).data

                # Log latest event if available
                if events:
                    latest_event = events[0]
                    logger.debug(f"Latest event: {latest_event.message}")
                    if hasattr(latest_event, "data"):
                        logger.debug(f"Event data: {latest_event.data}")

                # Check job status
                if job_status.status == "succeeded":
                    pbar.close()
                    logger.info("Fine-tuning completed successfully!")
                    break
                elif job_status.status == "failed":
                    pbar.close()
                    error_message = f"Fine-tuning failed: {getattr(job_status, 'error', 'Unknown error')}"
                    logger.error(error_message)
                    raise Exception(error_message)
                elif job_status.status == "cancelled":
                    pbar.close()
                    error_message = "Fine-tuning was cancelled"
                    logger.error(error_message)
                    raise Exception(error_message)

                # Update progress bar using events data if available
                if events and hasattr(events[0], "data"):
                    event_data = events[0].data
                    if hasattr(event_data, "step") and hasattr(
                        event_data, "train_loss"
                    ):
                        current_step = event_data.step
                        train_loss = event_data.train_loss
                        pbar.set_description(
                            f"Step {current_step} - Loss: {train_loss:.4f}"
                        )
                        pbar.update(1)

                # Fallback to basic progress indication if no detailed metrics
                elif (
                    hasattr(job_status, "trained_tokens")
                    and job_status.trained_tokens is not None
                ):
                    current_tokens = job_status.trained_tokens
                    if current_tokens > last_token_count:
                        pbar.update(current_tokens - last_token_count)
                        last_token_count = current_tokens

                    # Add percentage to progress bar description if total tokens available
                    if (
                        hasattr(job_status, "training_file")
                        and hasattr(job_status.training_file, "tokens")
                        and job_status.training_file.tokens
                    ):
                        total_tokens = job_status.training_file.tokens
                        percentage = (current_tokens / total_tokens) * 100
                        pbar.set_description(f"Fine-tuning progress: {percentage:.1f}%")
                else:
                    # If no metrics available, just show elapsed time
                    elapsed = time.time() - start_time
                    pbar.set_description(
                        f"Fine-tuning in progress (elapsed: {elapsed:.0f}s)"
                    )

            except Exception as e:
                logger.error(f"Error while monitoring fine-tuning: {str(e)}")
                # Continue monitoring even if there's an error getting status

            # Wait before checking again
            time.sleep(60)  # Check every minute

        # Update job details with final status
        job_details = {
            "job_id": job.id,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "training_file_id": training_file.id,
            "status": "succeeded",
            "created_at": job.created_at,
            "finished_at": job_status.finished_at,
            "fine_tuned_model": job_status.fine_tuned_model,
        }

        job_file_path = os.path.join(
            results_dir, f"{model_name}_{dataset_name}_job.json"
        )
        with open(job_file_path, "w") as f:
            json.dump(job_details, f, indent=4)

        logger.info(
            f"Fine-tuning job completed successfully. Job details saved to {job_file_path}"
        )
        return job_details

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        if client:
            __cancel_all_jobs(client)
        raise
    finally:
        # Always try to cancel any remaining jobs in case of error
        if client:
            __cancel_all_jobs(client)
        global_client = None  # Clear global client reference


# ---------------------------------------------------------------------------- #
#                               PRIVATE FUNCTIONS                              #
# ---------------------------------------------------------------------------- #


def __prepare_training_data(dataset: pd.DataFrame, system_prompt: bool = False) -> list:
    """
    Prepares training data in the format required by OpenAI's fine-tuning API.

    Args:
        dataset (pd.DataFrame): DataFrame containing sentence and difficulty/level1 columns
        system_prompt (bool): Whether to include system prompt

    Returns:
        list: List of conversation dictionaries in OpenAI format
    """
    logger.debug("Preparing training data for fine-tuning")
    logger.debug(f"Dataset columns: {dataset.columns.tolist()}")

    # Determine which columns to use
    text_column = "sentence"  # All datasets use "sentence" for text
    # Check if "difficulty" exists, otherwise use "level1"
    difficulty_column = "difficulty"

    logger.debug(f"Using columns: text={text_column}, difficulty={difficulty_column}")

    conversations = []
    default_system_prompt = """Vous êtes un évaluateur linguistique utilisant le Cadre européen commun de référence pour les langues (CECRL). 
    Votre mission est d'attribuer une note de compétence linguistique à ce texte, en utilisant les niveaux du CECRL, 
    allant de A1 (débutant) à C2 (avancé/natif). Évaluez ce texte et attribuez-lui la note correspondante du CECRL."""

    for _, row in dataset.iterrows():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": default_system_prompt})

        messages.extend(
            [
                {"role": "user", "content": str(row[text_column])},
                {"role": "assistant", "content": str(row[difficulty_column])},
            ]
        )

        conversations.append({"messages": messages})

    logger.debug(f"Prepared {len(conversations)} conversations for training")
    return conversations


def __save_training_file(conversations: list, filename: str) -> str:
    """
    Saves training data in JSONL format and returns the file path.

    Args:
        conversations (list): List of conversation dictionaries
        filename (str): Name of the file to save

    Returns:
        str: Path to the saved file
    """
    logger.debug(f"Saving training data to {filename}")

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save conversations in JSONL format
    with open(filename, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    return filename


def __validate_prediction(prediction: str) -> bool:
    """
    Validates that the prediction follows the expected CEFR format.

    Args:
        prediction (str): The prediction to validate

    Returns:
        bool: True if the prediction is valid, False otherwise
    """
    valid_levels = {"A1", "A2", "B1", "B2", "C1", "C2"}
    return prediction.strip().upper() in valid_levels


def __predict_difficulty(model_id: str, text: str, fallback_level: str = "A1") -> str:
    """
    Predicts the difficulty of a text using a fine-tuned GPT model.

    Args:
        model_id (str): The id of the fine-tuned model.
        text (str): The text to predict the difficulty of.
        fallback_level (str): The level to return if prediction fails validation.

    Returns:
        str: The predicted difficulty of the text.
    """
    logger.info(f"Predicting difficulty for text using model {model_id}")

    try:
        client = OpenAI()

        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": text}],
            temperature=0,
            max_tokens=10,
        )

        predicted_difficulty = response.choices[0].message.content.strip().upper()
        logger.debug(f"Raw prediction: {predicted_difficulty}")

        if __validate_prediction(predicted_difficulty):
            return predicted_difficulty
        else:
            logger.warning(
                f"Invalid prediction format: {predicted_difficulty}. Using fallback: {fallback_level}"
            )
            return fallback_level

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return fallback_level


def __test_predictions(model_id: str, dataset_name: str, system_prompt: bool):
    """
    Tests the model with example predictions and saves results.

    Args:
        model_id (str): The fine-tuned model ID
        dataset_name (str): Name of the dataset used for fine-tuning
        system_prompt (bool): Whether system prompt was used
    """
    # Example texts for testing
    test_texts = {
        "A1": "Je m'appelle Jean. J'ai 20 ans.",
        "C2": """L'épistémologie contemporaine, héritière des travaux séminaux de Bachelard 
                et Popper, s'interroge sur la démarcation entre science et pseudo-science, 
                tout en questionnant les fondements méthodologiques de la recherche empirique.""",
    }

    results_dir = "results/dmkd_additions/difficulty_estimation/predictions"
    os.makedirs(results_dir, exist_ok=True)

    results = []
    for difficulty, text in test_texts.items():
        prediction = __predict_difficulty(model_id, text)
        results.append(
            {
                "expected_level": difficulty,
                "text": text,
                "predicted_level": prediction,
                "model": model_id,
                "dataset": dataset_name,
                "system_prompt": system_prompt,
            }
        )

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{results_dir}/predictions_{model_id}_{dataset_name}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"Test predictions saved to {filename}")


def __cancel_all_jobs(client: OpenAI) -> None:
    """
    Cancels all running fine-tuning jobs.

    Args:
        client (OpenAI): The OpenAI client instance
    """
    logger.info("Cancelling all running fine-tuning jobs")
    try:
        jobs = client.fine_tuning.jobs.list(limit=50)
        with tqdm(desc="Cancelling jobs", unit="job") as pbar:
            for job in jobs:
                if job.status in ["running", "created", "pending"]:
                    logger.debug(f"Cancelling job {job.id}")
                    try:
                        client.fine_tuning.jobs.cancel(job.id)
                        logger.info(f"Successfully cancelled job {job.id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel job {job.id}: {str(e)}")
                    pbar.update(1)
    except Exception as e:
        logger.error(f"Error while cancelling jobs: {str(e)}")


if __name__ == "__main__":
    # Setup OpenAI token
    setup_openai_token()

    # Models to fine-tune
    models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]

    # Datasets to use
    datasets = ["french_difficulty", "ljl", "sentences"]

    for model in models:
        logger.info(f"Starting fine-tuning process for model: {model}")

        for dataset in datasets:
            logger.info(f"Fine-tuning on dataset: {dataset}")

            for system_prompt in [True, False]:
                try:
                    # Create backup directory
                    backup_dir = os.path.join(
                        "results",
                        "dmkd_additions",
                        "difficulty_estimation",
                        "backups",
                    )
                    os.makedirs(backup_dir, exist_ok=True)

                    # Save configuration before fine-tuning
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    backup_config = {
                        "model": model,
                        "dataset": dataset,
                        "system_prompt": system_prompt,
                        "timestamp": timestamp,
                    }

                    backup_file = f"{backup_dir}/config_{timestamp}.json"
                    with open(backup_file, "w") as f:
                        json.dump(backup_config, f, indent=4)

                    # Perform fine-tuning
                    job_details = fine_tune_gpt(
                        model_name=model,
                        dataset_name=dataset,
                        system_prompt=system_prompt,
                    )

                    if job_details and job_details.get("fine_tuned_model"):
                        logger.info(
                            f"Fine-tuning job completed successfully for {model} on {dataset}"
                        )
                        logger.info(
                            f"Fine-tuned model ID: {job_details['fine_tuned_model']}"
                        )

                        # Test predictions with the fine-tuned model
                        __test_predictions(
                            job_details["fine_tuned_model"], dataset, system_prompt
                        )
                    else:
                        logger.warning(
                            f"No fine-tuned model produced for {model} on {dataset}"
                        )

                except Exception as e:
                    logger.error(f"Error fine-tuning {model} on {dataset}: {str(e)}")
                    continue
