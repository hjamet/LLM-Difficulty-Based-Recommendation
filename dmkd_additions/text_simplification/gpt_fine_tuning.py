import os
import json
import signal
import pandas as pd
from openai import OpenAI
from typing import Dict, Any
from tqdm import tqdm
from src.utilitaries.Logger import Logger
import time

# Initialize logger
logger = Logger(__name__)

# Global variable to store OpenAI client for signal handler
global_client = None

# Constants
RESULTS_BASE_DIR = "results"
SCRATCH_BASE_DIR = "scratch"
DMKD_SUBDIR = os.path.join("dmkd_additions", "text_simplification", "gpt_fine_tuning")

DEFAULT_SYSTEM_PROMPT = """Vous êtes un modèle de langage naturel capable de simplifier des phrases en français. 
La phrase simplifiée doit avoir un sens aussi proche que possible de la phrase originale, 
mais elle est d'un niveau inférieur du CECRL et donc plus facile à comprendre. 
Par exemple, si une phrase est au niveau C1 du CECRL, simplifiez-la en B2. 
Si elle se situe au niveau B2, simplifiez-la en B1. 
Si elle se situe au niveau B1, simplifiez-la en A2. 
Si le niveau A2 est atteint, simplifiez en A1."""


def signal_handler(signum, frame):
    """Handler for system signals (Ctrl+C, system shutdown, etc.)."""
    logger.warning("Received interrupt signal. Cleaning up...")
    if global_client:
        _cancel_all_jobs(global_client)
    logger.info("Cleanup complete. Exiting...")
    exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # System shutdown


def fine_tune_gpt(
    model_name: str, system_prompt: bool = False, force: bool = False
) -> Dict[str, Any]:
    """Fine-tunes a GPT model for text simplification.

    Args:
        model_name (str): The name of the GPT model to fine-tune
        system_prompt (bool): Whether to use system prompt
        force (bool): If True, forces fine-tuning even if model already exists

    Returns:
        dict: Job details including the fine-tuned model ID
    """
    global global_client

    # Check if model already exists
    results_dir = os.path.join(RESULTS_BASE_DIR, DMKD_SUBDIR)
    system_prompt_suffix = "_with_system" if system_prompt else "_no_system"
    job_file_path = os.path.join(
        results_dir, f"{model_name}{system_prompt_suffix}_job.json"
    )

    if os.path.exists(job_file_path) and not force:
        logger.info(f"Found existing fine-tuned model for {model_name}")
        with open(job_file_path, "r") as f:
            job_details = json.load(f)
        logger.info(f"Loading existing model: {job_details.get('fine_tuned_model')}")
        return job_details

    # Confirmation for the user
    if (
        not input(
            f"Are you sure you want to fine-tune {model_name} {'with system prompt' if system_prompt else 'without system prompt'} ? (y/n): "
        ).lower()
        == "y"
    ):
        logger.info("Fine-tuning cancelled by user")
        return

    logger.info(f"Starting fine-tuning process for model {model_name}")

    try:
        # Initialize OpenAI client
        client = OpenAI()
        global_client = client

        # Download and prepare data
        from dmkd_additions.text_simplification.download_data import download_data

        data = download_data().content

        # Prepare training data
        train_conversations = __prepare_training_data(data, system_prompt)

        # Save training file
        scratch_dir = os.path.join(SCRATCH_BASE_DIR, DMKD_SUBDIR)
        train_file_path = os.path.join(
            scratch_dir,
            f"{model_name}{system_prompt_suffix}_train.jsonl",
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

        start_time = time.time()
        with tqdm(desc="Fine-tuning progress", unit="steps", ncols=100) as pbar:
            last_step = 0
            while True:
                try:
                    job_status = client.fine_tuning.jobs.retrieve(job.id)
                    events = client.fine_tuning.jobs.list_events(job.id, limit=1).data

                    if events:
                        latest_event = events[0]
                        event_data = (
                            latest_event.data if hasattr(latest_event, "data") else {}
                        )

                        if isinstance(event_data, dict):
                            current_step = event_data.get("step", 0)
                            total_steps = event_data.get("total_steps", 0)
                            train_loss = event_data.get("train_loss", None)

                            if current_step and current_step > last_step:
                                pbar.update(current_step - last_step)
                                last_step = current_step

                            desc = f"Step {current_step}/{total_steps}"
                            if train_loss is not None:
                                desc += f" | Loss: {train_loss:.3f}"

                            elapsed = time.time() - start_time
                            desc += f" | {elapsed:.0f}s"

                            pbar.set_description(desc)

                            if total_steps and pbar.total != total_steps:
                                pbar.total = total_steps

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
                        logger.error("Fine-tuning was cancelled")
                        raise Exception("Fine-tuning was cancelled")

                except Exception as e:
                    logger.error(f"Error while monitoring fine-tuning: {str(e)}")

                time.sleep(60)  # Check every minute

        # Save job details
        job_details = {
            "job_id": job.id,
            "model_name": model_name,
            "training_file_id": training_file.id,
            "status": "succeeded",
            "created_at": job.created_at,
            "finished_at": job_status.finished_at,
            "fine_tuned_model": job_status.fine_tuned_model,
        }

        os.makedirs(results_dir, exist_ok=True)
        with open(job_file_path, "w") as f:
            json.dump(job_details, f, indent=4)

        logger.info(
            f"Fine-tuning job completed successfully. Details saved to {job_file_path}"
        )
        return job_details

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}")
        if client:
            _cancel_all_jobs(client)
        raise
    finally:
        if client:
            _cancel_all_jobs(client)
        global_client = None


def __prepare_training_data(data: pd.DataFrame, system_prompt: bool = False) -> list:
    """Prepares training data in the format required by OpenAI's fine-tuning API."""
    logger.debug("Preparing training data for fine-tuning")

    conversations = []
    for _, row in data.iterrows():
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

        messages.extend(
            [
                {"role": "user", "content": row["Original"]},
                {"role": "assistant", "content": row["Simplified"]},
            ]
        )

        conversations.append({"messages": messages})

    logger.debug(f"Prepared {len(conversations)} conversations for training")
    return conversations


def __save_training_file(conversations: list, filename: str) -> str:
    """Saves training data in JSONL format."""
    logger.debug(f"Saving training data to {filename}")

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    return filename


def _cancel_all_jobs(client: OpenAI) -> None:
    """Cancels all running fine-tuning jobs."""
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
    from src.utilitaries.openai_utils import setup_openai_token

    # Setup OpenAI token
    setup_openai_token()

    # Models to fine-tune (using correct model IDs)
    models = [
        "gpt-4o-mini-2024-07-18",  # Comme dans difficulty_estimation
        "gpt-4o-2024-08-06",  # Comme dans difficulty_estimation
    ]

    for model in models:
        logger.info(f"Starting fine-tuning process for model: {model}")
        try:
            # Create backup directory
            backup_dir = os.path.join(RESULTS_BASE_DIR, DMKD_SUBDIR, "backups")
            os.makedirs(backup_dir, exist_ok=True)

            # Save configuration
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            backup_config = {
                "model": model,
                "system_prompt": True,  # Always True for text simplification
                "timestamp": timestamp,
            }

            backup_file = os.path.join(backup_dir, f"config_{timestamp}.json")
            with open(backup_file, "w") as f:
                json.dump(backup_config, f, indent=4)

            # Perform fine-tuning
            job_details = fine_tune_gpt(
                model_name=model,
                system_prompt=True,  # Always True for text simplification
                force=False,
            )

            if job_details and job_details.get("fine_tuned_model"):
                logger.info(f"Fine-tuning completed successfully for {model}")
                logger.info(f"Fine-tuned model ID: {job_details['fine_tuned_model']}")
            else:
                logger.warning(f"No fine-tuned model produced for {model}")

        except Exception as e:
            logger.error(f"Error fine-tuning {model}: {str(e)}")
            continue
