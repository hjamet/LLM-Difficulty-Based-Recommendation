from gpt_fine_tuning import fine_tune_gpt
from src.utilitaries.Logger import Logger
from .utils import setup_openai_token

logger = Logger(__name__)


def main():
    # Setup OpenAI token
    setup_openai_token()

    # Models to fine-tune
    models = ["gpt-4o-mini", "gpt-4o"]

    # Datasets to use
    datasets = ["french_difficulty", "ljl", "sentences"]

    for model in models:
        logger.info(f"Starting fine-tuning process for model: {model}")

        for dataset in datasets:
            logger.info(f"Fine-tuning on dataset: {dataset}")

            try:
                job_details = fine_tune_gpt(
                    model_name=model, dataset_name=dataset, system_prompt=True
                )

                logger.info(
                    f"Fine-tuning job created successfully for {model} on {dataset}"
                )
                logger.info(f"Job ID: {job_details['job_id']}")

            except Exception as e:
                logger.error(f"Error fine-tuning {model} on {dataset}: {str(e)}")
                continue


if __name__ == "__main__":
    main()
