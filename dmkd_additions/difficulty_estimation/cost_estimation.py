import logging
import tiktoken
from typing import Dict, List
from src.utilitaries.NamedTuples import DatasetResult, PricingEstimation
from src.utilitaries.Logger import Logger
from dmkd_additions.difficulty_estimation.gpt_fine_tuning import DEFAULT_SYSTEM_PROMPT

# Initialize logger
logger = Logger(__name__)


def estimate_cost(
    dataset: DatasetResult,
    system_prompt: bool,
    model_pricing: Dict[str, float],
) -> PricingEstimation:
    """
    Estimates the cost of fine-tuning and using a model based on dataset and pricing.

    Args:
        dataset (DatasetResult): Named tuple containing train and test datasets
        system_prompt (bool): Whether system prompt is used
        model_pricing (Dict[str, float]): Pricing information for the model (cost per 1M tokens)

    Returns:
        PricingEstimation: Named tuple containing cost breakdown and total
    """
    logger.debug(
        f"Estimating cost for dataset with {len(dataset.train.content)} training samples"
    )

    def __count_tokens_from_messages(messages: List[Dict[str, str]], model: str) -> int:
        """
        Return the number of tokens used by a list of messages.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found. Using o200k_base encoding.")
            encoding = tiktoken.get_encoding("o200k_base")

        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0

        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def __prepare_training_messages(
        text: str, level: str, system_prompt: str
    ) -> List[Dict[str, str]]:
        """
        Prepares messages for training in the format expected by OpenAI.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            [
                {"role": "user", "content": str(text)},
                {"role": "assistant", "content": str(level)},
            ]
        )
        return messages

    # Calculate token counts
    total_input_tokens = 0
    total_output_tokens = 0

    # Process each training example
    for _, row in dataset.train.content.iterrows():
        messages = __prepare_training_messages(
            text=row["sentence"],
            level=row["difficulty" if "difficulty" in row else "level1"],
            system_prompt=DEFAULT_SYSTEM_PROMPT if system_prompt else "",
        )

        message_tokens = __count_tokens_from_messages(messages, "gpt-4")

        # Count output tokens (last message)
        encoding = tiktoken.get_encoding("o200k_base")
        output_tokens = len(encoding.encode(messages[-1]["content"]))
        input_tokens = message_tokens - output_tokens

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    total_training_tokens = total_input_tokens + total_output_tokens

    # Calculate costs per million tokens using provided pricing
    input_cost = (total_input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * model_pricing["output"]
    training_cost = (total_training_tokens / 1_000_000) * model_pricing["training"]

    total_cost = input_cost + output_cost + training_cost

    logger.debug(
        f"Token counts - Input: {total_input_tokens}, Output: {total_output_tokens}, Training: {total_training_tokens}"
    )
    logger.debug(
        f"Cost breakdown - Input: ${input_cost:.2f}, Output: ${output_cost:.2f}, Training: ${training_cost:.2f}"
    )

    return PricingEstimation(
        input_tokens=input_cost,
        output_tokens=output_cost,
        training_tokens=training_cost,
        total=total_cost,
    )


if __name__ == "__main__":
    import os
    from datetime import datetime
    from dmkd_additions.difficulty_estimation.download_data import download_data

    # Constants for pricing (per 1M tokens)
    PRICING = {
        "gpt-4o-2024-08-06": {"input": 3.750, "output": 15.000, "training": 25.000},
        "gpt-4o-mini-2024-07-18": {"input": 0.300, "output": 1.200, "training": 3.000},
    }

    def __check_if_done(
        model: str, dataset: str, system_prompt: bool, task_type: str
    ) -> bool:
        """
        Checks if a specific task has already been completed.

        Args:
            model (str): Model name
            dataset (str): Dataset name
            system_prompt (bool): Whether system prompt was used
            task_type (str): Either 'fine_tuning' or 'evaluation'

        Returns:
            bool: True if task has been completed
        """
        suffix = "with_system" if system_prompt else "no_system"
        base_path = os.path.join("results", "dmkd_additions", "difficulty_estimation")

        if task_type == "fine_tuning":
            file_path = os.path.join(
                base_path, "gpt_fine_tuning", f"{model}_{dataset}_{suffix}_job.json"
            )
        else:  # evaluation
            file_path = os.path.join(
                base_path,
                "gpt_evaluation",
                f"{model}_{dataset}_{suffix}_evaluation.json",
            )

        return os.path.exists(file_path)

    # Prepare results directory
    results_dir = os.path.join(
        "results", "dmkd_additions", "difficulty_estimation", "cost_estimation"
    )
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_file = os.path.join(results_dir, f"estimation_{timestamp}.md")

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Cost Estimation for Fine-tuning Models\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        data = download_data()
        datasets = ["french_difficulty", "ljl", "sentences"]
        models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]

        total_cost = 0
        total_paid_cost = 0
        total_remaining_cost = 0

        for model_name in models:
            f.write(f"\n## Model: {model_name}\n\n")
            print(f"\nModel: {model_name}")
            print("-" * 40)

            for dataset_name in datasets:
                dataset = getattr(data, dataset_name)

                for use_system_prompt in [True, False]:
                    logger.info(
                        f"Processing {dataset_name} with model {model_name} {'with' if use_system_prompt else 'without'} system prompt"
                    )

                    cost_estimate = estimate_cost(
                        dataset=dataset,
                        system_prompt=use_system_prompt,
                        model_pricing=PRICING[model_name],
                    )

                    # Check what's already been done
                    fine_tuning_done = __check_if_done(
                        model_name, dataset_name, use_system_prompt, "fine_tuning"
                    )
                    evaluation_done = __check_if_done(
                        model_name, dataset_name, use_system_prompt, "evaluation"
                    )

                    # Calculate paid and remaining costs
                    paid_training = (
                        cost_estimate.training_tokens if fine_tuning_done else 0
                    )
                    paid_evaluation = (
                        (cost_estimate.input_tokens + cost_estimate.output_tokens)
                        if evaluation_done
                        else 0
                    )
                    paid_total = paid_training + paid_evaluation

                    remaining_training = (
                        0 if fine_tuning_done else cost_estimate.training_tokens
                    )
                    remaining_evaluation = (
                        0
                        if evaluation_done
                        else (cost_estimate.input_tokens + cost_estimate.output_tokens)
                    )
                    remaining_total = remaining_training + remaining_evaluation

                    total_cost += cost_estimate.total
                    total_paid_cost += paid_total
                    total_remaining_cost += remaining_total

                    prompt_type = (
                        "with system prompt"
                        if use_system_prompt
                        else "without system prompt"
                    )

                    # Write to markdown file
                    f.write(f"### Dataset: {dataset_name} ({prompt_type})\n\n")
                    f.write(
                        f"- Number of training samples: {len(dataset.train.content)}\n"
                    )
                    f.write(f"- Status:\n")
                    f.write(
                        f"  - Fine-tuning: {'✓ Done' if fine_tuning_done else '⨯ Pending'}\n"
                    )
                    f.write(
                        f"  - Evaluation: {'✓ Done' if evaluation_done else '⨯ Pending'}\n"
                    )
                    f.write(f"- Costs:\n")
                    f.write(f"  - Training: ${cost_estimate.training_tokens:.2f} ")
                    f.write(f"({'✓ Paid' if fine_tuning_done else '⨯ Pending'})\n")
                    f.write(
                        f"  - Evaluation: ${(cost_estimate.input_tokens + cost_estimate.output_tokens):.2f} "
                    )
                    f.write(f"({'✓ Paid' if evaluation_done else '⨯ Pending'})\n")
                    f.write(f"  - **Total cost: ${cost_estimate.total:.2f}**\n")
                    f.write(f"  - **Already paid: ${paid_total:.2f}**\n")
                    f.write(f"  - **Remaining to pay: ${remaining_total:.2f}**\n\n")

                    # Print to console (similar format)
                    print(f"\nDataset: {dataset_name} ({prompt_type})")
                    print(
                        f"Status: FT: {'✓' if fine_tuning_done else '⨯'}, Eval: {'✓' if evaluation_done else '⨯'}"
                    )
                    print(f"Total cost: ${cost_estimate.total:.2f}")
                    print(f"Already paid: ${paid_total:.2f}")
                    print(f"Remaining to pay: ${remaining_total:.2f}")

        # Write summary
        f.write("\n## Summary\n\n")
        f.write("```\n")
        f.write(f"Total estimated cost: ${total_cost:.2f}\n")
        f.write(f"Already paid: ${total_paid_cost:.2f}\n")
        f.write(f"Remaining to pay: ${total_remaining_cost:.2f}\n")
        f.write("```\n")

        # Print summary to console
        print("\n" + "=" * 40)
        print("RÉSUMÉ DES COÛTS TOTAUX")
        print("=" * 40)
        print(f"Coût total estimé: ${total_cost:.2f}")
        print(f"Déjà payé: ${total_paid_cost:.2f}")
        print(f"Reste à payer: ${total_remaining_cost:.2f}")
        print("=" * 40)

        logger.info(f"Results saved to {md_file}")
