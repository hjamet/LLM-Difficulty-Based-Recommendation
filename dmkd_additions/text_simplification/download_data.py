import os
import git
import pandas as pd
from huggingface_hub import snapshot_download
from src.utilitaries.Logger import Logger
from src.utilitaries.NamedTuples import DownloadedData

logger = Logger(__name__)


def download_data(force_download: bool = False) -> DownloadedData:
    """Downloads the dataset for text simplification from HuggingFace hub.

    Args:
        force_download (bool): If True, forces download even if files already exist.

    Returns:
        DownloadedData: Named tuple containing the file path and content of the dataset.
    """
    logger.info("Starting data download process...")

    # Get project root directory using git
    try:
        pwd = git.Repo(".", search_parent_directories=True).working_dir
    except git.exc.InvalidGitRepositoryError:
        pwd = os.getcwd()
        logger.warning("Not in a git repository. Using current directory.")

    # Set up paths
    output_dir = os.path.join(pwd, "data", "raw", "sentence_simplification")
    output_file = os.path.join(output_dir, "fine_tuning_dataset.csv")

    # Check if data already exists
    if os.path.exists(output_file) and not force_download:
        logger.warning("Data already downloaded. Loading existing file.")
        return DownloadedData(
            file_path=output_file,
            content=pd.read_csv(
                output_file, sep=" -> ", names=["Original", "Simplified"]
            ),
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Output directory set to: {output_dir}")

    try:
        # Download data directly to final location
        logger.info("Downloading data from HuggingFace hub")
        snapshot_download(
            repo_id="OloriBern/FLDE",
            allow_patterns=["sentence_simplification/*.csv"],
            local_dir=os.path.join(pwd, "data", "raw"),
            local_dir_use_symlinks=False,  # Get actual files instead of symlinks
            revision="main",
            repo_type="dataset",
        )

        # Load the downloaded data
        logger.info("Loading downloaded content")
        if os.path.exists(output_file):
            data = pd.read_csv(
                output_file, sep=" -> ", names=["Original", "Simplified"]
            )
        else:
            raise FileNotFoundError(
                "fine_tuning_dataset.csv not found in downloaded data"
            )

        logger.info("Data download completed successfully")
        return DownloadedData(file_path=output_file, content=data)

    except Exception as e:
        logger.error(f"Error during download process: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    result = download_data()
    print("\nText Simplification Dataset:")
    print(f"Data shape: {result.content.shape}")
    print("\nFirst few rows:")
    print(result.content.head())
