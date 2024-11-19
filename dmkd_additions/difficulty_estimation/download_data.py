import os
from pathlib import Path
import git
import shutil
from huggingface_hub import snapshot_download
from src.utilitaries.Logger import Logger

logger = Logger(__name__)


def download_data(force_download: bool = False) -> None:
    """Downloads the datasets for difficulty estimation from HuggingFace hub.

    Args:
        force_download (bool): If True, forces download even if files already exist.
    """
    logger.info("Starting data download process...")

    # Get project root directory using git
    try:
        pwd = git.Repo(".", search_parent_directories=True).working_dir
    except git.exc.InvalidGitRepositoryError:
        pwd = os.getcwd()
        logger.warning("Not in a git repository. Using current directory.")

    # Set up paths
    path_temp = os.path.join(pwd, "temp", "Data")
    output_dir = os.path.join(pwd, "results", "raw")

    # Clean temp directory if it exists
    if os.path.exists(path_temp):
        logger.debug("Cleaning temporary directory")
        shutil.rmtree(path_temp)

    # Check if data already exists
    if os.path.exists(output_dir):
        if not force_download:
            logger.warning("Data already downloaded.")
            return
        else:
            logger.info("Removing old data.")
            shutil.rmtree(output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Output directory set to: {output_dir}")

    try:
        # Download data from HuggingFace
        logger.info("Downloading data from HuggingFace hub")
        snapshot_download(
            repo_id="OloriBern/FLDE",
            allow_patterns=["Data/*.csv"],
            local_dir=os.path.join(pwd, "temp"),
            revision="main",
            repo_type="dataset",
        )

        # Move data to correct folder
        logger.info("Moving data to correct folder")
        for file in os.listdir(path_temp):
            if file.endswith(".csv"):
                shutil.copy2(
                    os.path.join(path_temp, file), os.path.join(output_dir, file)
                )

        # Clean up temp directory
        logger.debug("Cleaning up temporary files")
        shutil.rmtree(path_temp)

        logger.info("Data download completed successfully")

    except Exception as e:
        logger.error(f"Error during download process: {str(e)}")
        raise


if __name__ == "__main__":
    download_data()
