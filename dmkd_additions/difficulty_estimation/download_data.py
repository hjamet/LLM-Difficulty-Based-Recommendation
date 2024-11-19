import os
import git
import shutil
import pandas as pd
from huggingface_hub import snapshot_download
from src.utilitaries.Logger import Logger
from src.utilitaries.NamedTuples import DownloadedData, DatasetResult, DownloadResult

logger = Logger(__name__)


def download_data(force_download: bool = False) -> DownloadResult:
    """Downloads the datasets for difficulty estimation from HuggingFace hub.

    Args:
        force_download (bool): If True, forces download even if files already exist.

    Returns:
        DownloadResult: Named tuple containing DatasetResult for each dataset type
        (french_difficulty, ljl, sentences), where each DatasetResult contains train and test data.
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
    output_dir = os.path.join(pwd, "data", "raw")

    # Clean temp directory if it exists
    if os.path.exists(path_temp):
        logger.debug("Cleaning temporary directory")
        shutil.rmtree(path_temp)

    # Define expected datasets and their files
    datasets = {
        "french_difficulty": {
            "train": "french_difficulty_train.csv",
            "test": "french_difficulty_test.csv",
        },
        "ljl": {"train": "ljl_train.csv", "test": "ljl_test.csv"},
        "sentences": {"train": "sentences_train.csv", "test": "sentences_test.csv"},
    }

    # Check if data already exists
    downloaded_data = {}
    if os.path.exists(output_dir) and not force_download:
        logger.warning("Data already downloaded. Loading existing files.")
        for dataset_name, files in datasets.items():
            dataset_files = {}
            for data_type, filename in files.items():
                file_path = os.path.join(output_dir, filename)
                if os.path.exists(file_path):
                    dataset_files[data_type] = DownloadedData(
                        file_path=file_path, content=pd.read_csv(file_path)
                    )
            if len(dataset_files) == len(files):
                downloaded_data[dataset_name] = DatasetResult(**dataset_files)

        if len(downloaded_data) == len(datasets):
            return DownloadResult(**downloaded_data)

    if os.path.exists(output_dir) and force_download:
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

        # Move data to correct folder and store in namedtuples
        logger.info("Moving data to correct folder and loading content")
        for dataset_name, files in datasets.items():
            dataset_files = {}
            for data_type, filename in files.items():
                source_path = os.path.join(path_temp, filename)
                if os.path.exists(source_path):
                    dest_path = os.path.join(output_dir, filename)
                    shutil.copy2(source_path, dest_path)
                    dataset_files[data_type] = DownloadedData(
                        file_path=dest_path, content=pd.read_csv(dest_path)
                    )
                else:
                    logger.warning(f"File {filename} not found in downloaded data")

            if len(dataset_files) == len(files):
                downloaded_data[dataset_name] = DatasetResult(**dataset_files)

        # Clean up temp directory
        logger.debug("Cleaning up temporary files")
        shutil.rmtree(path_temp)

        logger.info("Data download completed successfully")
        return DownloadResult(**downloaded_data)

    except Exception as e:
        logger.error(f"Error during download process: {str(e)}")
        raise


if __name__ == "__main__":
    result = download_data()
    # Example of accessing the data:
    for dataset_name in ["french_difficulty", "ljl", "sentences"]:
        dataset = getattr(result, dataset_name)
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"Train data shape: {dataset.train.content.shape}")
        print(f"Test data shape: {dataset.test.content.shape}")
