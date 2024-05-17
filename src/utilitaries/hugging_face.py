# -------------------------- CONNECT TO HUGGINGFACE -------------------------- #
from huggingface_hub import login
import os
import git


def ft_get_huggingface_token(pwd: str = None):
    """
    Get the HuggingFace token from the user and save it in a file.

    Args:
        pwd (str): The current working directory. Defaults to None.
    """
    # Find PWD
    if pwd is None:
        repo = git.Repo(".", search_parent_directories=True)
        pwd = repo.working_dir
        
    try:
        with open(os.path.join(pwd, ".huggingface_key"), "r") as f:
            huggingface_token = f.read()
            # login(token=huggingface_token)
    except:
        print("Invalid token in .huggingface_key.")

    return huggingface_token


# ------------------------------- DOWNLOAD_DATA ------------------------------ #
from huggingface_hub import snapshot_download


def ft_download_data(data_name: str, pwd: str = None):
    """
    Download the data from the HuggingFace Hub.

    Args:
        data_name (str): The name of the data to download.
        pwd (str): The current working directory. Defaults to None.
    """
    # Find PWD
    if pwd is None:
        repo = git.Repo(".", search_parent_directories=True)
        pwd = repo.working_dir

    # Determine the path
    if data_name == "sentence_simplification":
        path = os.path.join(pwd, "data", "raw")
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        raise ValueError(f"The data {data_name} is not available.")

    # Download CSVs
    snapshot_download(
        repo_id="OloriBern/FLDE",
        allow_patterns=[f"{data_name}/*.csv"],
        local_dir=path,
        revision="main",
        repo_type="dataset",
    )

    # Return csv paths (recursively)
    csv_paths = [
        os.path.join(path, data_name, file)
        for file in os.listdir(os.path.join(path, data_name))
        if file.endswith(".csv")
    ]
    return csv_paths
