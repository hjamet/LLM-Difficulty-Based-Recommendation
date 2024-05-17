# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
from __future__ import annotations

import shutil
import logging
import pandas as pd
import os
import json
import subprocess
import mysql.connector
from zipfile import ZipFile
import time
import git
import huggingface_hub
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------- #
#                               CLASS DEFINITION                               #
# ---------------------------------------------------------------------------- #


class DataManager:
    """This class is responsible for downloading data from the internet.
    > Data comes from [Github](https://github.com/michalis0/project_johannesDavid2023/tree/main/LingoRank/Data) and has been uploaded on [Kaggle](https://www.kaggle.com/datasets/oloribern/flde-unil).
    """

    def __init__(self):
        """This method initializes the class.
        It will load kaggle token from the .kaggle file if it exists.
        """
        # Set logging level
        self.logger = logging.getLogger("DataManager")
        self.logger.info("Initializing data manager...")

        # Set pwd
        self.pwd = git.Repo(".", search_parent_directories=True).working_dir

        # Connect to HuggingFace
        try:
            # Log in to huggingface
            try:
                with open(os.path.join(self.pwd, ".huggingface_key"), "r") as f:
                    api_key = f.read()
            except:
                api_key = input("Please enter your huggingface API key : ")
                with open(os.path.join(self.pwd, ".huggingface_key"), "w") as f:
                    f.write(api_key)

            huggingface_hub.login(token=api_key)
            print("Logged in to huggingface hub")
        except Exception as e:
            pass

    def download(self, data_name: str = "all", force: bool = False) -> None:
        """This method downloads the data from kaggle.

        Args:
            data_name (str, optional): Which data to download. Can be either "all", "recommandation" or "difficulty_estimation". Defaults to "all".
        """
        path_temp = os.path.join(self.pwd, "temp", "Data")
        if os.path.exists(path_temp):
            shutil.rmtree(path_temp)

        # Download data
        ## Difficulty estimation data
        if data_name in ["all", "difficulty_estimation"]:
            self.logger.info("Downloading data from kaggle.")
            # Check if data has been downloaded
            path_difficulty = os.path.join(
                self.pwd, "data", "raw", "difficulty_estimation"
            )
            if os.path.exists(path_difficulty):
                if not force:
                    self.logger.warning("Data already downloaded.")
                else:
                    self.logger.info("Removing old data.")
                    shutil.rmtree(path_difficulty)
            if not os.path.exists(path_difficulty):
                snapshot_download(
                    repo_id="OloriBern/FLDE",
                    allow_patterns=["Data/*.csv"],
                    local_dir=os.path.join(self.pwd, "temp"),
                    revision="main",
                    repo_type="dataset",
                )
                # Move data to correct folder
                self.logger.info("Moving data to correct folder.")
                ## Move the content of the temp/Data folder to data
                shutil.copytree(path_temp, path_difficulty)
                ## Remove temp folder
                shutil.rmtree(path_temp)

        ## Recommander system data
        if data_name in ["all", "recommandation"]:
            # Check if data has been downloaded
            path_zeeguu = os.path.join(self.pwd, "data", "raw", "zeeguu")
            if os.path.exists(path_zeeguu):
                if not force:
                    self.logger.warning("Data already downloaded.")
                else:
                    self.logger.info("Removing old data.")
                    shutil.rmtree(path_zeeguu)
            if not os.path.exists(path_zeeguu):
                subprocess.run(
                    [
                        "git",
                        "lfs",
                        "clone",
                        "https://github.com/zeeguu/data-releases.git",
                        path_zeeguu,
                    ]
                )
                self.logger.info("Zeeguu data downloaded.")

        self.logger.info("Data downloaded.")

    def get_data(self, dataset: str = "all", type_set: str = "train") -> pd.DataFrame:
        """Returns the requested data in the form of a pandas dataframe.

        Args:
            dataset (str, optional): Which dataset to return. Can be either "french_difficulty", "ljl", or "sentences" . Defaults to "all".
            type_set (str, optional): Which data set to return. Could be either "train" or "test". Defaults to "train".

        Returns:
            pd.DataFrame: The requested dataset.
        """
        path = os.path.join(self.pwd, "data", "raw", "difficulty_estimation")

        # Check if data has been downloaded
        if not os.path.exists(path):
            self.logger.warning("Data has not been downloaded yet. Downloading now.")
            self.download()

        if dataset in ["french_difficulty", "ljl", "sentences"]:
            try:
                return pd.read_csv(f"{path}/{dataset}_{type_set}.csv")
            except Exception as e:
                self.logger.warning(f"Could not find dataset {dataset}_{type_set}.csv.")
                self.logger.warning(e)
                self.logger.warning("Returning None.")
                return None
        elif dataset == "all":
            try:
                return pd.concat(
                    [
                        pd.read_csv(f"{path}/{dataset}_{type_set}.csv")
                        .assign(dataset=dataset)
                        .assign(type_set=type_set)
                        for dataset in ["french_difficulty", "ljl", "sentences"]
                    ]
                )
            except Exception as e:
                self.logger.warning("Could not find all datasets.")
                self.logger.warning(e)
                self.logger.warning("Returning None.")
                return None

    def get_data_ready_for_fine_tuning(
        self,
        dataset: str = "all",
        save: bool = True,
        type_set: str = "train",
        context: str = "empty",
        model_name: str = "gpt-3.5-turbo",
    ) -> pd.DataFrame:
        """Returns the requested data in the form of a pandas dataframe.

        Args:
            dataset (str, optional): Which dataset to return. Can be either "french_difficulty", "ljl", or "sentences" . Defaults to "all".
            save (bool, optional): Whether to save the data to a json file. Defaults to True.
            type_set (str, optional): Which data set to return. Could be either "train" or "test". Defaults to "train".
            context (str, optional): Which context to use. Can be either "teacher", "CECRL" or "empty". Defaults to "empty".

        Returns:
            pd.DataFrame: The requested dataset.
        """
        # Get data
        data = self.get_data(dataset=dataset, type_set=type_set)

        # Remove ljl because of different labels
        # TODO: Find a solution for this
        if dataset == "all":
            data = data[data["dataset"] != "ljl"]

            # Drop columns
            data = data.drop(columns=["dataset", "type_set"])

            # Balance classes
            data = data.groupby("difficulty").head(
                min(data["difficulty"].value_counts())
            )

        # Shuffle data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Remove '\n' from sentences
        data["sentence"] = data["sentence"].str.replace("\n", "")

        # Rename columns
        data.rename(columns={"sentence": "prompt", "label": "completion"}, inplace=True)

        # Define context
        if context == "teacher":
            context_content = f"You are a French teacher who must assign to a text a level of difficulty among {len(data['difficulty'].unique())}. Here's the text:"
        elif context == "CECRL":
            context_content = "Vous êtes un évaluateur linguistique utilisant le Cadre européen commun de référence pour les langues (CECRL). Votre mission est d'attribuer une note de compétence linguistique à ce texte, en utilisant les niveaux du CECRL, allant de A1 (débutant) à C2 (avancé/natif). Évaluez ce texte et attribuez-lui la note correspondante du CECRL."
        elif context == "empty":
            context_content = ""
        else:
            self.logger.warning("Context not recognized. Using empty context.")
            context = "empty"
            context_content = ""

        formatted_data = []

        for prompt, completion in data.values:
            entry = {}
            if "gpt" in model_name:
                entry["messages"] = [
                    {
                        "role": "system",
                        "content": context_content,
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            elif "davinci" in model_name or "babbage" in model_name:
                entry["prompt"] = context_content + "\n\n" + prompt
                entry["completion"] = completion
            formatted_data.append(entry)

        # Save data
        if save:
            path = os.path.join(self.pwd, "data", "processed", "DataManager")
            # Create folder if it does not exist
            if not os.path.exists(path):
                os.makedirs(path)

            # Save data
            with open(
                os.path.join(
                    path,
                    f"{type_set}_{dataset}_{context}_{model_name}_prepared_for_fine_tuning.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                for entry in formatted_data:
                    json_str = json.dumps(entry, ensure_ascii=False)
                    f.write(json_str + "\n")

        return formatted_data

    def push_to_mysql(
        self, mysql_user: str = "zeeguu", mysql_password: str = "zeeguu"
    ) -> None:
        """This method uploads the data downloaded from zeeguu to a mysql database.

        CAUTION: This method assumes that you have a mysql instance running on your computer.
        This method will also REMOVE EXISTING DATABASE named "zeeguu" if it exists.
        """
        # Check if data has been downloaded
        if not os.path.exists("data/zeeguu"):
            self.logger.warning("Data has not been downloaded yet. Downloading now.")
            self.download(data_name="recommandation")

        # Define variables
        host = "127.0.0.1"
        new_database = "zeeguu"

        # Connect to mysql
        try:
            connection = mysql.connector.connect(
                host=host, user=mysql_user, password=mysql_password
            )
        except Exception as e:
            self.logger.warning("Could not connect to mysql.")
            self.logger.warning(e)
            self.logger.warning("Please make sure you have a mysql instance running.")
            return
        cursor = connection.cursor()

        # Create database zeeguu
        cursor.execute(f"DROP DATABASE IF EXISTS {new_database}")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {new_database}")
        cursor.close()
        connection.close()

        # Unzip data
        for file in os.listdir(os.path.join("data", "zeeguu")):
            if file.endswith(".zip"):
                file_path: str = os.path.join("data", "zeeguu", file)
                with ZipFile(file_path, "r") as zip:
                    zip.extractall(os.path.join("data", "zeeguu"))
                    self.logger.info(f"Unzipped {file_path}")

        # Insert data
        for file in os.listdir(os.path.join("data", "zeeguu")):
            if file.endswith(".sql"):
                file_path: str = os.path.join("data", "zeeguu", file)
                with open(
                    file_path,
                    "r",
                ) as f:
                    available = False
                    while not available:
                        try:
                            connection = mysql.connector.connect(
                                host=host,
                                user=mysql_user,
                                password=mysql_password,
                                database=new_database,
                            )
                            cursor = connection.cursor()
                            available = True
                        except:
                            self.logger.warning(
                                "Could not connect to mysql. Trying again in 1 seconds."
                            )
                            time.sleep(1)
                    script = f.read()
                    cursor.execute(script)
                    cursor.fetchall()
                    cursor.close()
                    connection.close()
                    self.logger.info(f"Inserted {file_path}")


# ---------------------------------------------------------------------------- #
#                                     TESTS                                    #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_manager = DataManager()
    data_manager.download()
    data_manager.get_data()
    data_manager.get_data_ready_for_fine_tuning()
