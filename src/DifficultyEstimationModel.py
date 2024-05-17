# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import openai
import logging
import time
import pandas as pd
import numpy as np
import json
import tqdm
import sklearn.metrics
from matplotlib import pyplot as plt
from typing import Tuple
import os
import git

# ---------------------------------------------------------------------------- #
#                               CLASS DEFINITION                               #
# ---------------------------------------------------------------------------- #


class DifficultyEstimationModel:
    """This class implements a model based on different versions of GPT to estimate the difficulty of a text."""

    def __init__(self, model: str = "gpt-3.5-turbo-1106", model_id: str = None) -> None:
        """Instantiates the model

        Args:
            model (str, optional): The model used as a basis by this class. Can be either "gpt-3.5-turbo-0613"
            model_id (str, optional): The id of a previously trained model to use. If None, the model will be trained. Defaults to None.
        """
        # Set logging level
        self.logger = logging.getLogger("DifficultyEstimationModel")
        self.logger.info("Initializing model {}...".format(model))

        self.model = model
        self.model_id = model_id  # This id will be used to retrieve the model from openai once it has been trained

        # Set pwd
        self.pwd = git.Repo(".", search_parent_directories=True).working_dir

        # Connect to openai
        self.__connect_to_openai()

    def __connect_to_openai(self):
        """This method will connect to openai with registered key if any. Otherwise, it will ask for it."""
        # Try to read saved key
        try:
            with open(".openai_key", "r") as key_file:
                key = key_file.read()
                openai.api_key = key
        except FileNotFoundError:
            # Ask for key
            key = input("Please enter your OpenAI API key:")
            openai.api_key = key
            # Save key
            with open(".openai_key", "w") as key_file:
                key_file.write(key)

    def fine_tune(self, file_name: str) -> None:
        """This method will fine tune the model on the given file.

        Args:
            file_name (str): The name of the json file to use for fine tuning. No need to specify the path or extension.
        """
        # Push training data to openai
        file = openai.File.create(
            file=open(
                os.path.join("data", "processed", "DataManager", f"{file_name}.json")
            ),
            purpose="fine-tune",
        )

        # Start fine tuning
        file_being_processed = True
        self.logger.info("Waiting for file to be processed...")
        while file_being_processed:
            # Check if file is being processed
            try:
                job = openai.FineTuningJob.create(
                    training_file=file["id"], model=self.model
                )
                file_being_processed = False
            except Exception as e:
                time.sleep(5)

        self.logger.info("File processed ! Starting fine tuning...")
        # Train until finished
        while openai.FineTuningJob.retrieve(job["id"]).status != "succeeded":
            time.sleep(10)

        # Save model
        self.logger.info("Fine tuning finished ! Saving model...")
        self.trained_model = openai.FineTuningJob.retrieve(job["id"])[
            "fine_tuned_model"
        ]
        self.model_id = self.trained_model
        self.__save_model(file_name, self.trained_model)
        self.logger.info("Model saved !")

    def __save_model(self, file_name: str, model_id: str):
        """This method saves the information needed to use the model in a csv file.

        Args:
            file_name (str): The name of the file used for training.
            model_id (str): The "fine_tuned_model" property of the training job.
        """
        # Try to load existing file
        path = os.path.join(
            "results", "DifficultyEstimationModel", "trained_models.csv"
        )

        # Create folder if it doesn't exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Load file
        try:
            df = pd.read_csv(path)
            self.logger.info("Loaded trained_models.csv.")
        except FileNotFoundError:
            self.logger.warning("Could not find trained_models.csv. Creating it now.")
            df = pd.DataFrame(columns=["file_name", "model_id", "datetime"], index=[])

        # Add new row
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[file_name, model_id, time.time()]],
                    columns=["file_name", "model_id", "datetime"],
                ),
            ]
        )

        # Save file
        df.to_csv(path, index=False)

    def predict(self, file_name: str, save: bool = True) -> pd.DataFrame:
        """This method will use the model to predict the difficulty of the given file.
        Note that the model needs to be trained first.

        Args:
            file_name (str): The name of the file to use for prediction. No need to specify the path or extension.
        """
        # Check if model has been trained
        if self.model_id is None:
            self.logger.warning(
                "Model has not been trained yet. Please train it first."
            )
            return

        # Read file
        with open(
            os.path.join(
                self.pwd, "data", "processed", "DataManager", f"{file_name}.json"
            ),
            "r",
        ) as file:
            data = file.read()

        # Transform every lines json into one json
        data = data.replace("\n", ",")
        data = data.replace("}{", "},{")
        data = f"[{data[:-1]}]"

        # Parse json
        data = json.loads(data)

        # Extract text
        context_list = []
        user_list = []
        assistant_list = []

        # Go trough every line
        for item in data:
            if "gpt" in self.model:
                messages = item["messages"]
                context_list.append(messages[0]["content"])
                user_list.append(messages[1]["content"])
                assistant_list.append(messages[2]["content"])
            elif "davinci" in self.model or "babbage" in self.model:
                context_list.append(item["prompt"].split("\n\n")[0])
                user_list.append(item["prompt"].split("\n\n")[1])
                assistant_list.append(item["completion"])

        # Create dataframe
        df = pd.DataFrame(
            {"context": context_list, "user": user_list, "assistant": assistant_list}
        )

        # Calculating results for each line
        results = []
        for row in tqdm.tqdm(df.iterrows(), total=len(df)):
            responseOK = False
            while not responseOK:
                try:
                    if "gpt" in self.model:
                        completion = openai.ChatCompletion.create(
                            model=self.model_id,
                            messages=[
                                {"role": "system", "content": row[1]["context"]},
                                {"role": "user", "content": row[1]["user"]},
                            ],
                        )
                        res = completion.choices[0].message["content"]
                    elif "davinci" in self.model or "babbage" in self.model:
                        prompt = (
                            row[1]["context"] + "\n\n" + row[1]["user"]
                            if row[1]["context"] != ""
                            else row[1]["user"]
                        )
                        completion = openai.Completion.create(
                            model=self.model_id,
                            prompt=prompt,
                        )
                        res = completion.choices[0].text
                    results.append(res)
                    responseOK = True
                except:
                    print("Rate limit error, waiting 10 seconds")
                    time.sleep(10)

        df["predictions"] = results

        # Save results
        if save:
            df.to_csv(
                os.path.join(
                    self.pwd,
                    "results",
                    "DifficultyEstimationModel",
                    f"{file_name}_predictions.csv",
                ),
                index=False,
            )

        return df

    def compute_metrics(self, file_name: str, save: bool = True) -> pd.Series:
        """This method will calculate the following metrics for the given file:
        - Accuracy
        - F1 Score (Micro)
        - F1 Score (Macro)
        - Precision (Micro)
        - Precision (Macro)
        - Recall (Micro)
        - Recall (Macro)

        If predictions have already been calculated, they will be loaded. Otherwise, they will be calculated.

        Args:
            file_name (str) : The test file name to use for metrics computation. No need to specify the path or extension.
            save (bool, optional): Whether to save the results to a csv file. Defaults to True.

        Returns:
            pd.Series: The metrics results.
        """
        # Make or load predictions if available
        try:
            results = pd.read_csv(
                os.path.join(
                    self.pwd,
                    "results",
                    "DifficultyEstimationModel",
                    f"{file_name}_predictions.csv",
                )
            )
        except FileNotFoundError:
            results = self.predict(
                file_name, save=True
            )  # Columns : context (input 1), user (input 2), assistant (labels), predictions (real predictions)

        # Compute metrics
        metrics = pd.Series(
            {
                "accuracy": sklearn.metrics.accuracy_score(
                    results["assistant"], results["predictions"]
                ),
                "f1_micro": sklearn.metrics.f1_score(
                    results["assistant"], results["predictions"], average="micro"
                ),
                "f1_macro": sklearn.metrics.f1_score(
                    results["assistant"], results["predictions"], average="macro"
                ),
                "precision_micro": sklearn.metrics.precision_score(
                    results["assistant"], results["predictions"], average="micro"
                ),
                "precision_macro": sklearn.metrics.precision_score(
                    results["assistant"], results["predictions"], average="macro"
                ),
                "recall_micro": sklearn.metrics.recall_score(
                    results["assistant"], results["predictions"], average="micro"
                ),
                "recall_macro": sklearn.metrics.recall_score(
                    results["assistant"], results["predictions"], average="macro"
                ),
            }
        )

        # Round results
        metrics = metrics.round(4)

        # Save results
        if save:
            metrics.to_csv(
                os.path.join(
                    self.pwd,
                    "results",
                    "DifficultyEstimationModel",
                    f"{file_name}_metrics.csv",
                ),
                index=True,
            )

        return metrics

    def compute_confusion_matrix(
        self, file_name: str, save: bool = True
    ) -> Tuple[pd.DataFrame, plt.plot]:
        """This method will calculate the confusion matrix for the given file.

        If predictions have already been calculated, they will be loaded. Otherwise, they will be calculated.

        Args:
            file_name (str) : The test file name to use for metrics computation. No need to specify the path or extension.
            save (bool, optional): Whether to save the results to a csv file. Defaults to True.

        Returns:
            pd.DataFrame: The confusion matrix.
            plt: The confusion matrix plot.
        """
        # Make or load predictions if available
        try:
            results = pd.read_csv(f"results/{file_name}_predictions.csv")
        except FileNotFoundError:
            results = self.predict(
                file_name, save=True
            )  # Columns : context (input 1), user (input 2), assistant (labels), predictions (real predictions)

        # Compute confusion matrix
        ## Replace predictions which are not in assistant's vocabulary with -1
        results["predictions"][
            ~results["predictions"].astype(str).isin(results["assistant"].astype(str))
        ] = -1
        ## Sort values of assistant's and predictions' vocabulary based on assistant's vocabulary
        assistant = results["assistant"].astype(str).sort_values()
        predictions = results["predictions"].astype(str).sort_values()

        confusion_matrix = sklearn.metrics.confusion_matrix(assistant, predictions)
        # if predictions.iloc[0] == -1:
        #     confusion_matrix = confusion_matrix[1:]
        # else:
        #     # Add a 0 column
        #     confusion_matrix = np.insert(confusion_matrix, 0, 0, axis=1)

        # Plot confusion matrix
        plt.figure(figsize=(10, 10))
        if not "ljl" in file_name:
            ## Add labels to axes
            plt.xticks(
                range(len(confusion_matrix[0])),
                assistant.unique().tolist()
                + ["Incorrect"]
                * (len(confusion_matrix) - len(assistant.unique().tolist())),
            )
            plt.yticks(
                range(len(confusion_matrix)),
                assistant.unique().tolist()
                + ["Incorrect"]
                * (len(confusion_matrix) - len(assistant.unique().tolist())),
            )
        else:
            pass

        # Title
        plt.title(f"Confusion matrix for {file_name}")

        plt.imshow(confusion_matrix, cmap="Blues")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")

        return confusion_matrix, plt


if __name__ == "__main__":
    model = DifficultyEstimationModel()
    model.fine_tune("train_french_difficulty_prepared_for_fine_tuning")
    model.fine_tune("train_ljl_prepared_for_fine_tuning")
    model.fine_tune("train_sentences_prepared_for_fine_tuning")
