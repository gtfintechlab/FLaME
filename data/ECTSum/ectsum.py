import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_upload
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

# TODO: check if this is the right way to import from the src folder
SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from datasets import Dataset, DatasetDict
import logging

HF_ORGANIZATION = "gtfintechlab"
DATASET = "ECTSum"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def huggify_data_ectsum(push_to_hub=False):
    try:
        directory_path = DATA_DIRECTORY / "ECTSum"
        logger.debug(f"Directory path: {directory_path}")

        ect_sum_train = pd.read_csv(f"{directory_path}/train.csv")
        ect_sum_test = pd.read_csv(f"{directory_path}/test.csv")
        ect_sum_val = pd.read_csv(f"{directory_path}/val.csv")

        train_input = ect_sum_train["input"]
        train_output = ect_sum_train["output"]

        test_input = ect_sum_test["input"]
        test_output = ect_sum_test["output"]

        val_input = ect_sum_val["input"]
        val_output = ect_sum_val["output"]

        splits = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "context": train_input,
                        "response": train_output,
                    }
                ),
                "test": Dataset.from_dict(
                    {"context": test_input, "response": test_output}
                ),
                "validation": Dataset.from_dict(
                    {"context": val_input, "response": val_output}
                ),
            }
        )

        # Push to HF Hub
        if push_to_hub:
            splits["train"].push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}-train",
                config_name="train",
                private=True,
            )
            splits["test"].push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}-test",
                config_name="test",
                private=True,
            )
            splits["validation"].push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}-validation",
                config_name="validation",
                private=True,
            )
            
            # TODO: push the dataset dict object not the datasets individually

            FILENAMES = ["train.csv", " test.csv", "val.csv"]
            for FILENAME in FILENAMES:
                hf_hub_upload(
                    repo_id=f"{HF_ORGANIZATION}/{DATASET}",
                    filename=FILENAME,
                    repo_type="dataset",
                    commit_message="Add CSVs for ECTSum dataset",
                )
        logger.info("Finished processing ECTSum dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing ECT Sum dataset: {str(e)}")
        raise e


if name == "__main__":
    huggify_data_ectsum(push_to_hub=True)


#######
# TODO: write the doc for the command to generate the ECT dataset
# README.md

# # ECTSum Dataset
# in order to build the dataset run `python /data/ectsum.py` from the root directory of the project
