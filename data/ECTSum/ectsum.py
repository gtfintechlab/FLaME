import os
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging
from flame.config import DATA_DIR, LOG_LEVEL

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HF_ORGANIZATION = "gtfintechlab"
DATASET = "ECTSum"
login(HUGGINGFACEHUB_API_TOKEN)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_data_ectsum(push_to_hub=False):
    try:
        directory_path = DATA_DIR / "ECTSum"
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
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

            # TODO: push the dataset dict object not the datasets individually

        logger.info("Finished processing ECTSum dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing ECT Sum dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_ectsum(push_to_hub=True)


#######
# TODO: write the doc for the command to generate the ECT dataset
# README.md

# # ECTSum Dataset
# in order to build the dataset run `python /data/ectsum.py` from the root directory of the project
