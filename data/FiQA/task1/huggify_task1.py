import os
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging

DATA_DIRECTORY = Path().cwd() / "data" / "FiQA" / "task1"
HF_TOKEN = os.getenv("HF_TOKEN")  
HF_ORGANIZATION = "gtfintechlab"  
DATASET = "FiQA/Task2"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def huggify_data_task1(push_to_hub=False):
    try:
        
        train_headline_file = DATA_DIRECTORY / "train" / "task1_headline_ABSA_train.json"
        train_post_file = DATA_DIRECTORY / "train" / "task1_post_ABSA_train.json"
        test_headline_file = DATA_DIRECTORY / "test" / "task1_headline_ABSA_test.json"
        test_post_file = DATA_DIRECTORY / "test" / "task1_post_ABSA_test.json"

        train_headline_data = pd.read_json(train_headline_file)
        train_post_data = pd.read_json(train_post_file)
        test_headline_data = pd.read_json(test_headline_file)
        test_post_data = pd.read_json(test_post_file)

        train_data = pd.concat([train_headline_data, train_post_data], ignore_index=True)
        test_data = pd.concat([test_headline_data, test_post_data], ignore_index=True)

        splits = DatasetDict(
            {
                "train": Dataset.from_pandas(train_data),
                "test": Dataset.from_pandas(test_data),
            }
        )

        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HF_TOKEN,
            )

        logger.info("Successfully processed and uploaded Task 1 dataset.")
        return splits

    except Exception as e:
        logger.error(f"Error processing Task 1 dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_task1(push_to_hub=True)
