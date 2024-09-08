import os
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging

DATA_DIRECTORY = Path().cwd().parent / "task1"
HF_TOKEN = os.getenv("HF_TOKEN")  
HF_ORGANIZATION = "gtfintechlab"  
DATASET = "FiQA_Task1_headlines"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def huggify_data_task1(push_to_hub=False):
    try:
        train_headline_file = DATA_DIRECTORY / "train" / "task1_headline_ABSA_train.json"
        #train_post_file = DATA_DIRECTORY / "train" / "task1_post_ABSA_train.json"
        test_headline_file = DATA_DIRECTORY / "test" / "task1_headline_ABSA_test.json"
        #test_post_file = DATA_DIRECTORY / "test" / "task1_post_ABSA_test.json"
        
        train_headline_data = pd.read_json(train_headline_file)
        #train_post_data = pd.read_json(train_post_file)
        test_headline_data = pd.read_json(test_headline_file)
        #test_post_data = pd.read_json(test_post_file)

        #train_data = pd.concat([train_headline_data, train_post_data], ignore_index=True)
        #test_data = pd.concat([test_headline_data, test_post_data], ignore_index=True)
        
        splits_headline_data = DatasetDict(
            {
                "train": Dataset.from_pandas(train_headline_data),
                "test": Dataset.from_pandas(test_headline_data),
            }
        )

        # splits_post_data = DatasetDict(
        #     {
        #         "train": Dataset.from_pandas(train_post_data),
        #         "test": Dataset.from_pandas(test_post_data),
        #     }
        # )

        # splits = DatasetDict(
        #     {
        #         "train": Dataset.from_pandas(train_data),
        #         "test": Dataset.from_pandas(test_data),
        #     }
        # )
        if push_to_hub:
            splits_headline_data.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HF_TOKEN,
            )
            # splits_post_data.push_to_hub(
            #     f"{HF_ORGANIZATION}/{DATASET}",
            #     config_name="main",
            #     private=True,
            #     token=HF_TOKEN,
            # )
            # splits.push_to_hub(
            #     f"{HF_ORGANIZATION}/{DATASET}",
            #     config_name="main",
            #     private=True,
            #     token=HF_TOKEN,
            # )

        logger.info("Successfully processed and uploaded Task 1 headlines dataset.")
        #return splits_headline_data, splits_post_data
        return splits_headline_data

    except Exception as e:
        logger.error(f"Error processing Task 1 headlines dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_task1(push_to_hub=True)