import os
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging

DATA_DIRECTORY = Path().cwd() / "data" / "FiQA" / "task2"
HF_TOKEN = os.getenv("HF_TOKEN")  
HF_ORGANIZATION = "gtfintechlab"  
DATASET = "FiQA/Task1"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def huggify_data_fiqa(push_to_hub=False):
    try:
        
        train_file = DATA_DIRECTORY / "train.json"
        test_file = DATA_DIRECTORY / "test.json"
        
        train_data = pd.read_json(train_file)
        test_data = pd.read_json(test_file)

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

        logger.info("Successfully processed and uploaded the FiQA dataset.")
        return splits

    except Exception as e:
        logger.error(f"Error processing FiQA dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_fiqa(push_to_hub=True)
