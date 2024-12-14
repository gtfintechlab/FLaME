import os
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging
from superflue import LOG_LEVEL

DATA_DIR = Path().cwd().parent / "task2"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FiQA_Task2"
login(HUGGINGFACEHUB_API_TOKEN)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_data_fiqa(push_to_hub=False):
    try:
        train_file = DATA_DIR / "train.json"
        test_file = DATA_DIR / "test.json"

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
                token=HUGGINGFACEHUB_API_TOKEN,
            )

        logger.info("Successfully processed and uploaded the FiQA dataset.")
        return splits

    except Exception as e:
        logger.error(f"Error processing FiQA dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_fiqa(push_to_hub=True)
