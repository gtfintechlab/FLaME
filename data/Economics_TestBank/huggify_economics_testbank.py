import argparse
import logging

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values
from ferrari.config import DATA_DIR, LOG_LEVEL
from huggingface_hub import login

HF_ORGANIZATION = "gtfintechlab"
DATASET = "Economics_TestBank"

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_economics_testbank(push_to_hub=False):

    config = dotenv_values(".env")
    token = config.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

    login(token)

    df = pd.read_csv(
        str(DATA_DIR) + "/Economics_TestBank/" + "Economics_TestBank.csv",
        index_col=False,
    )
    df = df[["book_name", "chapter_number", "chapter_name", "prompt", "answer"]]

    data_dict = {}
    for col in df.columns:
        data_dict[col] = list(df[col])

    dataset_dict = {}
    dataset_dict["train"] = Dataset.from_dict(data_dict)
    hf_dataset = DatasetDict(dataset_dict)

    path = DATASET
    if push_to_hub:
        hf_dataset.push_to_hub(
            f"{HF_ORGANIZATION}/{path}",
            config_name="main",
            private=True,
        )
        logger.info("Finished processing Economics TestBank ")
        return hf_dataset


if __name__ == "__main__":
    huggify_economics_testbank(push_to_hub=False)
