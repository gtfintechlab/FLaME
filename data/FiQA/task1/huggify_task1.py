import json
import os
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging

DATA_DIR = Path().cwd().parent / "task1"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FiQA_Task1"
login(HUGGINGFACEHUB_API_TOKEN)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_and_process_json(file_path):
    """Load a JSON file and process it into a DataFrame."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    records = []
    for item in data.values():
        sentence = item["sentence"]
        for info in item["info"]:
            records.append(
                {
                    "sentence": sentence,
                    "snippets": " ".join(info["snippets"]),  # Convert list to a string
                    "target": info["target"],
                    "sentiment_score": info["sentiment_score"],
                    "aspects": " ".join(info["aspects"]),  # Convert list to a string
                }
            )

    return pd.DataFrame(records)


def huggify_data_fiqa(push_to_hub=False):
    """
    This dataset involves analyzing English financial texts to detect and classify target
    aspects from a predefined list and predict sentiment scores for each target aspect.
    Sentiment scores range from -1 (negative) to 1 (positive), and the dataset evaluates precision,
    recall, F1-score, and MSE for sentiment prediction.

    Source: https://huggingface.co/datasets/ChanceFocus/fiqa-sentiment-classification
    https://sites.google.com/view/fiqa/home
    """
    try:
        train_file = DATA_DIR / "train.json"
        test_file = DATA_DIR / "test.json"
        valid_file = DATA_DIR / "valid.json"

        train_data = pd.read_json(train_file)
        test_data = pd.read_json(test_file)
        valid_data = pd.read_json(valid_file)

        train_data = load_and_process_json(train_file)
        test_data = load_and_process_json(test_file)
        valid_data = load_and_process_json(valid_file)

        splits = DatasetDict(
            {
                "train": Dataset.from_pandas(train_data),
                "test": Dataset.from_pandas(test_data),
                "validation": Dataset.from_pandas(valid_data),
            }
        )

        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

        logger.info("Successfully processed and uploaded the FiQA Task 1 dataset.")
        return splits

    except Exception as e:
        logger.error(f"Error processing FiQA Task 1 dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_fiqa(push_to_hub=True)
