import os
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging
from superflue import DATA_DIR, LOG_LEVEL

# Set environment variables
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FNXL"
login(HUGGINGFACEHUB_API_TOKEN)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_data(push_to_hub=False):
    try:
        # Path to the dataset
        directory_path = DATA_DIR / "FNXL"
        logger.debug(f"Directory path: {directory_path}")

        # Load the train, test, and validation CSV files
        train_df = pd.read_csv(f"{directory_path}/train_sample.csv")
        test_df = pd.read_csv(f"{directory_path}/test_sample.csv")
        val_df = pd.read_csv(f"{directory_path}/dev_sample.csv")
        # logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Validation shape: {val_df.shape}")
        # logger.info(f"Train columns: {train_df.columns}")
        # logger.info(f"Test columns: {test_df.columns}")
        # logger.info(f"Validation columns: {val_df.columns}")

        splits = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df),
                "validation": Dataset.from_pandas(val_df),
            }
        )

        # Push to Hugging Face Hub if requested
        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

        logger.info("Dataset processing complete.")
        return splits

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data(push_to_hub=True)
