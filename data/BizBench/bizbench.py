import os
from dotenv import dotenv_values
from huggingface_hub import login
from datasets import DatasetDict, Dataset, load_dataset
import logging
from ferrari.config import DATA_DIR, LOG_LEVEL
import argparse

HF_ORGANIZATION = "gtfintechlab"
DATASET = "BizBench"

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_bizbench(push_to_hub=False):
    # Load environment variables from .env file
    config = dotenv_values(".env")
    token = config.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")
    
    login(token)
    try:
        directory_path = DATA_DIR / "BizBench"
        logger.debug(f"Directory path: {directory_path}")

        dataset = load_dataset("kensho/bizbench", trust_remote_code=True)
        train_set = dataset["train"] if "train" in dataset else []
        validation_set = dataset["validation"] if "validation" in dataset else []
        test_set = dataset["test"] if "test" in dataset else []

        hf_dataset = DatasetDict()

        # train split
        if train_set: hf_dataset["train"] = train_set

        # Add test split
        if test_set: hf_dataset["test"] = test_set

        # Add val split
        if validation_set: hf_dataset["validation"] = validation_set

        # Push to HF Hub
        if push_to_hub:
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=token,
            )

        logger.info("Finished processing BizBench")
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing BizBench dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process BizBench dataset')
    parser.add_argument('--push-to-hub', action='store_true', help='Push dataset to HuggingFace Hub')
    args = parser.parse_args()
    
    huggify_bizbench(push_to_hub=args.push_to_hub)
