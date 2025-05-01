import logging
import argparse

from datasets import DatasetDict, load_dataset
from dotenv import dotenv_values
from superflue.config import DATA_DIR, LOG_LEVEL
from huggingface_hub import login

HF_ORGANIZATION = "gtfintechlab"
DATASET = "bizbench"

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


# TODO: huggify_bizbench needs to be fixed
@DeprecationWarning("This is not correct -- needs to be fixed")
def huggify_bizbench(push_to_hub=False):
    """
    Process bizbench dataset and optionally push to HuggingFace Hub.

    Args:
        push_to_hub (bool): Whether to push the dataset to HuggingFace Hub
    """
    # Load only the HF token from .env
    config = dotenv_values(".env")
    token = config.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

    login(token)

    try:
        directory_path = DATA_DIR / "bizbench"
        logger.debug(f"Directory path: {directory_path}")

        logger.info("Loading dataset from HuggingFace Hub")
        dataset = load_dataset("", trust_remote_code=True)
        if not dataset:
            raise ValueError("Failed to load dataset from HuggingFace Hub")

        hf_dataset = DatasetDict()

        # Process each split
        splits = {"train": "train", "validation": "val", "test": "test"}

        for target_split, source_split in splits.items():
            if source_split in dataset:
                logger.debug(f"Processing {source_split} split")
                hf_dataset[target_split] = dataset[source_split]
            else:
                logger.warning(f"Split '{source_split}' not found in source dataset")

        # Push to HF Hub
        if push_to_hub:
            logger.info(
                f"Pushing dataset to HuggingFace Hub at {HF_ORGANIZATION}/{DATASET}"
            )
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=token,
            )

        logger.info("Finished processing bizbench")
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing bizbench dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process bizbench dataset")
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push dataset to HuggingFace Hub"
    )
    args = parser.parse_args()

    huggify_bizbench(push_to_hub=args.push_to_hub)
