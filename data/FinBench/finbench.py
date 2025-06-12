import logging
import os

from datasets import DatasetDict, load_dataset
from huggingface_hub import login

from flame.config import DATA_DIR, LOG_LEVEL

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FinBench"
login(HUGGINGFACEHUB_API_TOKEN)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_finbench(push_to_hub=False, TASK=None):
    try:
        directory_path = DATA_DIR / "FinBench"
        logger.debug(f"Directory path: {directory_path}")

        dataset = load_dataset("yuweiyin/FinBench", trust_remote_code=True)

        train_set = dataset["train"] if "train" in dataset else []
        validation_set = dataset["validation"] if "validation" in dataset else []
        test_set = dataset["test"] if "test" in dataset else []

        hf_dataset = DatasetDict()

        # train split
        hf_dataset["train"] = train_set

        # Add test split
        hf_dataset["test"] = test_set

        # Add val split
        hf_dataset["validation"] = validation_set

        # Push to HF Hub
        if push_to_hub:
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

        logger.info("Finished processing FinBench")
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing FinBench dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    TASK = "FinBench"
    huggify_finbench(push_to_hub=True, TASK=TASK)
