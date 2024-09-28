import os
import sys
from pathlib import Path
from huggingface_hub import login
from datasets import DatasetDict
import logging

# TODO: check if this is the right way to import from the src folder
SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

os.environ["HF_HOME"] = ""
HF_TOKEN = ""
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FinSent"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def huggify_data_finsent(push_to_hub=False):
    try:
        directory_path = DATA_DIRECTORY / "FinSent"
        logger.debug(f"Directory path: {directory_path}")

        splits = DatasetDict({})

        # Push to HF Hub
        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HF_TOKEN,
            )

            # TODO: push the dataset dict object not the datasets individually

        logger.info("Finished processing FinSent dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing FinSent dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_finsent(push_to_hub=True)
