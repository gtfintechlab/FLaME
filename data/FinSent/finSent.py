import os
from huggingface_hub import login
from datasets import DatasetDict
import logging
from superflue import DATA_DIR, LOG_LEVEL

os.environ["HF_HOME"] = ""
HUGGINGFACEHUB_API_TOKEN = ""
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FinSent"
login(HUGGINGFACEHUB_API_TOKEN)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_data_finsent(push_to_hub=False):
    try:
        directory_path = DATA_DIR / "FinSent"
        logger.debug(f"Directory path: {directory_path}")

        splits = DatasetDict({})

        # Push to HF Hub
        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

            # TODO: push the dataset dict object not the datasets individually

        logger.info("Finished processing FinSent dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing FinSent dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_finsent(push_to_hub=True)
