import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login

from flame.config import DATA_DIR, LOG_LEVEL

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HF_ORGANIZATION = "gtfintechlab"
DATASET = "Numclaim"

login(HUGGINGFACEHUB_API_TOKEN)

# NumClaim label mapping - moved from LabelMapper class
NUMCLAIM_LABEL_MAP = {0: "outofclaim", 1: "inclaim"}


# Function to decode numeric labels into text labels
def decode_numclaim_label(label_number):
    """Convert a numeric NumClaim label to its uppercase text representation."""
    return NUMCLAIM_LABEL_MAP.get(label_number, "undefined").upper()


# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_numclaim(push_to_hub=False):
    try:
        directory_path = DATA_DIR / "numclaim_detection"
        logger.debug(f"Directory path: {directory_path}")

        numclaim_train = pd.read_excel(f"{directory_path}/numclaim-train-5768.xlsx")
        numclaim_test = pd.read_excel(f"{directory_path}/numclaim-test-5768.xlsx")

        train_texts = numclaim_train["text"]
        train_labels = numclaim_train["label"]

        test_texts = numclaim_test["text"]
        test_labels = numclaim_test["label"]

        splits = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "context": train_texts,
                        "response": list(map(decode_numclaim_label, train_labels)),
                    }
                ),
                "test": Dataset.from_dict(
                    {
                        "context": test_texts,
                        "response": list(map(decode_numclaim_label, test_labels)),
                    }
                ),
            }
        )

        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

            logger.info("Finished processing Numclaim dataset")
            return splits

    except Exception as e:
        logger.error(f"Error processing Numclaim dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_numclaim(push_to_hub=True)
