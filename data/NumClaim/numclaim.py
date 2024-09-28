import os
import logging
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
from superflue.utils.LabelMapper import LabelMapper
from superflue.config import DATA_DIR

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_ORGANIZATION = "gtfintechlab"
DATASET = "Numclaim"

login(HF_TOKEN)

# Include the LabelMapper instantiation for 'numclaim_detection' task
label_mapper = LabelMapper(task="numclaim_detection")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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
                        "response": list(map(label_mapper.decode, train_labels)),
                    }
                ),
                "test": Dataset.from_dict(
                    {
                        "context": test_texts,
                        "response": list(map(label_mapper.decode, test_labels)),
                    }
                ),
            }
        )

        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HF_TOKEN,
            )

            logger.info("Finished processing Numclaim dataset")
            return splits

    except Exception as e:
        logger.error(f"Error processing Numclaim dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_numclaim(push_to_hub=True)
