import os
import sys
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict
import logging

SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

HF_TOKEN = os.environ["HF_TOKEN"]
HF_ORGANIZATION = "gtfintechlab"
DATASET = "CausalDetection"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def huggify_data_cd(push_to_hub=False):
    try:
        directory_path = DATA_DIRECTORY / "CausalDetection"
        logger.debug(f"Directory path: {directory_path}")

        cd_train = pd.read_json(f"{directory_path}/train.json")
        cd_test = pd.read_json(f"{directory_path}/test.json")
        cd_val = pd.read_json(f"{directory_path}/valid.json")

        train_tokens = cd_train["tokens"]
        train_tags = cd_train["tags"]

        test_tokens = cd_test["tokens"]
        test_tags = cd_test["tags"]

        val_tokens = cd_val["tokens"]
        val_tags = cd_val["tags"]

        splits = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "tokens": train_tokens,
                        "tags": train_tags,
                    }
                ),
                "test": Dataset.from_dict({"tokens": test_tokens, "tags": test_tags}),
                "validation": Dataset.from_dict(
                    {"tokens": val_tokens, "tags": val_tags}
                ),
            }
        )

        # Push to HF Hub
        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HF_TOKEN,
            )

        logger.info("Finished processing Causal Detection dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing Causal Detection dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_cd(push_to_hub=True)
