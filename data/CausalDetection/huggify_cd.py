import os
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
from flame.config import DATA_DIR, LOG_DIR, LOG_LEVEL
from flame.utils.logging_utils import setup_logger

logger = setup_logger(
    name=__name__, log_file=LOG_DIR / "CausalDetectionhuggify.log", level=LOG_LEVEL
)


# TODO: Use logging helper function; get the HF creds from .env
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
HF_ORGANIZATION = "gtfintechlab"
DATASET = "CausalDetection"
login(HUGGINGFACEHUB_API_TOKEN)


def huggify_data_cd(push_to_hub=False):
    try:
        directory_path = DATA_DIR / "CausalDetection"
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
                token=HUGGINGFACEHUB_API_TOKEN,
            )

        logger.info("Finished processing Causal Detection dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing Causal Detection dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_cd(push_to_hub=True)
