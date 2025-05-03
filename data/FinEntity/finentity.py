import os
from huggingface_hub import login
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset
import logging
from flame.config import DATA_DIR, LOG_LEVEL


HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
HF_ORGANIZATION = "gtfintechlab"
DATASET = "finentity"
login(HUGGINGFACEHUB_API_TOKEN)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_data_finentity(
    push_to_hub=False, TASK=None, SEED=None, SPLITS=["train", "test"]
):
    try:
        directory_path = DATA_DIR / "FinEntity"
        logger.debug(f"Directory path: {directory_path}")

        dataset = load_dataset("yixuantt/FinEntity", trust_remote_code=True)

        df = dataset["train"]

        df = pd.DataFrame(df)

        hf_dataset = DatasetDict()

        train_df, test_df = train_test_split(df, test_size=0.3, random_state=SEED)

        # train split
        hf_dataset["train"] = Dataset.from_pandas(train_df)

        # Add test split
        hf_dataset["test"] = Dataset.from_pandas(test_df)

        # Push to HF Hub
        if push_to_hub:
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name=str(SEED),
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

            # TODO: push the dataset dict object not the datasets individually

        logger.info(f"Finished processing FinEntity dataset seed : {SEED}")
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing FinEntity dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    SPLITS = ["train", "test"]

    TASK = "FinEntity"

    SEEDS = (5768, 78516, 944601)

    for SEED in list(reversed(SEEDS)):
        huggify_data_finentity(push_to_hub=True, TASK=TASK, SEED=SEED, SPLITS=SPLITS)
