from huggingface_hub import hf_hub_upload
import pandas as pd
from datasets import Dataset
import logging
from ferrari.utils.process_qa import process_qa_pairs
from ferrari.utils.zip_to_csv import zip_to_csv
from ferrari.config import DATA_DIR, LOG_LEVEL

HF_ORGANIZATION = "gtfintechlab"
DATASET = "finqa"

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_data_finqa(push_to_hub=False):
    directory_path = DATA_DIR / "finqa"
    logger.debug(f"Directory path: {directory_path}")

    zip_to_csv(
        f"{directory_path}/train.json.zip", "train.json", f"{directory_path}/train.csv"
    )

    def csv_to_dataset(file_name):
        df = pd.read_csv(f"{directory_path}/{file_name}")
        processed = process_qa_pairs(df)
        return Dataset.from_dict(
            {"context": processed["input"], "response": processed["output"]}
        )

    finqa_datadict = {
        split: csv_to_dataset(f"{split}.csv") for split in ["train", "test", "dev"]
    }
    if push_to_hub:
        finqa_datadict["train"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-train",
            config_name="train",
            private=True,
        )
        finqa_datadict["test"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-test",
            config_name="test",
            private=True,
        )
        finqa_datadict["validation"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-validation",
            config_name="validation",
            private=True,
        )
        FILENAMES = ["train.csv", " test.csv", "val.csv"]
        for FILENAME in FILENAMES:
            hf_hub_upload(
                repo_id=f"{HF_ORGANIZATION}/{DATASET}",
                filename=FILENAME,
                repo_type="dataset",
                commit_message="Add files for Finqa Dataet",
            )
    logger.info("Finqa dataset done")
    return finqa_datadict


if __name__ == "__main__":
    huggify_data_finqa(push_to_hub=True)
