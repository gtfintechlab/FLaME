from huggingface_hub import hf_hub_upload
import pandas as pd
from datasets import Dataset, DatasetDict
from ferrari.utils.process_qa import process_qa_pairs
from ferrari.utils.zip_to_csv import zip_to_csv

from ferrari.config import DATA_DIR, LOG_DIR, LOG_LEVEL
from ferrari.utils.logging_utils import setup_logger

logger = setup_logger(
    name=__name__, log_file=LOG_DIR / "convfinqahuggify.log", level=LOG_LEVEL
)

HF_ORGANIZATION = "gtfintechlab"
DATASET = "convfinqa"


def huggify_data_convfinqa(push_to_hub=False):
    directory_path = DATA_DIR / "convfinqa"
    logger.debug(f"Directory path: {directory_path}")

    zip_file_path = f"{directory_path}/train.json.zip"
    json_file_name = "train.json"
    csv_file_path = f"{directory_path}/train.csv"
    zip_to_csv(zip_file_path, json_file_name, csv_file_path)

    def csv_to_dataset(file_name):
        df = pd.read_csv(f"{directory_path}/{file_name}")
        processed = process_qa_pairs(df)
        return Dataset.from_dict(
            {"context": processed["input"], "response": processed["output"]}
        )

    convfinqa_datadict = DatasetDict(
        {
            "train": csv_to_dataset("train.csv"),
            "validation": csv_to_dataset("dev.csv"),
        }
    )

    if push_to_hub:
        convfinqa_datadict["train"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-train",
            config_name="train",
            private=True,
        )

        convfinqa_datadict["validation"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-validation",
            config_name="validation",
            private=True,
        )
        FILENAMES = ["train.csv", "val.csv"]
        for FILENAME in FILENAMES:
            hf_hub_upload(
                repo_id=f"{HF_ORGANIZATION}/{DATASET}",
                filename=FILENAME,
                repo_type="dataset",
                commit_message="Add files for ConvFinqa Dataet",
            )
    logger.info("ConvFinqa dataset done")
    return convfinqa_datadict


if __name__ == "__main__":
    huggify_data_convfinqa(push_to_hub=True)
