import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_upload
import pandas as pd
from datasets import Dataset, DatasetDict
import logging

SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

HF_ORGANIZATION = "gtfintechlab"
DATASET = "finqa"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from src.utils.process_qa import process_qa_pairs

def huggify_data_finqa(push_to_hub=False):
    directory_path = DATA_DIRECTORY / "finqa"
    logger.debug(f"Directory path: {directory_path}")

    finqa_train = pd.read_csv(f"{directory_path}/train.csv")
    finqa_test = pd.read_csv(f"{directory_path}/test.csv")
    finqa_val = pd.read_csv(f"{directory_path}/val.csv")

    processed_train = process_qa_pairs(finqa_train)
    processed_test = process_qa_pairs(finqa_test)
    processed_val = process_qa_pairs(finqa_val)

    splits = DatasetDict(
        {
            "train": Dataset.from_dict({"context": processed_train["input"], "response": processed_train["output"]}),
            "test": Dataset.from_dict({"context": processed_test["input"], "response": processed_test["output"]}),
            "validation": Dataset.from_dict({"context": processed_val["input"], "response": processed_val["output"]}),
        }
    )
    
    if push_to_hub:
            splits["train"].push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}-train",
                config_name="train",
                private=True,
            )
            splits["test"].push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}-test",
                config_name="test",
                private=True,
            )
            splits["validation"].push_to_hub(
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
    return splits

if name == "__main__":
    huggify_data_finqa(push_to_hub=True)