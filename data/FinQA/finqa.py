from huggingface_hub import hf_hub_upload
import pandas as pd
from datasets import Dataset
import logging
from flame.utils.miscellaneous import zip_to_csv
from flame.config import DATA_DIR, LOG_LEVEL

HF_ORGANIZATION = "gtfintechlab"
DATASET = "finqa"

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def process_qa_pairs(data):
    """Process question-answer pairs from FinQA data.

    Extracts pre_text, post_text, table_ori, and QA pairs from the dataset.

    Args:
        data: DataFrame containing FinQA data

    Returns:
        DataFrame with processed QA pairs
    """
    pre_text, post_text, table_ori = [], [], []
    question_0, question_1, answer_0, answer_1 = [], [], [], []

    for _, row in data.iterrows():
        pre_text.append(row["pre_text"])
        post_text.append(row["post_text"])
        table_ori.append(row["table_ori"])

        if pd.notna(row["qa"]):
            question_0.append(row["qa"].get("question"))
            answer_0.append(row["qa"].get("answer"))
            question_1.append(None)
            answer_1.append(None)
        else:
            question_0.append(
                row["qa_0"].get("question") if pd.notna(row["qa_0"]) else None
            )
            answer_0.append(
                row["qa_0"].get("answer") if pd.notna(row["qa_0"]) else None
            )
            question_1.append(
                row["qa_1"].get("question") if pd.notna(row["qa_1"]) else None
            )
            answer_1.append(
                row["qa_1"].get("answer") if pd.notna(row["qa_1"]) else None
            )

    return pd.DataFrame(
        {
            "pre_text": pre_text,
            "post_text": post_text,
            "table_ori": table_ori,
            "question_0": question_0,
            "question_1": question_1,
            "answer_0": answer_0,
            "answer_1": answer_1,
        }
    )


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
