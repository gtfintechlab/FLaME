import os
from huggingface_hub import login
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from superflue import DATA_DIR

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
HF_ORGANIZATION = "gtfintechlab"
DATASET = "SubjECTiveQA"
login(HUGGINGFACEHUB_API_TOKEN)

from superflue.utils.logging_utils import get_logger

logger = get_logger(__name__)


def huggify_data_subjectiveqa(
    push_to_hub=False, TASK=None, SEED=None, SPLITS=["train", "test", "val"]
):
    try:
        directory_path = DATA_DIR / "SubjECTiveQA"
        logger.debug(f"Directory path: {directory_path}")

        hf_dataset = DatasetDict()

        df = pd.read_csv(f"{directory_path}/final_dataset.csv")

        input_columns = [
            "COMPANYNAME",
            "QUARTER",
            "YEAR",
            "ASKER",
            "RESPONDER",
            "QUESTION",
            "ANSWER",
        ]
        output_columns = [
            "CLEAR",
            "ASSERTIVE",
            "CAUTIOUS",
            "OPTIMISTIC",
            "SPECIFIC",
            "RELEVANT",
        ]

        train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=SEED)
        test_df, val_df = train_test_split(
            test_val_df, test_size=0.3, random_state=SEED
        )

        hf_dataset["train"] = Dataset.from_pandas(
            train_df[input_columns + output_columns]
        )

        # Add test split
        hf_dataset["test"] = Dataset.from_pandas(
            test_df[input_columns + output_columns]
        )

        hf_dataset["val"] = Dataset.from_pandas(val_df[input_columns + output_columns])

        # Push to HF Hub
        if push_to_hub:
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name=str(SEED),
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

            # TODO: push the dataset dict object not the datasets individually

        logger.info("Finished processing SubjECTive dataset seed : {SEED}")
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing SubjECTive dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    SPLITS = ["train", "test", "val"]

    TASK = "SubjECTive-QA"

    SEEDS = (5768, 78516, 944601)

    for SEED in list(reversed(SEEDS)):
        huggify_data_subjectiveqa(push_to_hub=True, TASK=TASK, SEED=SEED, SPLITS=SPLITS)
