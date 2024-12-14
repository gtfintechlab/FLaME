import os
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
from superflue import DATA_DIR
from superflue.utils.logging_utils import get_logger

logger = get_logger(__name__)

# TODO: Cleanup and remove this code below get it from dotenv etc
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_ORGANIZATION = "gtfintechlab"
DATASET = "CausalClassification"
login(HUGGINGFACEHUB_API_TOKEN)


def huggify_data_sc(push_to_hub=False):
    try:
        directory_path = DATA_DIR / "CausalClassification"
        logger.debug(f"Directory path: {directory_path}")

        sc_train = pd.read_json(f"{directory_path}/train.json")
        sc_test = pd.read_json(f"{directory_path}/test.json")
        sc_val = pd.read_json(f"{directory_path}/validation.json")

        train_input = sc_train["text"]
        train_output = sc_train["label"]

        test_input = sc_test["text"]
        test_output = sc_test["label"]

        val_input = sc_val["text"]
        val_output = sc_val["label"]

        splits = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "text": train_input,
                        "label": train_output,
                    }
                ),
                "test": Dataset.from_dict({"text": test_input, "label": test_output}),
                "validation": Dataset.from_dict(
                    {"text": val_input, "label": val_output}
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

        logger.info("Finished processing Causal Classification dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing Causal Classification dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_sc(push_to_hub=True)
