import os
import sys
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import logging


DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"
SRC_DIRECTORY = Path().cwd().resolve().parent
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

os.environ["HF_HOME"] = ""
HF_TOKEN = ""
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FinBench"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def huggify_finbench(push_to_hub=False, TASK=None):
    try:
        directory_path = DATA_DIRECTORY / "FinBench"
        logger.debug(f"Directory path: {directory_path}")

        dataset = load_dataset("yuweiyin/FinBench")

        train_set = dataset["train"] if "train" in dataset else []
        validation_set = dataset["validation"] if "validation" in dataset else []
        test_set = dataset["test"] if "test" in dataset else []
        
        hf_dataset = DatasetDict()
        
        #train split
        hf_dataset['train'] = train_set

        # Add test split
        hf_dataset['test'] = test_set
        
        # Add val split
        hf_dataset['validation'] = validation_set

        # Push to HF Hub
        if push_to_hub:
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name= "main",
                private=True,
                token=HF_TOKEN,
            )

        logger.info(f"Finished processing FinBench")
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing FinEntity dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    TASK = "FinBench"

    huggify_finbench(push_to_hub=True, TASK=TASK)
