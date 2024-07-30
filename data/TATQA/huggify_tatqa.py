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


HF_TOKEN = os.environ["HF_TOKEN"]
HF_ORGANIZATION = "gtfintechlab"
DATASET = "TATQA"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def huggify_TATQA(push_to_hub=False, TASK=None):
    try:
        directory_path = DATA_DIRECTORY / "TATQA"
        logger.debug(f"Directory path: {directory_path}")

        dataset = load_dataset("TheFinAI/flare-tatqa")

        test_set = dataset["test"] if "test" in dataset else []
        
        hf_dataset = DatasetDict()
    

        hf_dataset['test'] = test_set
        
        if push_to_hub:
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name= "main",
                private=True,
                token=HF_TOKEN,
            )

        logger.info(f"Finished processing TATQA")
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing TATQA dataset: {str(e)}")
        raise e

if __name__ == "__main__":
    TASK = "TATQA"

    huggify_TATQA(push_to_hub=True, TASK=TASK)