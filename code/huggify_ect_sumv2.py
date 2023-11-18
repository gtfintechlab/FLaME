import pandas as pd
import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

import pandas as pd

from datasets import Dataset, DatasetDict

HF_ORGANIZATION = "gtfintechlab"
DATASET = "ectsum"


def huggify_data_ectsum(TASK=None, seed = None, ):
    try:
        directory_path = ""
        ect_sum_train = pd.read_csv(f"{directory_path}/train.csv")
        ect_sum_test = pd.read_csv(f"{directory_path}/test.csv")
        
        train_texts = ect_sum_train['input']
        train_labels = ect_sum_train['output']
        
        test_texts = ect_sum_test['input']
        test_labels = ect_sum_test['output']
        
        
        splits = {}

        splits = {
            "train": Dataset.from_dict(
                {
                    "context": train_texts,
                    "response": list(map(decode, train_labels)),
                }
            ),
            "test": Dataset.from_dict(
                {"context": test_texts, "response": list(map(decode, test_labels))}
            ),
            "validation": Dataset.from_dict(
                {"context": val_texts, "response": list(map(decode, val_labels))}
            ),
        }

        # Push to HF Hub
        splits["train"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-train-{seed}",
            config_name="train",
            private=True,
        )
        splits["test"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-test-{seed}",
            config_name="test",
            private=True,
        )
        splits["validation"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-validation-{seed}",
            config_name="validation",
            private=True,
        )

        return splits
    except Exception as e:
        print(f"Error processing ECT Sum dataset: {str(e)}")
