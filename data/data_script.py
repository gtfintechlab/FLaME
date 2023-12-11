from huggingface_hub import hf_hub_download
import pandas as pd

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
DATASET = "ECTSum"

def encode(self, label_name):
    reversed_mapping = {v: k for k, v in self.mappings[self.task].items()}
    return reversed_mapping.get(label_name, -1)

def decode(self, label_number):
    return self.mappings[self.task].get(label_number, "undefined").upper()


def huggify_data_ectsum():
    try:
        directory_path = ""
        ect_sum_train = pd.read_csv(f"{directory_path}/train.csv")
        ect_sum_test = pd.read_csv(f"{directory_path}/test.csv")
        ect_sum_val = pd.read_csv(f"{directory_path}/val.csv")
        
        train_texts = ect_sum_train['input']
        train_labels = ect_sum_train['output']
        
        test_texts = ect_sum_test['input']
        test_labels = ect_sum_test['output']
        
        val_texts = ect_sum_val['input']
        val_labels = ect_sum_val['output']
        
        
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
        
        return splits
    except Exception as e:
        print(f"Error processing ECT Sum dataset: {str(e)}")

huggify_data_ectsum()



REPO_ID = ""

FILENAMES = ['train.csv', ' test.csv', 'val.csv']

for FILENAME in FILENAMES:
    dataset = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    )



