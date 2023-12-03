import os
import sys
from pathlib import Path

from tqdm.notebook import tqdm

SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"

class LabelMapper:
    def __init__(self, task):
        self.mappings = {
            'finer_ord': {
                0: "other",
                1: "person_b",
                2: "person_i",
                3: "location_b",
                4: "location_i",
                5: "organisation_b",
                6: "organisation_i"
            },
            'fomc_communication': {
                0: "dovish",
                1: "hawkish",
                2: "neutral"
            },
            'numclaim_detection': {
                0: "outofclaim",
                1: "inclaim"
            },
            'sentiment_analysis': {
                0: "positive",
                1: "negative",
                2: "neutral"
            }
        }
        if task not in self.mappings:
            raise ValueError(f"Task {task} not found in mappings.")
        self.task = task
        
    def encode(self, label_name):
        reversed_mapping = {v: k for k, v in self.mappings[self.task].items()}
        return reversed_mapping.get(label_name, -1)
    
    def decode(self, label_number):
        return self.mappings[self.task].get(label_number, "undefined").upper()

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

import pandas as pd

from datasets import Dataset, DatasetDict

HF_ORGANIZATION = "gtfintechlab"
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
DATASET = "Numclaim"

def encode(self, label_name):
    reversed_mapping = {v: k for k, v in self.mappings[self.task].items()}
    return reversed_mapping.get(label_name, -1)

def decode(self, label_number):
    return self.mappings[self.task].get(label_number, "undefined").upper()


def huggify_numclaim():
    try:
        directory_path = ""
        numclaim_train = pd.read_excel(f"{directory_path}/numclaim-train-5768.xlsx")
        numclaim_test = pd.read_excel(f"{directory_path}/numclaim-test-5768.xlsx")
        
        train_texts = numclaim_train['text']
        train_labels = numclaim_train['label']
        
        test_texts = numclaim_test['text']
        test_labels = numclaim_test['label']
        
        
        
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
        }

        # Push to HF Hub
        '''splits["train"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-train-{seed}",
            config_name="train",
            private=True,
        )
        splits["test"].push_to_hub(
            f"{HF_ORGANIZATION}/{DATASET}-test-{seed}",
            config_name="test",
            private=True,
        )'''
        
        return splits
    except Exception as e:
        print(f"Error processing Numclaim dataset: {str(e)}")
