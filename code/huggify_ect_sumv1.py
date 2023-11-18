import os
import sys
from pathlib import Path

from tqdm.notebook import tqdm

SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

import pandas as pd

from datasets import Dataset, DatasetDict

HF_ORGANIZATION = "gtfintechlab"
def huggify_data_ect_sum(TASK=None, SEED=None, SPLITS=['train', 'test','val']):
    try:
        #mapper = LabelMapper(TASK)
        ect_dataset = DatasetDict()
        data_path = ""
        # Load data
        for SPLIT in SPLITS:
            
            data_split = pd.read_csv( f"{data_path}_{SPLIT}_{SEED}.csv")
            data_split.rename(columns={'output': 'label_encoded'}, inplace=True) 
            #data_split['label_decoded'] = data_split['label_encoded'].apply(lambda x: mapper.decode(x))

            
            ect_dataset[SPLIT] = Dataset.from_pandas(data_split)

        ect_dataset.push_to_hub(
            f"{HF_ORGANIZATION}/{TASK}",
            config_name=str(SEED),
            private=True,
        )

    except Exception as e:
        print(e)

huggify_data_ect_sum(TASK='ect_sum', SEED=42)