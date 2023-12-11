import os
import sys
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download

from tqdm.notebook import tqdm

SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

import pandas as pd

from datasets import Dataset, DatasetDict
from huggingface_hub import notebook_login

HF_ORGANIZATION = "gtfintechlab"

# ## Convfinqa
import requests
import zipfile
import io
import pandas as pd
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import notebook_login

def download_zip_content(url):
    response = requests.get(url)
    return zipfile.ZipFile(io.BytesIO(response.content))

def process_qa_pairs(data):

    pre_text, post_text, table_ori = [], [], []
    question_0, question_1, answer_0, answer_1 = [], [], [], []

    for _, row in data.iterrows():
        pre_text.append(row['pre_text'])
        post_text.append(row['post_text'])
        table_ori.append(row['table_ori'])

        if pd.notna(row['qa']):
            question_0.append(row['qa'].get('question'))
            answer_0.append(row['qa'].get('answer'))
            question_1.append(None)
            answer_1.append(None)
        else:
            question_0.append(row['qa_0'].get('question') if pd.notna(row['qa_0']) else None)
            answer_0.append(row['qa_0'].get('answer') if pd.notna(row['qa_0']) else None)
            question_1.append(row['qa_1'].get('question') if pd.notna(row['qa_1']) else None)
            answer_1.append(row['qa_1'].get('answer') if pd.notna(row['qa_1']) else None)

    return pd.DataFrame({
        'pre_text': pre_text,
        'post_text': post_text,
        'table_ori': table_ori,
        'question_0': question_0,
        'question_1': question_1,
        'answer_0': answer_0,
        'answer_1': answer_1
    })

def huggify_data_convfinqa(task_name="convfinqa", namespace="Yangvivian"):
    try:
    
        notebook_login()

        hf_dataset = DatasetDict()

        zip_url = 'https://raw.githubusercontent.com/czyssrs/ConvFinQA/main/data.zip'

        with download_zip_content(zip_url) as zip_file:
            for SPLIT in ['train', 'dev']:
                with zip_file.open(f'data/{SPLIT}.json') as file:
                    json_str = file.read()
                    json_data = json.loads(json_str.decode('utf-8'))

                    data_split = pd.DataFrame(json_data)
                    qa_pairs_df = process_qa_pairs(data_split)
                    hf_dataset[SPLIT] = Dataset.from_pandas(qa_pairs_df)

        hf_dataset.push_to_hub(
            f"{namespace}/{task_name}",
            private=True
        )

    except Exception as e:
        print("An error occurred:", e)
        import traceback
        traceback.print_exc()
        return None

    return hf_dataset

# Example usage
namespace = "Yangvivian"  # change to gtfintechlab if necessary
task_name = "convfinqa"
hf_dataset = huggify_data_convfinqa(task_name=task_name, namespace=namespace)

# ## Finqa


def process_qa_pairs(data):
    pre_text, post_text, table_ori = [], [], []
    questions, answers = [], []

    for _, row in data.iterrows():
        pre_text.append(row['pre_text'])
        post_text.append(row['post_text'])
        table_ori.append(row['table_ori'])

        if pd.notna(row['qa']):
            questions.append(row['qa'].get('question'))
            answers.append(row['qa'].get('answer'))
        else:
            
            questions.append(None)
            answers.append(None)

    return pd.DataFrame({
        'pre_text': pre_text,
        'post_text': post_text,
        'table_ori': table_ori,
        'question': questions,
        'answer': answers
    })
def huggify_data_finqa(task_name="finqa", namespace="Yangvivian"):
    try:
        notebook_login()

        hf_dataset = DatasetDict()

        base_url = 'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset'
        splits = ['train', 'test', 'dev']

        for split in splits:
            url = f'{base_url}/{split}.json'
            response = requests.get(url)
            json_data = response.json()

            data_split = pd.DataFrame(json_data)
            qa_pairs_df = process_qa_pairs(data_split)
            hf_dataset[split] = Dataset.from_pandas(qa_pairs_df)

        hf_dataset.push_to_hub(
            f"{namespace}/{task_name}",
            private=True
        )

    except Exception as e:
        print("An error occurred:", e)
        import traceback
        traceback.print_exc()
        return None

    return hf_dataset

namespace = "gtfintechlab"  # gtfintechlab
task_name = "finqa"
hf_dataset = huggify_data_finqa(task_name=task_name, namespace=namespace)

# Optional print statements
if hf_dataset is not None:
    print("Dataset created successfully.")
    print("Sample data from train set:")
    print(hf_dataset['train'][0:1])
else:
    print("Failed to create datasets.")
    
    
    
REPO_ID = ""

FILENAMES = ['train.csv', ' test.csv', 'val.csv']

for FILENAME in FILENAMES:
    dataset = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    )