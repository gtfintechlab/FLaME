import os
import sys
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging
import json

SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

HF_TOKEN = os.environ["HF_TOKEN"]
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FSRL"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_list(item):
    if item is None:
        return ["N/A"]
    if isinstance(item, list):
        return item
    return [item]

def sanitize_data(data):
    sanitized_data = []
    for idx, item in enumerate(data):
        sanitized_item = {
            "tokens": ensure_list(item.get("tokens")),
            "nodes": ensure_list(item.get("nodes")),
            "edges": ensure_list(item.get("edges"))
        }
        sanitized_data.append(sanitized_item)
    return sanitized_data

def transform_tokens_to_list(data):
    for item in data:
        if 'tokens' in item and isinstance(item['tokens'], str):
            item['tokens'] = [item['tokens']]
    return data

def huggify_data_fsrl(push_to_hub=False):
    try:
        directory_path = DATA_DIRECTORY / "FSRL"
        logger.debug(f"Directory path: {directory_path}")

        def read_json_file(file_path):
            with open(file_path, 'r') as file:
                data = [json.loads(line) for line in file]
            return data
        
        train_data = read_json_file(f'{directory_path}/train.json')
        test_data = read_json_file(f'{directory_path}/test.json')
        val_data = read_json_file(f'{directory_path}/validation.json')

        # Transform tokens to lists
        train_data = transform_tokens_to_list(train_data)
        test_data = transform_tokens_to_list(test_data)
        val_data = transform_tokens_to_list(val_data)

        # Sanitize data
        train_data = sanitize_data(train_data)
        test_data = sanitize_data(test_data)
        val_data = sanitize_data(val_data)

        def extract_data(data):
            tokens = []
            nodes = []
            edges = []
            for idx, item in enumerate(data):
                try:
                    tokens.append(item['tokens'])
                    nodes.append(item['nodes'])
                    edges.append(item['edges'])
                except Exception as e:
                    logger.error(f"Error processing item at index {idx}: {str(e)}")
                    logger.error(f"Problematic item: {json.dumps(item, indent=2)}")
                    raise e
            return tokens, nodes, edges

        # Extract tokens, nodes, and edges for train, test, and validation sets
        train_tokens, train_nodes, train_edges = extract_data(train_data)
        test_tokens, test_nodes, test_edges = extract_data(test_data)
        val_tokens, val_nodes, val_edges = extract_data(val_data)

        # Create the DatasetDict
        splits = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "tokens": train_tokens,
                        "nodes": train_nodes,
                        "edges": train_edges,
                    }
                ),
                "test": Dataset.from_dict(
                    {
                        "tokens": test_tokens,
                        "nodes": test_nodes,
                        "edges": test_edges,
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "tokens": val_tokens,
                        "nodes": val_nodes,
                        "edges": val_edges,
                    }
                ),
            }
        )

        # Push to HF Hub
        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HF_TOKEN,
            )

        logger.info("Finished processing FSLR dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing FSLR dataset: {str(e)}")
        raise e

if __name__ == "__main__":
    huggify_data_fsrl(push_to_hub=True)
