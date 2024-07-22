import os
import sys
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging
import json

# TODO: check if this is the right way to import from the src folder
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

        def ensure_list(item):
            if not isinstance(item, list):
                return [item]
            return item

        def extract_data(data):
            tokens = []
            nodes = []
            edges = []
            for item in data:
                if item:
                    if 'tokens' in item and 'nodes' in item and 'edges' in item:
                        tokens.append(ensure_list(item['tokens']))
                        nodes.append(ensure_list(item['nodes']))
                        edges.append(ensure_list(item['edges']))
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


