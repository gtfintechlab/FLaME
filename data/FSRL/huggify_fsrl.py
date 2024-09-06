import os
import sys
from pathlib import Path
from huggingface_hub import login
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
    """Ensure that an item is a list."""
    if item is None:
        return ["N/A"]
    if isinstance(item, list):
        return item
    return [item]

def log_structure(data, name):
    """Log the structure of the dataset and the lengths of lists."""
    logger.info(f"Logging structure of {name}")
    total_items = len(data)
    logger.info(f"Total items: {total_items}")

    for idx, item in enumerate(data):
        tokens = item.get("tokens", [])
        nodes = item.get("nodes", [])
        edges = item.get("edges", [])

        logger.debug(f"Item {idx} - tokens: {len(tokens)} | nodes: {len(nodes)} | edges: {len(edges)}")

        # Check for mixed types inside lists
        if any(not isinstance(t, (str, list)) for t in tokens):
            logger.error(f"Item {idx} - tokens contains mixed types: {tokens}")
        if any(not isinstance(n, (str, list)) for n in nodes):
            logger.error(f"Item {idx} - nodes contains mixed types: {nodes}")
        if any(not isinstance(e, (str, list)) for e in edges):
            logger.error(f"Item {idx} - edges contains mixed types: {edges}")

        # Log potential anomalies
        if len(tokens) == 0:
            logger.warning(f"Item {idx} - tokens list is empty.")
        if len(nodes) == 0:
            logger.warning(f"Item {idx} - nodes list is empty.")
        if len(edges) == 0:
            logger.warning(f"Item {idx} - edges list is empty.")

def transform_data_to_list(data):
    """Ensure all tokens, nodes, and edges are lists, logging their structures."""
    logger.info("Transforming data to ensure lists...")
    for idx, item in enumerate(data):
        if "tokens" in item:
            item["tokens"] = ensure_list(item["tokens"])
        if "nodes" in item:
            item["nodes"] = [ensure_list(node) for node in item["nodes"]] if "nodes" in item else []
        if "edges" in item:
            item["edges"] = [ensure_list(edge) for edge in item["edges"]] if "edges" in item else []
    return data

def verify_data(data):
    """Log errors for non-list fields and check for mixed types."""
    logger.info("Verifying data consistency...")
    for idx, item in enumerate(data):
        if not isinstance(item["tokens"], list):
            logger.error(f"Non-list item found in tokens at index {idx}: {item['tokens']}")
        if not isinstance(item["nodes"], list):
            logger.error(f"Non-list item found in nodes at index {idx}: {item['nodes']}")
        if not isinstance(item["edges"], list):
            logger.error(f"Non-list item found in edges at index {idx}: {item['edges']}")

def huggify_data_fsrl(push_to_hub=False):
    """Main function to process the FSRL dataset and optionally push it to the Hugging Face Hub."""
    try:
        directory_path = DATA_DIRECTORY / "FSRL"
        logger.debug(f"Directory path: {directory_path}")

        def read_json_file(file_path):
            """Read a JSON file and log info about its contents."""
            with open(file_path, "r") as file:
                data = json.load(file)
                logger.info(f"Read {len(data)} items from {file_path}")
                logger.info(f"Data type: {type(data)}")
            return data

        # Read the train, test, and validation datasets
        train_data = read_json_file(f"{directory_path}/converted_train.json")
        test_data = read_json_file(f"{directory_path}/converted_test.json")
        val_data = read_json_file(f"{directory_path}/converted_validation.json")

        # Log the structure of the datasets before transformation
        log_structure(train_data, "train_data")
        log_structure(test_data, "test_data")
        log_structure(val_data, "val_data")

        # Transform the data to ensure tokens, nodes, and edges are lists
        train_data = transform_data_to_list(train_data)
        test_data = transform_data_to_list(test_data)
        val_data = transform_data_to_list(val_data)

        # Verify that all fields are now lists
        verify_data(train_data)
        verify_data(test_data)
        verify_data(val_data)

        # Log the structure of the datasets after transformation
        log_structure(train_data, "train_data (after transformation)")
        log_structure(test_data, "test_data (after transformation)")
        log_structure(val_data, "val_data (after transformation)")

        def extract_data(data):
            """Extract tokens, nodes, and edges from the data."""
            tokens, nodes, edges = [], [], []
            for idx, item in enumerate(data):
                try:
                    tokens.append(item["tokens"])
                    nodes.append(item["nodes"])
                    edges.append(item["edges"])
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

        # Optionally push to Hugging Face Hub
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
