import os
import sys
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from datasets import Dataset, DatasetDict
import logging
import json
 
# Set up directory paths
SRC_DIRECTORY = Path().cwd().resolve().parent
DATA_DIRECTORY = Path().cwd().resolve().parent.parent / "data"
 
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))
 
# Environment variables for Hugging Face authentication
HF_TOKEN = os.getenv("HF_TOKEN")
HF_ORGANIZATION = "gtfintechlab"
DATASET = "REFinD"
 
# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
def read_json_file(file_path):
    """Read a JSON file and return the data."""
    with open(file_path, "r") as file:
        data = json.load(file)  # Load entire JSON file as a dictionary
    return data
 
def extract_data(data, unique_keys):
    """Extract specific keys from the data."""
    extracted_data = {key: [] for key in unique_keys}
    
    for item in data:
        for key in unique_keys:
            extracted_data[key].append(item.get(key, "N/A"))  # Default to "N/A" if key is missing
    
    return extracted_data
 
def huggify_data_refind(push_to_hub=False):
    """Process the REFinD dataset and prepare it for Hugging Face Hub."""
    try:
        directory_path = DATA_DIRECTORY / "REFinD"
        logger.debug(f"Directory path: {directory_path}")
 
        # Read JSON files
        train_data = read_json_file(f"{directory_path}/train.json")
        test_data = read_json_file(f"{directory_path}/test.json")
        validation_data = read_json_file(f"{directory_path}/validation.json")  
 
        # Define the unique keys
        unique_keys = [
            "e1_type", "e2", "e2_type", "id", "e1_end", "docid",
            "spacy_deprel", "spacy_ner", "sdp_tok_idx", "sdp",
            "rel_group", "e2_end", "token_test", "e2_start",
            "relation", "e1", "token", "spacy_head", "spacy_pos", "e1_start"
        ]
        
        # Extract data based on unique keys
        train_extracted = extract_data(train_data, unique_keys)
        test_extracted = extract_data(test_data, unique_keys)
        validation_extracted = extract_data(validation_data, unique_keys)
 
        # Create the DatasetDict
        splits = DatasetDict(
            {
                "train": Dataset.from_dict(train_extracted),
                "test": Dataset.from_dict(test_extracted),
                "validation": Dataset.from_dict(validation_extracted),
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
 
        logger.info("Finished processing REFinD dataset")
        return splits
 
    except Exception as e:
        logger.error(f"Error processing REFinD dataset: {str(e)}")
        raise e
 
if __name__ == "__main__":
    huggify_data_refind(push_to_hub=True)