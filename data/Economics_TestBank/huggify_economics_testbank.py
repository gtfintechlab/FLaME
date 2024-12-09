import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values
from ferrari.config import DATA_DIR, LOG_LEVEL
from huggingface_hub import login

HF_ORGANIZATION = "glennmatlin"
DATASET = "Economics_TestBank"

# Configure logging to show on console with timestamp and level
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def huggify_economics_testbank(push_to_hub: bool = False) -> DatasetDict:
    """
    Convert Economics TestBank CSV to HuggingFace dataset format.
    
    Args:
        push_to_hub: Whether to push the dataset to HuggingFace Hub
        
    Returns:
        DatasetDict containing the processed dataset
    
    Raises:
        ValueError: If HuggingFace token is not found
    """
    logger.info("Starting Economics TestBank processing")
    
    # Load and validate environment
    logger.info("Loading environment configuration")
    config = dotenv_values(".env")
    token = config.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        logger.error("HUGGINGFACEHUB_API_TOKEN not found in .env file")
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

    # Login to Hugging Face
    logger.info("Logging into Hugging Face Hub")
    try:
        login(token)
        logger.info("Successfully logged into Hugging Face Hub")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face Hub: {str(e)}")
        raise

    # Load and process CSV
    csv_path = Path(DATA_DIR) / "Economics_TestBank" / "Economics_TestBank.csv"
    logger.info(f"Loading CSV from: {csv_path}")
    
    if not csv_path.exists():
        logger.error(f"CSV file not found at: {csv_path}")
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    
    df = pd.read_csv(csv_path, index_col=False)
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Select and convert relevant columns
    columns = ["book_name", "chapter_number", "chapter_name", "prompt", "answer"]
    df = df[columns]
    logger.info("Selected relevant columns")

    # Convert DataFrame to dataset format
    logger.info("Converting DataFrame to HuggingFace dataset format")
    data_dict = {col: df[col].astype(str).tolist() for col in df.columns}
    hf_dataset = DatasetDict({"train": Dataset.from_dict(data_dict)})
    logger.info("Successfully created HuggingFace dataset")

    if push_to_hub:
        repo_id = f"{HF_ORGANIZATION}/{DATASET}"
        logger.info(f"Pushing dataset to HuggingFace Hub at {repo_id}")
        try:
            hf_dataset.push_to_hub(
                repo_id,
                config_name="main",
                private=True,
            )
            logger.info("Successfully pushed Economics TestBank to HuggingFace Hub")
        except Exception as e:
            logger.error(f"Failed to push to HuggingFace Hub: {str(e)}")
            raise

    return hf_dataset


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Process Economics TestBank data")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to HuggingFace Hub",
    )
    args = parser.parse_args()
    
    try:
        huggify_economics_testbank(push_to_hub=args.push_to_hub)
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
