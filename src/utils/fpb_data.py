# TODO: Move over to using the HF dataset https://huggingface.co/datasets/financial_phrasebank

import sys
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.utils.logging import setup_logger
from utils.config import SEEDS

logger = setup_logger(__name__)


from src.utils.label_utils import encode


# def get_FPB_dataset_from_HF():
#     logger.info(f"Getting FPB from HuggingFace datasets")
#     # Load the dataset
#     dataset = datasets.load_dataset("financial_phrasebank")
#     # Path to save the dataset as a zip file
#     zip_path = DATA_DIRECTORY / "FinancialPhraseBank-v1.0.zip"
#     # Save the dataset to a zip file
#     logger.info(f"Saving FPB to {zip_path}")
#     dataset.save_to_disk(zip_path)
#     # Unzip the file
#     logger.info(f"Unzipping {zip_path}")
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall("ExtractedFinancialPhraseBank")


def get_FPB_dataset():
    zip_path = DATA_DIRECTORY / "FinancialPhraseBank-v1.0.zip"
    logger.info(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("ExtractedFinancialPhraseBank")


def process_data():
    FPB_DIRECTORY = DATA_DIRECTORY / "ExtractedFinancialPhraseBank"
    FPB_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing the FPB data into {FPB_DIRECTORY}")
    df = pd.read_csv(
        FPB_DIRECTORY / "Sentences_AllAgree.txt",
        sep="@",
        encoding="latin1",
        names=["sentence", "label"],
    )
    df["label"] = df["label"].apply(lambda x: encode(x))

    # TODO: have it build to the directories when reused i.e. numclaim_detection
    TRAIN_DIRECTORY = DATA_DIRECTORY / "sentiment_analysis" / "train"
    TRAIN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TEST_DIRECTORY = DATA_DIRECTORY / "sentiment_analysis" / "test"
    TEST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for seed in tqdm(SEEDS):
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
        train_df.to_excel(
            TRAIN_DIRECTORY / f"FPB-sentiment-analysis-allagree-train-{seed}.xlsx",
            index=False,
        )
        test_df.to_excel(
            TEST_DIRECTORY / f"FPB-sentiment-analysis-allagree-test-{seed}.xlsx",
            index=False,
        )


if __name__ == "__main__":
    ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
    if str(ROOT_DIRECTORY) not in sys.path:
        sys.path.insert(0, str(ROOT_DIRECTORY))
with open("src/utils/config.yaml", "r") as file:
    config = yaml.safe_load(file)

DATA_DIRECTORY = Path(config["fpb"]["data_directory"])
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Building the FinancialPhraseBank dataset in data directory {DATA_DIRECTORY}"
    )
    get_FPB_dataset()
    process_data()
