import sys
import zipfile
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from superflue.utils.logging_utils import setup_logger
from superflue.config import SEEDS

logger = setup_logger(__name__)

from superflue.utils.label_utils import encode


def get_FPB_dataset():
    zip_path = DATA_DIR / "FinancialPhraseBank-v1.0.zip"
    logger.info(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("ExtractedFinancialPhraseBank")


def process_data():
    FPB_DIRECTORY = DATA_DIR / "ExtractedFinancialPhraseBank"
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
    TRAIN_DIRECTORY = DATA_DIR / "sentiment_analysis" / "train"
    TRAIN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TEST_DIRECTORY = DATA_DIR / "sentiment_analysis" / "test"
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
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

    DATA_DIR = Path(config["fpb"]["DATA_DIR"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Building the FinancialPhraseBank dataset in data directory {DATA_DIR}"
    )
    get_FPB_dataset()
    process_data()
