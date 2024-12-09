import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login

from ferrari.config import DATA_DIR, LOG_LEVEL

HF_ORGANIZATION = "gtfintechlab"
DATASET = "FinCausal2020"

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def huggify_fincausal(
    push_to_hub=False,
    TASK=None,
):
    load_dotenv()
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    login(HUGGINGFACEHUB_API_TOKEN)
    try:
        base_path = TASK + "/fnp2020-fincausal-"
        total_path = str(DATA_DIR) + "/FinCausal/" + base_path
        dataset_dict = {}
        for suffix in ["practice", "trial", "evaluation"]:
            df = pd.read_csv(total_path + suffix + ".csv", sep=";", index_col="Index")
            df.columns = df.columns.str.replace(" ", "")
            data_dict = {}
            for col in df.columns:
                data_dict[col] = list(df[col])
            if suffix == "evaluation":
                if TASK == "Task2":
                    for col in [
                        "Cause",
                        "Effect",
                        "Offset_Sentence2",
                        "Offset_Sentence3",
                        "Sentence",
                    ]:
                        data_dict[col] = [""] * len(df)
                    for col in [
                        "Cause_Start",
                        "Cause_End",
                        "Effect_Start",
                        "Effect_End",
                    ]:
                        data_dict[col] = [0] * len(df)
                else:
                    data_dict["Gold"] = [0] * len(df)
            dataset_dict[suffix] = Dataset.from_dict(data_dict)

        hf_dataset = DatasetDict(dataset_dict)

        # Push to HF Hub
        path = DATASET + "_" + TASK
        if push_to_hub:
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{path}",
                config_name="main",
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )
        logger.info("Finished processing FinCausal " + TASK)
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing FinCausal dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    for task in ["Task1", "Task2"]:
        huggify_fincausal(push_to_hub=True, TASK=task)
