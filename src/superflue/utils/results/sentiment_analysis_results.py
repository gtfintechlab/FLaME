import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from superflue.config import LOG_LEVEL, RESULTS_DIR, LOG_DIR
from superflue.utils.logging_utils import setup_logger

import nltk

from superflue.utils.results.decode import sentiment_analysis_decode

nltk.download("punkt")

logger = setup_logger(
    name=__name__, log_file=LOG_DIR / "sentiment_analysis_results.log", level=LOG_LEVEL
)


def compute_metrics(files, outputs_directory):
    acc_list = []
    f1_list = []
    missing_perc_list = []

    for file in files:
        df = pd.read_csv(outputs_directory / file)

        # Decode the predicted label
        df["predicted_label"] = df["text_output"].apply(sentiment_analysis_decode)

        # Calculate metrics
        acc_list.append(accuracy_score(df["true_label"], df["predicted_label"]))
        f1_list.append(
            f1_score(df["true_label"], df["predicted_label"], average="weighted")
        )
        missing_perc_list.append(
            (len(df[df["predicted_label"] == -1]) / df.shape[0]) * 100.0
        )

    return acc_list, f1_list, missing_perc_list


def main(args):
    LLM_OUTPUTS_DIRECTORY = (
        RESULTS_DIR / args.task_name / "llm_prompt_outputs" / args.quantization
    )

    # Filter out relevant files
    files = [
        f
        for f in LLM_OUTPUTS_DIRECTORY.iterdir()
        if args.model_id in f.name and f.suffix == ".csv"
    ]

    acc_list, f1_list, missing_perc_list = compute_metrics(files, LLM_OUTPUTS_DIRECTORY)

    # Print results
    print("f1 score mean: ", format(np.mean(f1_list), ".4f"))
    print("f1 score std: ", format(np.std(f1_list), ".4f"))
    print(
        "Percentage of cases when didn't follow instruction: ",
        format(np.mean(missing_perc_list), ".4f"),
        "\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics for sentiment analysis results."
    )
    parser.add_argument(
        "-m", "--model_id", type=str, required=True, help="Name of the model used."
    )
    parser.add_argument(
        "-q",
        "--quantization",
        type=str,
        required=True,
        help="Quantization level of the model.",
    )
    parser.add_argument(
        "-t", "--task_name", type=str, required=True, help="Name of the task."
    )
    args = parser.parse_args()

    main(args)
