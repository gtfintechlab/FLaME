import re
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from flame.utils.logging_utils import setup_logger
from flame.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from litellm.types.utils import (
    ModelResponse,
    Choices,
    Message,
    Usage,
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
)
from datetime import date


def parse_label_list(raw_string: str):
    """
    Safely parse a string that may contain bracketed label lists with extra commas,
    single quotes, or additional text. Returns a clean list of tags or an empty list.
    """
    if not isinstance(raw_string, str):
        return []

    start = raw_string.find("[")
    end = raw_string.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    bracketed = raw_string[start : end + 1]

    bracketed = bracketed.replace("'", '"')

    bracketed = re.sub(r'(".*?"),\s*', r"\1,", bracketed)  # handle embedded strings
    bracketed = re.sub(r",\s*\]", "]", bracketed)  # remove trailing commas before the ]

    try:
        parsed = json.loads(bracketed)
        cleaned = [
            re.sub(r",$", "", item.strip()) for item in parsed if isinstance(item, str)
        ]
        return cleaned
    except Exception as e:
        print(f"Exception parsing label list: {e}")
        print(f"Error parsing label list: {raw_string}")
        return []


def adjust_tags(row):
    actual = row["actual_tags"]
    predicted = row["predicted_tags"]
    if len(predicted) > len(actual):
        return predicted[: len(actual)]
    elif len(predicted) < len(actual):
        return None
    else:
        return predicted


def casual_detection_evaluate(file_name, args):
    logger = setup_logger(
        name="causal_detection_evaluation",
        log_file=LOG_DIR / "causal_detection_evaluation.log",
        level=LOG_LEVEL,
    )
    task = args.dataset.strip('“”"')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    type_dict = {
        "ModelResponse": ModelResponse,
        "Choices": Choices,
        "Message": Message,
        "Usage": Usage,
        "CompletionTokensDetailsWrapper": CompletionTokensDetailsWrapper,
        "PromptTokensDetailsWrapper": PromptTokensDetailsWrapper,
    }
    df["complete_responses"] = df["complete_responses"].apply(
        lambda x: eval(x, type_dict)
    )
    df["llm_responses"] = df["complete_responses"].apply(
        lambda x: x.choices[0].message.content
    )
    df["llm_responses"] = df["llm_responses"].apply(
        lambda x: x[(x.find("</think>") + 8) :]
    )

    df["predicted_tags"] = df["llm_responses"].apply(lambda x: x.strip())

    df["actual_tags"] = df["actual_tags"].apply(parse_label_list)
    df["predicted_tags"] = df["predicted_tags"].apply(parse_label_list)

    print(df["predicted_tags"][0])

    df["adjusted_predicted_tags"] = df.apply(adjust_tags, axis=1)  # type: ignore

    df["length_match"] = df["adjusted_predicted_tags"].notnull()

    df["row_accuracy"] = df.apply(
        lambda row: (
            accuracy_score(row["actual_tags"], row["adjusted_predicted_tags"])
            if row["length_match"]
            else 0.0
        ),  # type: ignore
        axis=1,
    )  # type: ignore

    valid_rows = df[df["length_match"]]

    flat_actual = [tag for tags in valid_rows["actual_tags"] for tag in tags]
    flat_predicted = [
        tag for tags in valid_rows["adjusted_predicted_tags"] for tag in tags
    ]

    labels = ["B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT", "O"]
    print("Token Classification Report:")
    print(classification_report(flat_actual, flat_predicted, labels=labels))

    accuracy = accuracy_score(flat_actual, flat_predicted)
    print(f"Overall Token-Level Accuracy: {accuracy:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_actual, flat_predicted, average="weighted"
    )

    logger.info(
        f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}"
    )
    df.to_csv(evaluation_results_path, index=False)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
        }
    )

    # Save metrics DataFrame
    metrics_path = evaluation_results_path.with_name(
        f"{evaluation_results_path.stem}_metrics.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
