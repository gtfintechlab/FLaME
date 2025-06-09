import re
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_DIR, LOG_LEVEL
from litellm.types.utils import (
    ModelResponse,
    Choices,
    Message,
    Usage,
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
)

# Setup logger at module level
logger = setup_logger(
    name="causal_detection_evaluation",
    log_file=LOG_DIR / "causal_detection_evaluation.log",
    level=LOG_LEVEL,
)


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
        logger.error(f"Exception parsing label list: {e}")
        logger.error(f"Error parsing label list: {raw_string}")
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


def causal_detection_evaluate_direct(file_name, args):
    # Logger already setup at module level
    # support legacy args.dataset for tests, prefer args.task
    task = (
        getattr(args, "task", None)
        or getattr(args, "dataset", None)
        or "causal_detection"
    )
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Note: Path definition removed - evaluate.py handles saving

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

    logger.debug(f"First predicted tags: {df['predicted_tags'][0]}")

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
    logger.info("Token Classification Report:")
    logger.info(classification_report(flat_actual, flat_predicted, labels=labels))

    accuracy = accuracy_score(flat_actual, flat_predicted)
    logger.info(f"Overall Token-Level Accuracy: {accuracy:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_actual, flat_predicted, average="weighted"
    )

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}.")
    # Note: File saving removed - evaluate.py handles saving

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

    # Note: Metrics saving removed - evaluate.py handles saving

    return df, metrics_df
