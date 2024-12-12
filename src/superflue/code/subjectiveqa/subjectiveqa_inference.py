from datetime import date
import pandas as pd
from datasets import load_dataset

from superflue.code.prompts import subjectiveqa_prompt

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import LOG_LEVEL, LOG_DIR, RESULTS_DIR
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry

logger = setup_logger(
    name="subjectiveqa_inference",
    log_file=LOG_DIR / "subjectiveqa_inference.log",
    level=LOG_LEVEL,
)


def subjectiveqa_inference(args):
    definition_map = {
        "RELEVANT": "The speaker has answered the question entirely and appropriately.",
        "SPECIFIC": "The speaker includes specific and technical details in the answer.",
        "CAUTIOUS": "The speaker answers using a more conservative, risk-averse approach.",
        "ASSERTIVE": "The speaker answers with certainty about the company's events.",
        "CLEAR": "The speaker is transparent in the answer and about the message to be conveyed.",
        "OPTIMISTIC": "The speaker answers with a positive tone regarding outcomes.",
    }

    today = date.today()
    logger.info(f"Starting SubjectiveQA inference on {today}")

    # Load only the 'test' split of the dataset
    dataset = load_dataset(
        "gtfintechlab/subjectiveqa", "5768", split="test", trust_remote_code=True
    )

    # Initialize lists to store actual labels, model responses, and complete responses
    questions = [row["QUESTION"] for row in dataset]  # type: ignore
    answers = [row["ANSWER"] for row in dataset]  # type: ignore

    feature_labels = {
        feature: [row[feature] for row in dataset] for feature in definition_map.keys()
    }  # type: ignore
    feature_responses = {feature: [] for feature in definition_map.keys()}

    batch_size = 10
    total_batches = len(questions) // batch_size + int(len(questions) % batch_size > 0)
    logger.info(f"Processing {len(questions)} rows in {total_batches} batches.")

    # Create batches
    question_batches = chunk_list(questions, batch_size)
    answer_batches = chunk_list(answers, batch_size)
    # label_batches = {  # Unused variable
    #     feature: chunk_list(labels, batch_size)
    #     for feature, labels in feature_labels.items()
    # }

    for batch_idx, (question_batch, answer_batch) in enumerate(
        zip(question_batches, answer_batches)
    ):
        messages_batch = [
            [
                {"role": "system", "content": "You are an expert sentence classifier."},
                {
                    "role": "user",
                    "content": subjectiveqa_prompt(
                        feature, definition_map[feature], q, a
                    ),
                },
            ]
            for q, a in zip(question_batch, answer_batch)
            for feature in definition_map.keys()
        ]

        try:
            # Process batch with retry logic
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

            # Process responses for each feature in the batch
            response_idx = 0
            for q, a in zip(question_batch, answer_batch):
                for feature in definition_map.keys():
                    try:
                        response_label = (
                            batch_responses[response_idx]
                            .choices[0]
                            .message.content.strip()
                        )  # type: ignore
                        feature_responses[feature].append(response_label)
                        response_idx += 1
                    except (KeyError, IndexError, AttributeError) as e:
                        logger.error(
                            f"Error extracting label for feature '{feature}': {e}"
                        )
                        feature_responses[feature].append("error")

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            for feature in definition_map.keys():
                feature_responses[feature].extend(["error"] * len(question_batch))
            continue

    # Create a DataFrame to store the results
    df = pd.DataFrame(
        {
            "questions": questions,
            "answers": answers,
            **{
                f"{feature}_response": feature_responses[feature]
                for feature in definition_map.keys()
            },
            **{
                f"{feature}_actual_label": feature_labels[feature]
                for feature in definition_map.keys()
            },
        }
    )

    # Save results to a CSV file
    results_path = (
        RESULTS_DIR
        / "subjectiveqa"
        / f"subjectiveqa_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
