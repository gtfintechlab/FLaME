from datetime import date
import pandas as pd
from datasets import load_dataset
from superflue.code.prompts import subjectiveqa_prompt
from superflue.utils.logging_utils import get_logger
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.utils.save_utils import save_inference_results

logger = get_logger(__name__)


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

    # Extract provider and model info for metadata
    model_parts = args.model.split("/")
    provider = model_parts[0] if len(model_parts) > 1 else "unknown"
    model_name = model_parts[-1]

    # Save results with metadata
    metadata = {
        "model": args.model,
        "provider": provider,
        "model_name": model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
        "success_rate": (
            df[[f"{feature}_response" for feature in definition_map.keys()]]
            .notna()
            .sum()
            .sum()
            / (len(df) * len(definition_map))
        )
        * 100,
    }

    # Use our save utility
    save_inference_results(
        df=df, task="subjectiveqa", model=args.model, metadata=metadata
    )

    return df
