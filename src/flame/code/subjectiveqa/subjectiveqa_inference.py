import traceback

import litellm
import pandas as pd
from tqdm import tqdm

from flame.code.prompts import PromptFormat, get_prompt
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.dataset_utils import safe_load_dataset
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "subjectiveqa")

litellm.drop_params = True


# litellm.set_verbose = True
def subjectiveqa_inference(args):
    definition_map = {
        "RELEVANT": "The speaker has answered the question entirely and appropriately.",
        "SPECIFIC": "The speaker includes specific and technical details in the answer.",
        "CAUTIOUS": "The speaker answers using a more conservative, risk-averse approach.",
        "ASSERTIVE": "The speaker answers with certainty about the company's events.",
        "CLEAR": "The speaker is transparent in the answer and about the message to be conveyed.",
        "OPTIMISTIC": "The speaker answers with a positive tone regarding outcomes.",
    }

    task = (
        getattr(args, "task", None) or getattr(args, "dataset", None) or "subjectiveqa"
    )
    logger.info(f"Starting inference for {task} using model {args.model}.")
    try:
        dataset = safe_load_dataset(
            "gtfintechlab/subjectiveqa", name="5768", trust_remote_code=True
        )
        dataset = dataset["test"]  # Get the test split
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        logger.error(traceback.format_exc())
        return None
    try:
        questions = [row["QUESTION"] for row in dataset]  # type: ignore
        answers = [row["ANSWER"] for row in dataset]  # type: ignore
        feature_labels = {
            feature: [row[feature] for row in dataset]
            for feature in definition_map.keys()
        }  # type: ignore
    except KeyError as e:
        logger.error(f"Missing expected columns in dataset: {e}")
        return None
    except Exception as e:
        logger.error(f"Error while extracting dataset fields: {e}")
        logger.error(traceback.format_exc())
        return None

    feature_responses = {feature: [] for feature in definition_map.keys()}

    if args.prompt_format == "fewshot":
        subjectiveqa_prompt = get_prompt("subjectiveqa", PromptFormat.FEW_SHOT)
    else:
        subjectiveqa_prompt = get_prompt("subjectiveqa", PromptFormat.ZERO_SHOT)
    if subjectiveqa_prompt is None:
        raise RuntimeError("SubjectiveQA prompt not found in registry")

    # Create batches for processing
    question_batches = chunk_list(questions, args.batch_size)
    answer_batches = chunk_list(answers, args.batch_size)
    total_batches = len(question_batches)
    logger.info(f"Processing {len(questions)} rows in {total_batches} batches.")

    pbar = tqdm(
        enumerate(zip(question_batches, answer_batches)),
        total=total_batches,
        desc="Processing SubjectiveQA entries",
    )
    for batch_idx, (question_batch, answer_batch) in pbar:
        messages_batch = []
        for q, a in zip(question_batch, answer_batch):
            for feature in definition_map.keys():
                messages_batch.append(
                    [
                        {
                            "role": "system",
                            "content": "You are an expert sentence classifier.",
                        },
                        {
                            "role": "user",
                            "content": subjectiveqa_prompt(
                                feature, definition_map[feature], q, a
                            ),
                        },
                    ]
                )
                # time.sleep(random.uniform(0.5, 1.5))  # Removed sleep for better performance
                pass
        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} processing failed: {e}")
            logger.error(traceback.format_exc())
            for feature in definition_map.keys():
                feature_responses[feature].extend(["error"] * len(question_batch))
            continue  # Move to next batch
        response_idx = 0
        for q, a in zip(question_batch, answer_batch):
            for feature in definition_map.keys():
                try:
                    response_label = (
                        batch_responses[response_idx].choices[0].message.content.strip()
                    )  # type: ignore
                    feature_responses[feature].append(response_label)
                except (KeyError, IndexError, AttributeError) as e:
                    logger.error(
                        f"Error extracting label for feature '{feature}' at batch {batch_idx + 1}: {e}"
                    )
                    logger.error(traceback.format_exc())
                    feature_responses[feature].append("error")
                response_idx += 1
    try:
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
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        logger.error(traceback.format_exc())
        return None
    try:
        # Calculate success metrics
        success_count = 0
        total_responses = 0
        for feature in definition_map.keys():
            success_count += sum(1 for r in feature_responses[feature] if r != "error")
            total_responses += len(feature_responses[feature])

        success_rate = (
            (success_count / total_responses) * 100 if total_responses > 0 else 0
        )
        logger.info(f"Inference completed. Success rate: {success_rate:.1f}%")
    except Exception as e:
        logger.error(f"Error calculating success metrics: {e}")
        logger.error(traceback.format_exc())
        return None

    return df
