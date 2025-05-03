import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from flame.code.prompts_zeroshot import subjectiveqa_zeroshot_prompt
from flame.code.prompts_fewshot import subjectiveqa_fewshot_prompt
from flame.utils.logging_utils import setup_logger
from flame.config import LOG_LEVEL, LOG_DIR, RESULTS_DIR
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
import random
import traceback
import litellm

logger = setup_logger(
    name="subjectiveqa_inference",
    log_file=LOG_DIR / "subjectiveqa_inference.log",
    level=LOG_LEVEL,
)

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

    today = date.today()
    logger.info(f"Starting SubjectiveQA inference on {today}")
    try:
        # Dataset Loading
        dataset = load_dataset(
            "gtfintechlab/subjectiveqa", "5768", split="test", trust_remote_code=True
        )
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
        subjectiveqa_prompt = subjectiveqa_fewshot_prompt
    elif args.prompt_format == "zeroshot":
        subjectiveqa_prompt = subjectiveqa_zeroshot_prompt

    batch_size = args.batch_size
    total_batches = len(questions) // batch_size + int(len(questions) % batch_size > 0)
    logger.info(f"Processing {len(questions)} rows in {total_batches} batches.")

    question_batches = chunk_list(questions, batch_size)
    answer_batches = chunk_list(answers, batch_size)

    for batch_idx, (question_batch, answer_batch) in enumerate(
        zip(question_batches, answer_batches)
    ):
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
                time.sleep(
                    random.uniform(0.5, 1.5)
                )  # Add a short randomized delay between features
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
        results_path = (
            RESULTS_DIR
            / "subjectiveqa"
            / f"subjectiveqa_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Inference completed. Results saved to {results_path}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")
        logger.error(traceback.format_exc())
        return None

    return df
