import time
import pandas as pd
from datasets import load_dataset
from litellm import completion
from datetime import date
from superflue.code.prompts import finqa_prompt
from superflue.utils.path_utils import get_inference_path
from superflue.utils.logging_utils import get_logger
from tqdm import tqdm

# Get logger for this module
logger = get_logger(__name__)


def finqa_inference(args):
    """Run inference on the FinQA dataset using the specified model."""
    today = date.today()
    logger.info(f"Starting FinQA inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/finqa", trust_remote_code=True)

    # Initialize lists to store data
    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on FinQA with model {args.model}...")

    for entry in tqdm(dataset["test"], desc="Processing entries"):  # type: ignore
        pre_text = " ".join(entry["pre_text"])  # type: ignore
        post_text = " ".join(entry["post_text"])  # type: ignore
        table_text = " ".join([" ".join(row) for row in entry["table_ori"]])  # type: ignore
        combined_text = f"{pre_text} {post_text} {table_text} {entry['question']}"  # type: ignore
        context.append(combined_text)
        actual_label = entry["answer"]  # type: ignore
        actual_labels.append(actual_label)

        try:
            logger.debug(f"Processing entry with question: {entry['question']}")  # type: ignore
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": finqa_prompt(combined_text)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model)
            )

            # Log and process the response
            logger.debug(f"Model response: {model_response}")
            complete_responses.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore
            llm_responses.append(response_text)

        except Exception as e:
            logger.error(f"Error processing entry: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(10.0)
            continue

    # Create DataFrame with results
    df = pd.DataFrame(
        {
            "context": context,
            "llm_responses": llm_responses,
            "actual_label": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Save results using consistent path utility
    results_path = get_inference_path(args.dataset, args.model)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    return df
