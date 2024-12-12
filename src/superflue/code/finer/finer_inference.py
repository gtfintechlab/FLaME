import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from litellm import completion
from superflue.code.prompts import finer_prompt

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.path_utils import get_inference_save_path
from superflue.config import LOG_DIR, LOG_LEVEL
from tqdm import tqdm

logger = setup_logger(
    name="finer_inference",
    log_file=LOG_DIR / "finer_inference.log",
    level=LOG_LEVEL,
)


def finer_inference(args):
    """Run inference on the FINER dataset using the specified model."""
    today = date.today()
    logger.info(f"Starting FINER inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/finer-ord-bio", trust_remote_code=True)

    # Initialize lists to store data
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on FINER with model {args.model}...")

    for i in tqdm(range(len(dataset["test"])), desc="Processing sentences"):  # type: ignore
        sentence = dataset["test"][i]["tokens"]  # type: ignore
        actual_label = dataset["test"][i]["tags"]  # type: ignore
        sentences.append(sentence)
        actual_labels.append(actual_label)

        try:
            logger.debug(f"Processing sentence {i+1}/{len(dataset['test'])}")  # type: ignore
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": finer_prompt(sentence)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model),
            )

            complete_responses.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore
            logger.debug(f"Model response: {response_text}")
            llm_responses.append(response_text)

        except Exception as e:
            logger.error(f"Error processing sentence {i+1}: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(10.0)
            continue

    # Create DataFrame with results
    df = pd.DataFrame(
        {
            "sentences": sentences,
            "llm_responses": llm_responses,
            "actual_labels": actual_labels,
            "complete_responses": complete_responses,
        }
    )

    # Save results using consistent path utility
    results_path = get_inference_save_path(args.dataset, args.model)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
