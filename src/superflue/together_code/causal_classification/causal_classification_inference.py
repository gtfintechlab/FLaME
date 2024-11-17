from litellm import completion 
import pandas as pd
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from datasets import load_dataset
from datetime import date
from superflue.together_code.prompts import causal_classification_prompt
from superflue.together_code.tokens import tokens
from superflue.config import LOG_LEVEL, LOG_DIR, RESULTS_DIR
from superflue.utils.logging_utils import setup_logger

logger = setup_logger(
    name="causal_classification_inference",
    log_file=LOG_DIR / "causal_classification_inference.log",
    level=LOG_LEVEL,
)

def causal_classification_inference(args):
    today = date.today()
    logger.info(f"Starting Causal Classification inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/CausalClassification", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    texts = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on causal classification task with model {args.model}")
    # start_t = time.time()
    for i in range(len(dataset["test"])):  # type: ignore
        text = dataset["test"][i]["text"]  # type: ignore
        actual_label = dataset["test"][i]["label"]  # type: ignore
        texts.append(text)
        actual_labels.append(actual_label)
        try:
            logger.info(f"Processing text {i+1}/{len(dataset['test'])}")  # type: ignore
    
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": causal_classification_prompt(text)}],
                temperature=args.temperature,
                tokens=args.max_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            logger.info(f"Model response: {model_response.choices[0].message.content}")  # type: ignore
            complete_responses.append(model_response.choices[0].message.content) # type: ignore
            response_label = model_response.choices[0].message.content # type: ignore
            llm_responses.append(response_label)

            df = pd.DataFrame(
                {
                    "texts": texts,
                    "llm_responses": llm_responses,
                    "actual_labels": actual_labels,
                    "complete_responses": complete_responses,
                }
            )

        except Exception as e:
            logger.error(f"Error processing text {i+1}: {e}")
            time.sleep(10.0)
            continue

        results_path = (
            RESULTS_DIR
            / "causal_classification"
            / f"causal_classification_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")

    return df
