import time
from datetime import date

import nltk
import pandas as pd
from datasets import load_dataset

import together
from together import Together
from superflue.together_code.prompts import finentity_prompt
from superflue.together_code.tokens import tokens

nltk.download("punkt")

from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="finentity_inference",
    log_file=LOG_DIR / "finentity_inference.log",
    level=LOG_LEVEL,
)

client = Together()

def finentity_inference(args):
    
    today = date.today()
    logger.info(f"Starting FinEntity inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/finentity", "5768", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on FinEntity...")
    # start_t = time.time()
    for i in range(len(dataset["test"])): # type: ignore
        sentence = dataset["test"][i]["content"] # type: ignore
        actual_label = dataset["test"][i]["annotations"] # type: ignore
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            logger.debug(f"Processing sentence {i+1}/{len(dataset['test'])}") # type: ignore
            model_response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": finentity_prompt(sentence)}],
            tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop=tokens(args.model),
            )
            
            complete_responses.append(model_response)
            logger.info(f"Model response: {model_response.choices[0].message.content}") # type: ignore
            response_label = model_response.choices[0].message.content # type: ignore
            llm_responses.append(response_label)

            df = pd.DataFrame(
                {
                    "sentences": sentences,
                    "llm_responses": llm_responses,
                    "actual_labels": actual_labels,
                    "complete_responses": complete_responses,
                }
            )

        except Exception as e:
            logger.error(f"Error processing sentence {i+1}: {e}")
            time.sleep(20.0)
            continue

    results_path = (
        RESULTS_DIR
        / "finentity"
        / f"finentity_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
