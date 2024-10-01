import time
from datetime import date

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import word_tokenize

import together
from superflue.together_code.prompts import numclaim_prompt
from superflue.together_code.chat import get_stop_tokens

nltk.download("punkt")

from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="numclaim_inference",
    log_file=LOG_DIR / "numclaim_inference.log",
    level=LOG_LEVEL,
)


def numclaim_inference(args):
    
    today = date.today()
    logger.info(f"Starting Numclaim inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/Numclaim", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    sentences = []
    llm_responses = []
    llm_first_word_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on {args.task}...")
    # start_t = time.time()
    for i in range(len(dataset["test"])): # type: ignore
        time.sleep(5.0)
        sentence = dataset["test"][i]["context"] # type: ignore
        actual_label = dataset["test"][i]["response"] # type: ignore
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            logger.info(f"Processing sentence {i+1}/{len(dataset['test'])}") # type: ignore
            model_response = together.Complete.create(
                prompt=numclaim_prompt(sentence),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=get_stop_tokens(args.model),
            )
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"]
            words = word_tokenize(response_label.strip())
            llm_first_word_responses.append(words[0])
            llm_responses.append(response_label)
            logger.info(f"Model response: {response_label}")
            time.sleep(10)

        except Exception as e:
            logger.error(f"Error processing sentence {i+1}: {e}")
            time.sleep(10.0)
            complete_responses.append(None)
            llm_responses.append(None)
            llm_first_word_responses.append(None)

        df = pd.DataFrame(
            {
                "sentences": sentences,
                "complete_responses": complete_responses,
                "llm_responses": llm_responses,
                "llm_first_word_responses": llm_first_word_responses,
                "actual_labels": actual_labels,
            }
        )
        results_path = (
            RESULTS_DIR
            / args.task
            / f"{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Intermediate results saved to {results_path}")

    logger.info(f"Inference completed. Final results saved to {results_path}")
    return df
