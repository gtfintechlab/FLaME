import logging
import time
from datetime import date
from pathlib import Path

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import word_tokenize

import together
from superflue.together_code.prompts import numclaim_prompt
from superflue.together_code.tokens import tokens

nltk.download("punkt")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def numclaim_inference(args):
    together.api_key = args.api_key
    today = date.today()
    logger.info(f"Starting Numclaim inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/Numclaim", token=args.hf_token)

    # Initialize lists to store actual labels and model responses
    sentences = []
    llm_responses = []
    llm_first_word_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on {args.task}...")
    start_t = time.time()
    for i in range(len(dataset["test"])):
        time.sleep(5.0)
        sentence = dataset["test"][i]["context"]
        actual_label = dataset["test"][i]["response"]
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            logger.info(f"Processing sentence {i+1}/{len(dataset['test'])}")
            model_response = together.Complete.create(
                prompt=numclaim_prompt(sentence),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
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
            ROOT_DIR
            / "results"
            / args.task
            / f"{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Intermediate results saved to {results_path}")

    logger.info(f"Inference completed. Final results saved to {results_path}")
    return df
