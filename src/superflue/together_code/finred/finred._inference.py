import logging
import time
from datetime import date
from pathlib import Path

import nltk
import pandas as pd
from datasets import load_dataset

# Mock imports for the custom FinRED prompt and tokens
from superflue.together_code.prompts import finred_prompt  # You need to implement finred_prompt for FinRED
from superflue.together_code.tokens import tokens  # Token logic for FinRED

nltk.download("punkt")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def finred_inference(args):
    today = date.today()
    logger.info(f"Starting FinRED inference on {today}")

    logger.info("Loading dataset...")
    # Replace "finred_dataset" with the appropriate Hugging Face dataset for FinRED or custom dataset
    dataset = load_dataset("gtfintechlab/FinRed", token=args.hf_token)

    # Initialize lists to store sentences, actual labels, and model responses
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on {args.task}...")
    start_t = time.time()
    for i in range(len(dataset["test"])):
        sentence = dataset["test"][i]["content"]
        actual_label = dataset["test"][i]["annotations"]
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            logger.info(f"Processing sentence {i+1}/{len(dataset['test'])}")
            # FinRED-specific prompt logic, create the prompt for relation extraction
            model_response = together.Complete.create(
                prompt=finred_prompt(sentence),
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
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
