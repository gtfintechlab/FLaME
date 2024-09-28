import logging
import time
from datetime import date
from pathlib import Path
import together

import pandas as pd
from datasets import load_dataset
import nltk

# Custom imports for FNXL prompt and token handling
from superflue.together_code.prompts import fnxl_prompt  # Custom prompt function for FNXL
from superflue.together_code.tokens import tokens  # Custom token handling function for FNXL

nltk.download("punkt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def fnxl_inference(args):
    today = date.today()
    logger.info(f"Starting FNXL inference on {today}")

    logger.info("Loading dataset...")
    # Replace with your Hugging Face or custom FNXL dataset path
    dataset = load_dataset("gtfintechlab/fnxl", token=args.hf_token)

    # Initialize lists to store information
    sentences = []
    numerals_tags = []
    companies = []
    predicted_labels = []
    actual_labels = []
    complete_responses = []

    logger.info(f"Starting inference on {args.task}...")
    start_t = time.time()
    for i in range(len(dataset["test"])): # type: ignore
        sentence = dataset["test"][i]["sentence"] # type: ignore
        numeral_tag = dataset["test"][i]["numerals-tags"] # type: ignore
        company = dataset["test"][i]["company"] # type: ignore
        actual_label = dataset["test"][i]["ner_tags"] # type: ignore

        sentences.append(sentence)
        numerals_tags.append(numeral_tag)
        companies.append(company)
        actual_labels.append(actual_label)

        try:
            logger.info(f"Processing sentence {i+1}/{len(dataset['test'])}") # type: ignore
            # FNXL-specific prompt to classify numerals in financial sentences
            model_response = together.Complete.create(
                prompt=fnxl_prompt(sentence, numeral_tag, company),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            complete_responses.append(model_response)
            predicted_label = model_response["output"]["choices"][0]["text"] # type: ignore
            predicted_labels.append(predicted_label)

            df = pd.DataFrame(
                {
                    "sentences": sentences,
                    "numerals_tags": numerals_tags,
                    "companies": companies,
                    "predicted_labels": predicted_labels,
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
