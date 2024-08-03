# from datasets import load_dataset

# dataset = load_dataset("gtfintechlab/banking77", token="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")

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from src.together.prompts import banking77_prompt
from src.together.tokens import tokens

# print(dataset)
import together

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def banking77_inference(args):
    together.api_key = args.api_key
    dataset = load_dataset("gtfintechlab/banking77", token=args.hf_token)
    today = date.today()
    documents = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    for i in range(len(dataset["test"])):
        document = dataset["test"][i]["text"]
        actual_label = dataset["test"][i]["label"]
        documents.append(document)
        actual_labels.append(actual_label)
        try:
            model_response = together.Complete.create(
                prompt=banking77_prompt(document),
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
                    "documents": documents,
                    "llm_responses": llm_responses,
                    "actual_labels": actual_labels,
                    "complete_responses": complete_responses,
                }
            )
            results_path = (
                ROOT_DIR
                / "results"
                / args.task
                / f"{args.task}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(results_path, index=False)
            logger.info(f"Inference completed for {i}. Results saved to {results_path}")
        except Exception as e:
            print(e)
            i = i - 1
            time.sleep(20.0)

        return df
