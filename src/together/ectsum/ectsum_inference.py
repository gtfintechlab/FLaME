import time
from datetime import date

import pandas as pd
from datasets import load_dataset

import together
from src.together.prompts import ectsum_prompt
from src.together.tokens import tokens


def ectsum_inference(args):
    together.api_key = args.api_key
    today = date.today()
    # OPTIONAL TODO: make configs an argument of some kind LOW LOW LOW PRIORITY
    # configs = ["documents_50agree", "documents_66agree", "documents_75agree", "documents_allagree"]
    dataset = load_dataset("gtfintechlab/ECTSum", token=args.hf_token)

    # Initialize lists to store actual labels and model responses
    documents = []
    llm_responses = []
    llm_first_word_responses = []
    actual_labels = []
    complete_responses = []

    # Iterating through the train split of the dataset
    start_t = time.time()
    for i in range(len(dataset["test"])):
        document = dataset["test"][i]["context"]
        actual_label = dataset["test"][i]["response"]
        documents.append(document)
        actual_labels.append(actual_label)
        try:
            model_response = together.Complete.create(
                prompt=ectsum_prompt(document),
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
        except Exception as e:
            print(e)
            i = i - 1
            time.sleep(20.0)

    return df
