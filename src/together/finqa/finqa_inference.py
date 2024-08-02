import time
from datetime import date

import pandas as pd
from datasets import load_dataset

import together
from src.together.prompts import finqa_prompt
from src.together.tokens import tokens


def finqa_inference(args):
    together.api_key = args.api_key
    today = date.today()
    # OPTIONAL TODO: make configs an argument of some kind LOW LOW LOW PRIORITY
    # configs = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    dataset = load_dataset("gtfintechlab/finqa", token=args.hf_token)
    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []
    start_t = time.time()
    for entry in dataset["test"]:  # type: ignore
        pre_text = " ".join(entry["pre_text"])  # type: ignore
        post_text = " ".join(entry["post_text"])  # type: ignore
        table_text = " ".join([" ".join(row) for row in entry["table_ori"]])  # type: ignore
        combined_text = f"{pre_text} {post_text} {table_text} {entry['question']}"  # type: ignore
        context.append(combined_text)
        actual_label = entry["answer"]  # type: ignore
        actual_labels.append(actual_label)
        try:
            model_response = together.Complete.create(
                prompt=finqa_prompt(combined_text),
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
                    "context": context,
                    "response": llm_responses,
                    "actual_label": actual_labels,
                    "complete_responses": complete_responses,
                }
            )
            time.sleep(10)

        except Exception as e:
            print(e)
            i = i - 1
            time.sleep(20.0)

    return df
