import time

import pandas as pd
from datasets import load_dataset

from superflue.together_code.prompts import convfinqa_prompt
from superflue.together_code.tokens import tokens
from litellm import completion 

def convfinqa_inference(args):
    
    # today = date.today()
    # OPTIONAL TODO: make configs an argument of some kind LOW LOW LOW PRIORITY
    # configs = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
    dataset = load_dataset("gtfintechlab/finqa", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    # Iterating through the train split of the dataset
    # start_t = time.time()
    for entry in dataset["train"]:  # type: ignore
        pre_text = " ".join(entry["pre_text"]) # type: ignore
        post_text = " ".join(entry["post_text"]) # type: ignore

        table_text = " ".join([" ".join(map(str, row)) for row in entry["table_ori"]]) # type: ignore

        question_0 = str(entry["question_0"]) if entry["question_0"] is not None else "" # type: ignore
        question_1 = str(entry["question_1"]) if entry["question_1"] is not None else "" # type: ignore
        answer_0 = str(entry["answer_0"]) if entry["answer_0"] is not None else "" # type: ignore
        answer_1 = str(entry["answer_1"]) if entry["answer_1"] is not None else "" # type: ignore

        combined_text = f"{pre_text} {post_text} {table_text} {question_0} {answer_0} {question_1} {answer_1}"
        context.append(combined_text)

        actual_label = entry["answer_1"]  # type: ignore
        actual_labels.append(actual_label)
        try:
            model_response = completion(
                messages=[{"role": "user", "content": convfinqa_prompt(combined_text)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content
            llm_responses.append(response_label)

            df = pd.DataFrame(
                {
                    "context": context,
                    "response": llm_responses,
                    "actual_label": actual_labels,
                    "complete_responses": complete_responses,
                }
            )

        except Exception as e:
            print(e)
            # i = i - 1
            time.sleep(10.0)

    return df
