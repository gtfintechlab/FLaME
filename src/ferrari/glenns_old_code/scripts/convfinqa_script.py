import time

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from together_pipeline import generate

model = "meta-llama/Llama-2-7b-hf"
task = "convfinqa"
dataset = load_dataset(
    "gtfintechlab/ConvFinQa", token="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"
)
api_key = ""


context = []
llm_responses = []
complete_responses = []
actual_labels = []

for entry in dataset["train"]:
    pre_text = " ".join(entry["pre_text"])
    post_text = " ".join(entry["post_text"])

    table_text = " ".join([" ".join(map(str, row)) for row in entry["table_ori"]])

    question_0 = str(entry["question_0"]) if entry["question_0"] is not None else ""
    question_1 = str(entry["question_1"]) if entry["question_1"] is not None else ""
    answer_0 = str(entry["answer_0"]) if entry["answer_0"] is not None else ""
    answer_1 = str(entry["answer_1"]) if entry["answer_1"] is not None else ""

    combined_text = f"{pre_text} {post_text} {table_text} {question_0} {answer_0} {question_1} {answer_1}"
    context.append(combined_text)

    actual_label = entry["answer_1"]
    actual_labels.append(actual_label)
    model_response = generate(task, model, api_key, combined_text)
    complete_responses.append(model_response)

    response_label = model_response["output"]["choices"][0]["text"]
    llm_responses.append(response_label)

    df = pd.DataFrame(
        {
            "context": context,
            "complete_responses": complete_responses,
            "response": llm_responses,
            "actual_label": actual_labels,
        }
    )

    df.to_csv("convfinqa_llama_2.csv", index=False)
