import pandas as pd
import time
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


model = "meta-llama/Llama-2-7b-hf"
task = "banking77"
dataset = load_dataset(
    "gtfintechlab/banking77", token=""
)
api_key = ""

# Initialize lists to store actual labels and model responses
context = []
llm_responses = []
complete_responses = []
actual_labels = []

# Iterating through the train split of the dataset
for sentence in dataset["train"]:
    context.append(sentence["text"])
    actual_label = sentence["label"]
    actual_labels.append(actual_label)
    model_response = generate(task, model, api_key, sentence["text"])
    complete_responses.append(model_response)
    response_label = model_response["output"]["choices"][0]["text"]
    print(response_label)
    llm_responses.append(response_label)
    df = pd.DataFrame(
        {
            "context": context,
            "complete_response": complete_responses,
            "response": llm_responses,
            "actual_label": actual_labels,
        }
    )
    df.to_csv("banking77_results_llama_2.csv", index=False)

# Iterating through the train split of the dataset
for sentence in dataset["test"]:
    context.append(sentence["text"])
    actual_label = sentence[
        "label"
    ]
    actual_labels.append(actual_label)
    model_response = generate(task, model, api_key, sentence["sentence"])
    complete_responses.append(model_response)
    response_label = model_response["output"]["choices"][0]["text"]
    llm_responses.append(response_label)
    df = pd.DataFrame(
        {
            "context": context,
            "complete_response": complete_responses,
            "response": llm_responses,
            "actual_label": actual_labels,
        }
    )
    df.to_csv("banking77_results_llama_2.csv", index=False)

# Evaluating metrics for the train split
# accuracy = accuracy_score(actual_labels, llm_responses)
# precision = precision_score(actual_labels, llm_responses)
# recall = recall_score(actual_labels, llm_responses)
# f1 = f1_score(actual_labels, llm_responses)
# roc_auc = roc_auc_score(actual_labels, llm_responses)

# # Creating DataFrames for metrics
# metrics = pd.DataFrame(
#     {
#         "accuracy": [accuracy],
#         "precision": [precision],
#         "recall": [recall],
#         "f1_score": [f1],
#         "roc_auc": [roc_auc],
#     }
# )

# # Saving DataFrames to CSV files
# metrics.to_csv("banking77_llama2_metrics.csv", index=False)