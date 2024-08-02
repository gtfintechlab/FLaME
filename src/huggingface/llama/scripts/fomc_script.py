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

# today = date.today()
model = "meta-llama/Llama-2-7b-chat-hf"
task = "fomc"
dataset = load_dataset("gtfintechlab/fomc_communication", token="")
api_key = ""

# Initialize lists to store actual labels and model responses
context = []
llm_responses = []
llm_first_word_responses = []
complete_responses = []
actual_labels = []

# Iterating through the train split of the dataset
# start_t = time()
for i in range(len(dataset["train"])):
    sentence = dataset["train"][i]["sentence"]
    context.append(sentence)
    actual_label = dataset["train"][i]["label"]
    actual_labels.append(actual_label)
    model_response = generate("fomc", model, api_key, sentence)
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
    df.to_csv("fomc_train_results_llama_2_7b.csv", index=False)

# Iterating through the test split of the dataset
for i in range(len(dataset["test"])):
    sentence = dataset["test"][i]["sentence"]
    context.append(sentence)
    actual_label = dataset["test"][i]["label"]
    actual_labels.append(actual_label)
    model_response = generate("fomc", model, api_key, sentence)
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

    df.to_csv(f"fomc_results_test_llama_2_7b.csv", index=False)


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
#     }#
# )
# # Saving DataFrames to CSV files
# metrics.to_csv("fomc_llama2_metrics.csv", index=False)
