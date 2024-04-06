import pandas as pd
from time import sleep, time
from datetime import date
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

today = date.today()

model = "meta-llama/Llama-2-7b-hf"
task = "fomc"
dataset = load_dataset(
    "gtfintechlab/fomc_communication", token="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"
)
api_key = "d88605e587297179a8a38ba7769c8cc8ce3a62ba173add159e7155dec7f1d30e"

# Initialize lists to store actual labels and model responses
context = []
llm_responses = []
complete_responses = []
actual_labels = []

# Iterating through the train split of the dataset
start_t = time()
for sentence in dataset["train"]:
    context.append(sentence["sentence"])
    actual_label = sentence["label"]
    actual_labels.append(actual_label)
    model_response = generate("fomc", model, api_key, sentence["sentence"])
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

    df.to_csv(f'fomc_train_llama_2_7b',index = False)

# Iterating through the train split of the dataset
for sentence in dataset["test"]:
    context.append(sentence["sentence"])
    actual_label = sentence[
        "label"
    ]
    actual_labels.append(actual_label)
    model_response = generate("fomc", model, api_key, sentence["sentence"])
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
    #time_taken = time() - start_t
    df.to_csv(f'fomc_test_llama_2_7b.csv', index=False)
    
    



