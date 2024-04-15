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
task = "finqa"
dataset = load_dataset(
    "gtfintechlab/finqa", token=""
)
api_key = ""


context = []
llm_responses = []
complete_responses =[]
actual_labels = []

for entry in dataset["train"]:
    pre_text = " ".join(entry['pre_text'])
    post_text = " ".join(entry['post_text'])
    table_text = " ".join([" ".join(row) for row in entry['table_ori']])
    combined_text = f"{pre_text} {post_text} {table_text} {entry['question']}"
    
    context.append(combined_text)

    actual_label = entry["answer"]
    actual_labels.append(actual_label)
    model_response = generate(task, model, api_key, combined_text)
    complete_responses.append(model_response)
    
    response_label = model_response["output"]["choices"][0]["text"]
    llm_responses.append(response_label)
   
    df = pd.DataFrame({
        "context": context,
        "complete_responses": complete_responses,
        "response": llm_responses,
        "actual_label": actual_labels,
    })

    df.to_csv("finqa_llama_2.csv", index=False)