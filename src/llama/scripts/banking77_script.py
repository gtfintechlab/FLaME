import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from datetime import date
from time import sleep, time

nltk.download("punkt")

today = date.today()
model = "meta-llama/Llama-2-7b-hf"
task = "banking77"
dataset = load_dataset("gtfintechlab/banking77", token="")
api_key = ""

# Initializing lists to store actual labels and model responses
context = []
llm_responses = []
llm_first_word_responses = []
complete_responses = []
actual_labels = []

# Iterating through the train and test splits of the dataset
start_t = time()
for split in ["train", "test"]:
    for i, sentence in enumerate(dataset[split]):
        context.append(sentence["text"])
        actual_label = sentence["label"]
        actual_labels.append(actual_label)
        try:
            model_response = generate(task, model, api_key, sentence["text"])
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"]
            words = word_tokenize(response_label.strip())
            llm_first_word_responses.append(words[0] if words else "")
            llm_responses.append(response_label)
            time_taken = int((time() - start_t) / 60.0)
            df = pd.DataFrame(
                {
                    "context": context,
                    "complete_response": complete_responses,
                    "response": llm_responses,
                    "first_word_response": llm_first_word_responses,
                    "actual_labels": actual_labels,
                }
            )
            df.to_csv(
                f'banking77_results_llama_2_{today.strftime("%d_%m_%Y")}_{time_taken}.csv',
                index=False,
            )
        except Exception as e:
            print(e)
            sleep(10.0)

# Evaluating metrics
# accuracy = accuracy_score(actual_labels, llm_first_word_responses)
# precision = precision_score(actual_labels, llm_first_word_responses, average='micro')
# recall = recall_score(actual_labels, llm_first_word_responses, average='micro')
# f1 = f1_score(actual_labels, llm_first_word_responses, average='micro')
# # roc_auc = roc_auc_score(actual_labels, llm_first_word_responses) # Uncomment if applicable

# # Creating DataFrames for metrics
# metrics = pd.DataFrame({'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1_score': [f1]})
# # metrics.to_csv(f'banking77_llama2_metrics_{today.strftime("%d_%m_%Y")}.csv', index=False)
