#TODO: CONFIRM HOW TO EVALUATE

import pandas as pd
import time
import nltk
from nltk.tokenize import word_tokenize
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import date

nltk.download('punkt')

today = date.today()
model = "meta-llama/Llama-2-7b-hf"
task = "finer"
dataset = load_dataset("gtfintechlab/finer_ord_encoded", token="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")
api_key = "1ba68d2ffcbdad1ac7dbc992797cfa0200a9031ab7c886e6701674892ba4acbf"

# Initializing lists to store actual labels and model responses
context = []
llm_responses = []
llm_first_word_responses = []
complete_responses = []
actual_labels = []

# Iterating through the train split of the dataset
start_t = time.time()
for i in range(len(dataset['train'])):
    sentence = dataset['train'][i]['']
    context.append(sentence)
    actual_label = dataset['train'][i]['label']
    actual_labels.append(actual_label)
    try:
        model_response = generate(task, model, api_key, sentence)
        complete_responses.append(model_response)
        response_label = model_response["output"]["choices"][0]["text"]
        words = word_tokenize(response_label.strip())
        llm_first_word_responses.append(words[0])
        llm_responses.append(response_label)
        df = pd.DataFrame({'context': context, 'complete_responses': complete_responses, 'llm_responses': llm_responses, 'llm_first_word_responses': llm_first_word_responses, 'actual_labels': actual_labels})
        df.to_csv(f'finer_results_llama_2_{today.strftime("%d_%m_%Y")}.csv', index=False)
    except Exception as e:
        print(e)
        i = i - 1
        time.sleep(10.0)

# # Evaluating metrics for the train split
# accuracy = accuracy_score(actual_labels, llm_first_word_responses)
# precision = precision_score(actual_labels, llm_first_word_responses, average='micro')
# recall = recall_score(actual_labels, llm_first_word_responses, average='micro')
# f1 = f1_score(actual_labels, llm_first_word_responses, average='micro')
# # roc_auc = roc_auc_score(actual_labels, llm_first_word_responses) # Uncomment if applicable

# # Creating DataFrames for metrics
# metrics = pd.DataFrame({'accuracy': [accuracy],
#                         'precision': [precision],
#                         'recall': [recall],
#                         'f1_score': [f1]})
#                         #'roc_auc': [roc_auc]}) # Uncomment if applicable

# # Saving DataFrames to CSV files
# metrics.to_csv(f'finer_llama2_metrics_{today.strftime("%d_%m_%Y")}.csv', index=False)
