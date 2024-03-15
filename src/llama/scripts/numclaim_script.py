import pandas as pd 
from time import sleep, time
from datetime import date
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

today = date.today()
model = "meta-llama/Llama-2-70b-hf"
task = "numclaim"
dataset = load_dataset("gtfintechlab/Numclaim", token= "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")
api_key = "1ba68d2ffcbdad1ac7dbc992797cfa0200a9031ab7c886e6701674892ba4acbf"

# Initialize lists to store actual labels and model responses
context = []
llm_responses = []
complete_responses = []
actual_labels = []

# test_context = []
# test_response = []
# test_actual_labels = []

# Iterating through the train split of the dataset
start_t = time()
for i in range(len(dataset['train'])):
    sentence = dataset['train'][i]
    context.append(sentence['context'])
    actual_label = sentence['response']
    actual_labels.append(actual_label)
    try:
        model_response = generate(task, model, api_key, sentence['context'])
        complete_responses.append(model_response)
        response_label = model_response["output"]["choices"][0]["text"]
        print(response_label)
        llm_responses.append(response_label)
        df = pd.DataFrame({'context': context, 'complete_response': complete_responses, 'llm_responses': llm_responses, 'actual_labels': actual_labels})
        time_taken = int((time() - start_t)/60.0)
        df.to_csv(f'numclaim_results_llama_2_{today.strftime("%d_%m_%Y")}.csv', index=False)
        
    except Exception as e:
                print(e)
                i = i - 1
                sleep(10.0)


# Iterating through the test split of the dataset
for i in range(len(dataset['test'])):
    sentence = dataset['test'][i]
    actual_label = sentence['response']  # Assuming the response key contains the actual label
    actual_labels.append(actual_label)
    try:
        model_response = generate("numclaim", model, api_key, sentence['context'])
        complete_responses.append(model_response)
        response_label = model_response["output"]["choices"][0]["text"]
        llm_responses.append(response_label)
        df = pd.DataFrame({'context': context,'complete_response': complete_responses, 'llm_responses': llm_responses, 'actual_labels': actual_labels})
        time_taken = int((time() - start_t)/60.0)
        # sleep(1.0)
        df.to_csv(f'numclaim_results_llama_2_{today.strftime("%d_%m_%Y")}.csv', index=False)
        
    except Exception as e:
        print(e)
        i = i - 1
        sleep(10.0)
        

# Evaluating metrics for the train split
accuracy = accuracy_score(actual_labels, llm_responses)
precision = precision_score(actual_labels, llm_responses, average='micro')
recall = recall_score(actual_labels, llm_responses, average='micro')
f1 = f1_score(actual_labels, llm_responses, average='micro')
# roc_auc = roc_auc_score(actual_labels, llm_responses)


# Creating DataFrames for metrics
metrics = pd.DataFrame({'accuracy': [accuracy],
                        'precision': [precision],
                        'recall': [recall],
                        'f1_score': [f1]})
                        #'roc_auc': [roc_auc]})

# Saving DataFrames to CSV files
metrics.to_csv('numclaim_llama2_metrics.csv', index=False)
# test_metrics.to_csv('numclaim_llama2_test_metrics.csv', index=False)


