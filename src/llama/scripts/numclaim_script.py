import pandas as pd 
import time
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


model = "meta-llama/Llama-2-7b-hf"
task = "numclaim"
dataset = load_dataset("gtfintechlab/Numclaim", token= "")
api_key = ""

# Initialize lists to store actual labels and model responses
context = []
llm_responses = []
complete_responses = []
actual_labels = []

# test_context = []
# test_response = []
# test_actual_labels = []

# Iterating through the train split of the dataset
for sentence in dataset['train']:
    context.append(sentence['context'])
    actual_label = sentence['response']
    actual_labels.append(actual_label)
    model_response = generate("numclaim", model, api_key, sentence['context'])
    complete_responses.append(model_response)
    response_label = model_response["output"]["choices"][0]["text"]
    print(response_label)
    llm_responses.append(response_label)
    train_df = pd.DataFrame({'context': context, 'complete_response': complete_responses, 'llm_response': llm_responses, 'actual_label': actual_labels})
    train_df.to_csv('numclaim_train_llama_2.csv', index=False)
 

# Iterating through the test split of the dataset
for sentence in dataset['test']:
    context.append(sentence['context'])
    actual_label = sentence['response']  # Assuming the response key contains the actual label
    actual_labels.append(actual_label)
    model_response = generate("numclaim", model, api_key, sentence['context'])
    response_label = model_response["output"]["choices"][0]["text"]
    llm_responses.append(response_label)
    test_df = pd.DataFrame({'context': context, 'response': llm_responses, 'actual_label': actual_labels})
    train_df.to_csv('numclaim_train_llama_2.csv', index=False)
    

# Evaluating metrics for the train split
train_accuracy = accuracy_score(actual_labels, llm_responses)
train_precision = precision_score(actual_labels, llm_responses, average='micro')
train_recall = recall_score(actual_labels, llm_responses, average='micro')
train_f1 = f1_score(actual_labels, llm_responses, average='micro')
# train_roc_auc = roc_auc_score(actual_labels, llm_responses)

# # Evaluating metrics for the test split
# test_accuracy = accuracy_score(test_actual_labels, test_response)
# test_precision = precision_score(test_actual_labels, test_response)
# test_recall = recall_score(test_actual_labels, test_response)
# test_f1 = f1_score(test_actual_labels, test_response)
# test_roc_auc = roc_auc_score(test_actual_labels, test_response)

# Creating DataFrames for metrics
train_metrics = pd.DataFrame({'accuracy': [train_accuracy],
                              'precision': [train_precision],
                              'recall': [train_recall],
                              'f1_score': [train_f1]})
                            #   'roc_auc': [train_roc_auc]})

# test_metrics = pd.DataFrame({'accuracy': [test_accuracy],
#                              'precision': [test_precision],
#                              'recall': [test_recall],
#                              'f1_score': [test_f1],
#                              'roc_auc': [test_roc_auc]})

# Saving DataFrames to CSV files
train_metrics.to_csv('numclaim_llama2_train_metrics.csv', index=False)
# test_metrics.to_csv('numclaim_llama2_test_metrics.csv', index=False)


