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
train_context = []
train_response = []
train_actual_labels = []

test_context = []
test_response = []
test_actual_labels = []

# Iterating through the train split of the dataset
for sentence in dataset['train']:
    train_context.append(sentence['context'])
    train_actual_label = sentence['response']
    train_actual_labels.append(train_actual_label)
    model_response = generate("numclaim", model, api_key, sentence['context'])
    train_response_label = model_response["output"]["choices"][0]["text"]
    print(train_response_label)
    train_response.append(train_response_label)
    train_df = pd.DataFrame({'context': train_context, 'response': train_response, 'actual_label': train_actual_labels})
    train_df.to_csv('numclaim_train_llama_2.csv', index=False)

# Iterating through the test split of the dataset
for sentence in dataset['test']:
    test_context.append(sentence['context'])
    test_actual_label = sentence['response']  # Assuming the response key contains the actual label
    test_actual_labels.append(test_actual_label)
    model_response = generate("numclaim", model, api_key, sentence['context'])
    test_response_label = model_response["output"]["choices"][0]["text"]
    test_response.append(test_response_label)
    test_df = pd.DataFrame({'context': test_context, 'response': test_response, 'actual_label': test_actual_labels})
    test_df.to_csv('numclaim_test_llama_2.csv', index=False)

# Evaluating metrics for the train split
train_accuracy = accuracy_score(train_actual_labels, train_response)
train_precision = precision_score(train_actual_labels, train_response)
train_recall = recall_score(train_actual_labels, train_response)
train_f1 = f1_score(train_actual_labels, train_response)
train_roc_auc = roc_auc_score(train_actual_labels, train_response)

# Evaluating metrics for the test split
test_accuracy = accuracy_score(test_actual_labels, test_response)
test_precision = precision_score(test_actual_labels, test_response)
test_recall = recall_score(test_actual_labels, test_response)
test_f1 = f1_score(test_actual_labels, test_response)
test_roc_auc = roc_auc_score(test_actual_labels, test_response)

# Creating DataFrames for metrics
train_metrics = pd.DataFrame({'accuracy': [train_accuracy],
                              'precision': [train_precision],
                              'recall': [train_recall],
                              'f1_score': [train_f1],
                              'roc_auc': [train_roc_auc]})

test_metrics = pd.DataFrame({'accuracy': [test_accuracy],
                             'precision': [test_precision],
                             'recall': [test_recall],
                             'f1_score': [test_f1],
                             'roc_auc': [test_roc_auc]})

# Saving DataFrames to CSV files
train_metrics.to_csv('numclaim_llama2_train_metrics.csv', index=False)
test_metrics.to_csv('numclaim_llama2_test_metrics.csv', index=False)


