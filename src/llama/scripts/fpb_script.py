import pandas as pd 
import time
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


model = "meta-llama/Llama-2-7b-hf"
task = "fpb"
api_key = "1ba68d2ffcbdad1ac7dbc992797cfa0200a9031ab7c886e6701674892ba4acbf"


configs = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
for config in configs:
    dataset = load_dataset("financial_phrasebank", config , token= "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")

# Initialize lists to store actual labels and model labels
    sentences = []
    llm_responses = []
    actual_labels = []
    complete_responses =[]

    # Iterating through the train split of the dataset
    for sentence in dataset['train']:
        sentences.append(sentence['sentence'])
        actual_label = sentence['label']
        actual_labels.append(actual_label)
        model_label = generate(task, model, api_key, sentence['sentence'])
        complete_responses.append(model_label)
        llm_response = model_label["output"]["choices"][0]["text"]
        print(llm_response)
        llm_responses.append(llm_response)
        df = pd.DataFrame({'sentences': sentences, 'complete_responses': complete_responses, 'llm_responses': llm_responses, 'actual_label': actual_labels})
        df.to_csv('fpb_llama_2.csv', index=False)

    # Evaluating metrics for the train split
    accuracy = accuracy_score(actual_labels, llm_responses)
    precision = precision_score(actual_labels, llm_responses)
    recall = recall_score(actual_labels, llm_responses)
    f1 = f1_score(actual_labels, llm_responses)
    roc_auc = roc_auc_score(actual_labels, llm_responses)

    # Creating DataFrames for metrics
    metrics = pd.DataFrame({'accuracy': [accuracy],
                                'precision': [precision],
                                'recall': [recall],
                                'f1_score': [f1],
                                'roc_auc': [roc_auc]})

    # Saving DataFrames to CSV files
    metrics.to_csv('fpb_llama2_metrics.csv', index=False)