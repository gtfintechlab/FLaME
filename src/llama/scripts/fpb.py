import pandas as pd 
import time
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


model = "meta-llama/Llama-2-7b-hf"
task = "fpb"
api_key = ""


configs = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
for config in configs:
    dataset = load_dataset("financial_phrasebank", config , token= "")

# Initialize lists to store actual labels and model labels
    train_sentence = []
    train_label = []
    train_actual_labels = []

    # Iterating through the train split of the dataset
    for sentence in dataset['train']:
        train_sentence.append(sentence['sentence'])
        train_actual_label = sentence['label']
        train_actual_labels.append(train_actual_label)
        model_label = generate("fpb", model, api_key, sentence['sentence'])
        train_label_label = model_label["output"]["choices"][0]["text"]
        print(train_label_label)
        train_label.append(train_label_label)
        train_df = pd.DataFrame({'sentence': train_sentence, 'label': train_label, 'actual_label': train_actual_labels})
        train_df.to_csv('fpb_train_llama_2.csv', index=False)

    # Evaluating metrics for the train split
    train_accuracy = accuracy_score(train_actual_labels, train_label)
    train_precision = precision_score(train_actual_labels, train_label)
    train_recall = recall_score(train_actual_labels, train_label)
    train_f1 = f1_score(train_actual_labels, train_label)
    train_roc_auc = roc_auc_score(train_actual_labels, train_label)

    # Creating DataFrames for metrics
    train_metrics = pd.DataFrame({'accuracy': [train_accuracy],
                                'precision': [train_precision],
                                'recall': [train_recall],
                                'f1_score': [train_f1],
                                'roc_auc': [train_roc_auc]})

    # Saving DataFrames to CSV files
    train_metrics.to_csv('fpb_llama2_train_metrics.csv', index=False)