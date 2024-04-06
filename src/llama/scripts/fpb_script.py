
from datetime import date
import pandas as pd 


import time
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import date

model = "meta-llama/Llama-2-7b-chat-hf"
api_key = ""
task = "fpb"

today = date.today()
start_t = time.time()
configs = ["sentences_allagree"]
for config in configs:
    dataset = load_dataset("financial_phrasebank", config , token= "")
    sentences = []
    llm_responses = []
    llm_first_word_responses = []
    actual_labels = []
    complete_responses = []

    for data_point in dataset['train']:
        sentences.append(data_point['sentence'])
        actual_label = data_point['label']
        actual_labels.append(actual_label)
        success = False
        while not success:
            try:
                model_label = generate(task, model, api_key, data_point['sentence'])
                success = True
            except Exception as e:
                print(e)
                time.sleep(20.0)  

        complete_responses.append(model_label)
        llm_response = model_label["output"]["choices"][0]["text"]
        print(llm_response)
        llm_responses.append(llm_response)
    #time_taken = time() - start_t
    df = pd.DataFrame({'sentence': sentences, 'complete_responses': complete_responses, 'llm_responses': llm_responses, 'actual_label': actual_labels})
    df.to_csv(f'fpb_llama_2_7b.csv', index=False)
    



    # Evaluating metrics for the train split
    accuracy = accuracy_score(actual_labels, llm_first_word_responses)
    precision = precision_score(actual_labels, llm_first_word_responses, average='micro')
    recall = recall_score(actual_labels, llm_first_word_responses, average='micro')
    f1 = f1_score(actual_labels, llm_first_word_responses, average='micro')
    # roc_auc = roc_auc_score(actual_labels, llm_first_word_responses) # Uncomment if applicable

    # Creating DataFrames for metrics
    metrics = pd.DataFrame({'accuracy': [accuracy],
                            'precision': [precision],
                            'recall': [recall],
                            'f1_score': [f1]})
                            #'roc_auc': [roc_auc]}) # Uncomment if applicable

    # Saving DataFrames to CSV files
    metrics.to_csv(f'fpb_llama2_metrics_{config}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)

