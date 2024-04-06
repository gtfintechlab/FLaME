<<<<<<< HEAD
from datetime import date
import pandas as pd 

=======
import pandas as pd
>>>>>>> origin/baslines/together/huzaifa
import time
from together_pipeline import generate
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import date
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

<<<<<<< HEAD
# use this as reference 
# goal: run scripts on each model (low priority)
# get the output working correctly (high priority): default model, llama 
# move the together_pipeline.py file (not really a pipeline)

model = "meta-llama/Llama-2-7b-chat-hf"
api_key = "d88605e587297179a8a38ba7769c8cc8ce3a62ba173add159e7155dec7f1d30e"
task = "fpb"

today = date.today()
start_t = time.time()
configs = ["sentences_allagree"]
for config in configs:
    dataset = load_dataset("financial_phrasebank", config , token= "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")

=======
today = date.today()
model = "meta-llama/Llama-2-7b-hf"
task = "fpb"
api_key = "1ba68d2ffcbdad1ac7dbc992797cfa0200a9031ab7c886e6701674892ba4acbf"

configs = ["sentences_50agree", "sentences_66agree", "sentences_75agree", "sentences_allagree"]
for config in configs:
    dataset = load_dataset("financial_phrasebank", config, token="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")

    # Initialize lists to store actual labels and model responses
>>>>>>> origin/baslines/together/huzaifa
    sentences = []
    llm_responses = []
    llm_first_word_responses = []
    actual_labels = []
    complete_responses = []

<<<<<<< HEAD
    # 
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
    
=======
    # Iterating through the train split of the dataset
    start_t = time.time()
    for i in range(len(dataset['train'])):
        sentence = dataset['train'][i]['sentence']
        actual_label = dataset['train'][i]['label']
        sentences.append(sentence)
        actual_labels.append(actual_label)
        try:
            model_response = generate(task, model, api_key, sentence)
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"]
            words = word_tokenize(response_label.strip())
            llm_first_word_responses.append(words[0])
            llm_responses.append(response_label)
            df = pd.DataFrame({'sentences': sentences, 'complete_responses': complete_responses, 'llm_responses': llm_responses, 'llm_first_word_responses': llm_first_word_responses, 'actual_labels': actual_labels})
            df.to_csv(f'fpb_results_llama_2_{config}_{today.strftime("%d_%m_%Y")}.csv', index=False)
        except Exception as e:
            print(e)
            i = i - 1
            time.sleep(10.0)

    time_taken = time.time() - start_t

>>>>>>> origin/baslines/together/huzaifa
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

