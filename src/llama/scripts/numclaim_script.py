import pandas as pd 
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers
from transformers import pipeline
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from datasets import load_dataset





#load dataset from labs hf
dataset = load_dataset("gtfintechlab/Numclaim", token= "")
#load model from hf
model = "meta-llama/Llama-2-7b-hf"
#load tokenizer for llama
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

#this is the pipeline object
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map = "auto")

#this is our method to call llama
def get_llama_response(sentence:str) -> None:

    #prompt from the paper
    prompt = f'''Discard all the previous instructions. Behave like you are an expert sentence senti-
    ment classifier. Classify the following sentence into ‘INCLAIM’, or ‘OUTOFCLAIM’ class.
    Label ‘INCLAIM’ if consist of a claim and not just factual past or present information, or
    ‘OUTOFCLAIM’ if it has just factual past or present information. Provide the label in the
    first line and provide a short explanation in the second line. The sentence:{sentence}'''

    #using the pipeline object we created from above
    seq = pipe(
                prompt,
                do_sample=True,
                top_k =10,
                num_return_sequences=1,
                eos_token_id = tokenizer.eos_token_id,
                max_length = 256
                )
    return (seq[0]['generated_text'])


#calling our method on all values in the train split of our dataset
context = []
response = []
start_time = time.time()
for i in range(len(dataset['train'])):
    context.append(dataset['train'][i]['context'])
    response.append(get_llama_response(dataset['train'][i]['context']))

end_time = time.time()
print(end_time - start_time)

#evaluating
accuracy_score = accuracy_score(dataset['train']['response'], response)
precision_score = precision_score(dataset['train']['response'], response)
recall_score = recall_score(dataset['train']['response'], response)
f1_score = f1_score(dataset['train']['response'], response)
roc_auc_score = roc_auc_score(dataset['train']['response'], response)
print(accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

#creating output csv
df = pd.DataFrame({'context': context, 'response': response,'accuracy':accuracy_score,'precision':precision_score, 'recall_score':recall_score, 'f1_score':f1_score, 'roc':roc_auc_score})
df.to_csv('numclaim_train_llama_2.csv', index=False)

#doing the same thing for the test split
context = []
response = []

start_time = time.time()
for i in range(len(dataset['test'])):
    context.append(dataset['test'][i]['context'])
    response.append(get_llama_response(dataset['test'][i]['context']))
    
end_time = time.time()
print(end_time - start_time)

accuracy_score = accuracy_score(dataset['test']['response'], response)
precision_score = precision_score(dataset['test']['response'], response)
recall_score = recall_score(dataset['test']['response'], response)
f1_score = f1_score(dataset['test']['response'], response)
roc_auc_score = roc_auc_score(dataset['test']['response'], response)
print(accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

df = pd.DataFrame({'context': context, 'response': response,'accuracy':accuracy_score,'precision':precision_score, 'recall_score':recall_score, 'f1_score':f1_score, 'roc':roc_auc_score})
df.to_csv('numclaim_test_llama_2.csv', index=False)




