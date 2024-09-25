import sys
sys.path.insert(0, r"C:\Users\mikad\Documents\GitHub\textgrad")
import textgrad
from textgrad import Variable
import concurrent
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import math
import random
from together import Together

# Evaluate given sample using custom evaluation function
def eval_sample(x, y, eval_fn, model):
    x = Variable(x, requires_grad=False, role_description="query to the language model")
    y = Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    inputs = dict(prediction = response, ground_truth_answer = y)
    eval_output_variable = eval_fn(inputs = inputs)
    return eval_output_variable
    
# Evaluate dataset using custom evaluation function
def eval_dataset(test_set, eval_fn, model, max_samples = None, max_workers = 24):
    if max_samples is None: 
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            future = executor.submit(eval_sample, sample[0], sample[1], eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            output = future.result()
            output = int(output.value)
            accuracy_list.append(output)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list 

# Evaluate system prompt and revert to previous prompt if new prompt has lower test accuracy
def run_val_revert(system_prompt, results, eval_fn, model, validation_dataset):
    curr_accuracy = np.mean(eval_dataset(validation_dataset, eval_fn, model))
    prev_accuracy = np.mean(results["val_accuracy"][-1])
    print(f"Val accuracy: {curr_accuracy}\nPrevious val accuracy: {prev_accuracy}")
    previous_prompt = results["prompt"][-1]

    results["new_prompt_val_accuracy"].append(curr_accuracy)
    
    if curr_accuracy < prev_accuracy:
        print(f"Rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        curr_accuracy = prev_accuracy

    results["val_accuracy"].append(curr_accuracy)

# Extract answer from model response using LLM prompt
def extract_answer(prompt):
    client = Together()
    try: 
        model_response = client.chat.completions.create(
            model='mistralai/Mixtral-8x7B-Instruct-v0.1',
            messages=[{'role': 'system', 'content': "Discard all previous instructions. Behave as if you are an text extraction model. You are only extracting the answer."},
                        {'role': 'user', 'content': prompt}],
            max_tokens=20, temperature=0.7, top_k=50, top_p=0.7,repetition_penalty=1.1
        )
        response = model_response.choices[0].message.content
        return response.strip().split()[0]
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None
    
# Load and split huggingface dataset into training, validation, and testing sets 
def load_hf_dataset(hf_token, dataset_name, extract_x, y_column, train_size = 0.5, val_size = 0.1):
    if hf_token is None:
        raise ValueError("Please provide a valid Hugging Face API token.")
    if (train_size + val_size) > 1:
        raise ValueError("Train and validation sizes must sum to less than 1.")
    dataset = load_dataset(dataset_name, token=hf_token)
    training = [(extract_x(data), data[y_column]) for data in dataset['train']]
    random.shuffle(training)
    training_data = training[:math.ceil(len(training)*train_size)]
    val_data = training[math.ceil(len(training)*val_size):]
    testing_data = [(extract_x(data), data[y_column]) for data in dataset['test']]
    return training_data, val_data, testing_data

# Pick n random samples from each class in the dataset
def pick_random(data, n):
    labels = {}
    for sentence, label in data:
        if label not in labels:
            labels[label] = []
        labels[label].append((sentence, label))
    result = []
    for label in labels:
        result.extend(random.sample(labels[label], n))
    random.shuffle(result)
    return result

# Load and split huggingface dataset into training, validation, and testing sets with equal class balance
def load_hf_dataset_class_split(hf_token, dataset_name, extract_x, y_column, num_classes, train_size = 0.5, val_size = 0.1):
    if hf_token is None:
        raise ValueError("Please provide a valid Hugging Face API token.")
    if (train_size + val_size) > 1:
        raise ValueError("Train and validation sizes must sum to less than 1.")
    dataset = load_dataset(dataset_name, token=hf_token)
    
    training = [(extract_x(data), data[y_column]) for data in dataset['train']]
    random.shuffle(training)
    training_data_options = training[:math.ceil(len(training)*(1-val_size*2))]
    val_data_options = training[math.ceil(len(training)*(1-val_size*2)):]

    training_count = math.ceil(len(training) * train_size)
    training_data = pick_random(training_data_options, n = math.ceil(training_count // num_classes))
    val_count = math.ceil(len(training) * val_size)
    val_data = pick_random(val_data_options, n = math.ceil(val_count // num_classes))
    testing_data = [(extract_x(data), data[y_column]) for data in dataset['test']]
    return training_data, val_data, testing_data

# Helper function for loading FOMC communication dataset
def load_fomc_communication(hf_token):
    return load_hf_dataset_class_split(
        hf_token=hf_token, dataset_name='gtfintechlab/fomc_communication',
        extract_x = lambda x : f"Sentence to classify: {x['sentence']}", y_column = 'label', 
        num_classes = 3, train_size = 0.5, val_size = 0.1
    )

# Helper function for evaluating FOMC communication responses
def eval_fomc_communication(prediction: Variable, ground_truth_answer: Variable):
    mapping = {0: 'dovish', 1: 'hawkish', 2: 'neutral'}
    pred = extract_answer(f"Extract the one word answer (HAWKISH, DOVISH, or NEUTRAL) from the following text: {str(prediction.value).lower()}.\nDo not enter any other text.")
    return int((pred != None and str(pred).lower() == str(mapping[int(ground_truth_answer.value)]).lower()))

def load_ect_sum(hf_token):
    return load_hf_dataset(
        hf_token=hf_token, dataset_name='gtfintechlab/ECTSum',
        extract_x = lambda x : f"Context to extract: {x['context']}", y_column = 'response',
        train_size = 0.5, val_size = 0.1
    )

def load_finbench(hf_token):
    return load_hf_dataset(
        hf_token=hf_token, dataset_name='gtfintechlab/finbench',
        extract_x = lambda x : f"Tabular data: {x['X_ml']}\nProfile data: {x['X_profile']}", y_column = 'y',
        train_size = 0.5, val_size = 0.1
    )

def load_finentity(hf_token):
    return load_hf_dataset(
        hf_token=hf_token, dataset_name='gtfintechlab/finentity',
        extract_x = lambda x : f"Sentence: {x['content']}", y_column = 'annotations',
        train_size = 0.5, val_size = 0.1
    )

def load_finer(hf_token):
    return load_hf_dataset(
        hf_token=hf_token, dataset_name='gtfintechlab/finer_ord_encoded',
        extract_x = lambda x : f"Sentence: {x['context']}", y_column = 'response',
        train_size = 0.5, val_size = 0.1
    )

def load_finqa(hf_token):
    return load_hf_dataset(
        hf_token=hf_token, dataset_name='gtfintechlab/finqa',
        extract_x = lambda x : f"{' '.join(x['pre_text'])} {' '.join(x['post_text'])} {' '.join([' '.join(row) for row in x['table_ori']])}\nQuestion: {x['question']}", y_column = 'answer',
        train_size = 0.5, val_size = 0.1
    )

def load_fpb(hf_token):
    # load_dataset("financial_phrasebank", "sentences_allagree")
    # no testing dataset
    return load_hf_dataset(
        hf_token=hf_token, dataset_name='gtfintechlab/financial_phrasebank',
        extract_x = lambda x : f"Sentence: {x['sentence']}", y_column = 'label',
        train_size = 0.5, val_size = 0.1
    )

def load_numclaim(hf_token):
    return load_hf_dataset(
        hf_token=hf_token, dataset_name='gtfintechlab/Numclaim',
        extract_x = lambda x : f"Sentence: {x['context']}", y_column = 'response',
        train_size = 0.5, val_size = 0.1
    )

def eval_numclaim(prediction: Variable, ground_truth_answer: Variable):
    pred = extract_answer(f"Extract the exact answer (IN CLAIM, OUT OF CLAIM) from the following text: {str(prediction.value).lower()}.\nDo not enter any other text.")
    return pred != None and (str(pred).upper().replace(' ', '') == str(ground_truth_answer))

def load_banking77(hf_token):
    return load_hf_dataset(
        hf_token=hf_token, dataset_name='gtfintechlab/banking77',
        extract_x = lambda x : f"Sentence: {x['text']}", y_column = 'label',
        train_size = 0.5, val_size = 0.1
    )

banking77_list = [] # fill in later - in prompts.py

def eval_banking77(prediction: Variable, ground_truth_answer: Variable):
    pred = extract_answer(f"Extract the answer that is an option in this list {', '.join(banking77_list)} from the following text: {str(prediction.value).lower()}.\nDo not enter any other text. Your output must be in the same format as the option in the list.")
    if (str(pred) not in banking77_list):
        raise ValueError(f"Invalid output: {pred}. Not in banking77 list.")
    return pred != None and (str(pred) == str(ground_truth_answer))

# Map task names to task-specific helpers
# Each task should have a starting prompt, constraints (if any exist), evaluation function, and dataset loading function
textgrad_task_map = {
    'fomc_communication': {
        'starting_prompt': "Classify the following sentence from FOMC into hawkish, dovish, or neutral based on its stance on monetary policy. Explain your reasoning.",
        'constraints': "Must include text instructions to output one of hawkish, dovish, and neutral, written with those exact labels and not synonyms.",
        'eval_fn': eval_fomc_communication,
        'load_dataset': load_fomc_communication
    },
    'ECT_Sum': {
        'starting_prompt': "Behave like you are an expert at summarization tasks. Below an earnings call transcript of a Russell 3000 Index company is provided. Perform extractive summarization followed by paraphrasing the transcript in bullet point format according to the experts-written short telegram-style bullet point summaries derived from corresponding Reuters articles. The target length of the summary should be at most 50 words.",
        'constraints': "Must give a bullet pointed summary list.",
        'eval_fn': None,
        'load_dataset': load_ect_sum
    }
}

# Fetch task-specific helpers based on task name
def fetch_task_specific_helpers(task_name):
    try: 
        return textgrad_task_map[task_name]
    except KeyError:
        raise ValueError(f"Task {task_name} not found! Please double check the dataset name.")