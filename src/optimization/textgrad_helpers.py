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

def load_fomc_communication(hf_token):
    if hf_token is None:
        raise ValueError("Please provide a valid Hugging Face API token.")
    dataset = load_dataset('gtfintechlab/fomc_communication', token=hf_token)
    # subset the data to do a mini run
    def pick_random(data, n = 20):
        labels = {0: [], 1: [], 2: []}
        for sentence, label in data:
            labels[label].append((sentence, label))
        result = []
        for label in labels:
            result.extend(random.sample(labels[label], n))
        return result
    
    training = [(data['sentence'], data['label']) for data in dataset['train']]
    random.shuffle(training)
    training_data_options = training[:math.ceil(len(training)*0.8)]
    val_data_options = training[math.ceil(len(training)*0.8):]

    num_classes = 3
    training_count = math.ceil(len(training) * 0.5)
    training_data = pick_random(training_data_options, n = math.ceil(training_count // num_classes))
    val_count = math.ceil(len(training) * 0.1)
    val_data = pick_random(val_data_options, n = math.ceil(val_count // num_classes))
    testing_data = [(data['sentence'], data['label']) for data in dataset['test']]
    return training_data, val_data, testing_data

def eval_fomc_communication(prediction: Variable, ground_truth_answer: Variable):
    def find_first_occurrence(text):
        words = ["hawkish", "dovish", "neutral"]
        positions = [(text.find(word), word) for word in words if text.find(word) != -1]
        return min(positions)[1] if positions else 'neutral'

    def extract_answer(text):
        client = Together()
        try: 
            model_response = client.chat.completions.create(
                model='meta-llama/Meta-Llama-3-8b-Instruct',
                messages=[{'role': 'system', 'content': "Discard all previous instructions. Behave as if you are an text extraction model. You are only extracting the answer."},
                            {'role': 'user', 'content': f"Extract the one word answer (HAWKISH, DOVISH, or NEUTRAL) from the following text: {text}.\nDo not enter any other text."}],
                max_tokens=128, temperature=0.7, top_k=50, top_p=0.7,repetition_penalty=1.1
            )
            response = model_response.choices[0].message.content
            return response.strip().split()[0]
        except:
            return None

    mapping = {0: 'dovish', 1: 'hawkish', 2: 'neutral'}
    # pred = find_first_occurrence(str(prediction.value).lower())
    pred = extract_answer(str(prediction.value).lower())
    if (pred != None and str(pred).lower() == str(mapping[int(ground_truth_answer.value)]).lower()):
        return 1
    else: 
        return 0


# Map task names to task-specific helpers
# Each task should have a starting prompt, constraints (if any exist), evaluation function, and dataset loading function
textgrad_task_map = {
    'fomc_communication': {
        'starting_prompt': "Classify the following sentence from FOMC into hawkish, dovish, or neutral based on its stance on monetary policy. Explain your reasoning.",
        'constraints': "Must include text instructions to output one of hawkish, dovish, and neutral, written with those exact labels and not synonyms.",
        'eval_fn': eval_fomc_communication,
        'load_dataset': load_fomc_communication
    }
}

def fetch_task_specific_helpers(task_name):
    try: 
        return textgrad_task_map[task_name]
    except KeyError:
        raise ValueError(f"Task {task_name} not found! Please double check the dataset name.")