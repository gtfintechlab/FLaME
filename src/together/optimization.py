import together
import numpy as np
from datasets import load_dataset
from tokens import tokens
import os
import textgrad
from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.optimizer import TextualGradientDescent
from textgrad.tasks import DataLoader 
import concurrent
from tqdm import tqdm
import argparse

# args: api_key, hf_token, dataset_name, model, starting_prompt, num_epochs, batch_size, eval_fn, max_tokens, temperature, top_k, top_p, repetition_penalty
def textgrad_opt_classification(args):
    together.api_key = args.api_key
    dataset = load_dataset(args.dataset_name, token=args.hf_token)

    # initialize textgrad model & optimizer
    engine = get_engine(args.model)
    system_prompt = Variable(args.starting_prompt, requires_grad=True, role_description="system prompt to the language model")
    model = textgrad.BlackboxLLM(engine, system_prompt, max_tokens=args.max_tokens, 
                                 temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, 
                                 repetition_penalty=args.repetition_penalty)
    optimizer = TextualGradientDescent(engine=engine, parameters=[system_prompt], 
                                 max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, 
                                 top_p=args.top_p, repetition_penalty=args.repetition_penalty)

    # extract training and testing data
    training_data = [(data['sentence'], data['label']) for data in dataset['train']]
    testing_data = [(data['sentence'], data['label']) for data in dataset['test']]

    # initialize output dictionary with starting prompt
    start_test_accuracy = np.mean(eval_dataset(testing_data, args.eval_fn, model))
    results = {"test_accuracy": [start_test_accuracy], "prompt": [args.starting_prompt], 
               "new_prompt": [args.starting_prompt], "new_prompt_test_accuracy": [start_test_accuracy]}
    print(f"Starting prompt: {args.starting_prompt}")
    print(f"Starting test accuracy: {start_test_accuracy}")
    
    # optimize system prompt
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.num_epochs):
        for step, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            # initialize training step
            pbar.set_description(f"Epoch {epoch}: Training step {step}")
            optimizer.zero_grad()
            losses = []

            # train off items in batch
            for (x, y) in zip(batch_x, batch_y):
                eval_output = eval_sample(x, y, args.eval_fn, model)
                losses.append(eval_output)

            # sum losses
            total_loss = textgrad.sum(losses)
            total_loss.backward(engine)
            optimizer.step()
            
            # evaluate new system prompt
            # if worse, revert to previous prompt
            results["new_prompt"].append(system_prompt.get_value())
            print("New prompt: ", system_prompt.get_value())
            run_test_revert(system_prompt, results, args.eval_fn, model, testing_data)
            
            print("Current system prompt: ", system_prompt)
            results["prompt"].append(system_prompt.get_value())
            
            if step >= 5:
                break

    return results

# Evaluate given sample using custom evaluation function
def eval_sample(x, y, eval_fn, model):
    x = Variable(x, requires_grad=False, role_description="query to the language model")
    y = Variable(y, requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    inputs = dict(prediction = response, ground_truth_answer = y)
    eval_output_variable = eval_fn(inputs = inputs)
    return eval_output_variable
    
# Evaluate dataset using custom evaluation function
def eval_dataset(test_set, eval_fn, model, max_samples = None, max_workers = 16):
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
def run_test_revert(system_prompt, results, eval_fn, model, testing_dataset):
    curr_accuracy = np.mean(eval_dataset(testing_dataset, eval_fn, model))
    prev_accuracy = np.mean(results["test_accuracy"][-1])
    print(f"Test accuracy: {curr_accuracy}\nPrevious test accuracy: {prev_accuracy}")
    previous_prompt = results["prompt"][-1]

    results["new_prompt_test_accuracy"].append(curr_accuracy)
    
    if curr_accuracy < prev_accuracy:
        print(f"Rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        curr_accuracy = prev_accuracy

    results["test_accuracy"].append(curr_accuracy)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Optimize prompt on TogetherAI over the SuperFLUE dataset"
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--task", type=str, help="Task to use")
    parser.add_argument("--api_key", type=str, help="API key to use")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token to use")
    parser.add_argument('--starting_prompt', type=str, help='Starting prompt')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature to use")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p to use")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k to use")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty to use")
    return parser.parse_args()

def fomc_example():
    def find_first_occurrence(text):
        words = ["hawkish", "dovish", "neutral"]
        positions = [(text.find(word), word) for word in words if text.find(word) != -1]
        return min(positions)[1] if positions else None

    def string_based_equality_fn(prediction: Variable, ground_truth_answer: Variable):
        mapping = {0: 'dovish', 1: 'hawkish', 2: 'neutral'}
        pred = find_first_occurrence(str(prediction.value).lower())
        if (pred != None and str(pred).lower() == str(mapping[int(ground_truth_answer.value)]).lower()):
            return 1
        else: 
            return 0

    eval_fn = textgrad.autograd.StringBasedFunction(string_based_equality_fn, "Evaluate if LLM answer matches ground truth")
    args = {
        "api_key": together_token,
        "hf_token": hf_token,
        "dataset_name": "gtfintechlab/fomc_communication",
        "model": "together-mistralai/Mistral-7B-Instruct-v0.3",
        "starting_prompt": "Classify the following sentence from FOMC into hawkish, dovish, or neutral based on its stance on monetary policy. Output your prediction on the first line and your explanation after.",
        "num_epochs": 3,
        "batch_size": 5,
        "eval_fn": eval_fn
    }

    args = argparse.Namespace(**args)
    textgrad_opt_classification(args)

if __name__ == "__main__":
    fomc_example()
