import together
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, r"C:\Users\mikad\Documents\GitHub\textgrad")
import textgrad
from textgrad.engine import get_engine
from textgrad import Variable
from textgrad.tasks import DataLoader 
from textgrad.optimizer import TextualGradientDescent
from tqdm import tqdm
import argparse
from textgrad_helpers import fetch_task_specific_helpers, eval_dataset, eval_sample, run_val_revert

# things unique for each dataset
# task --> hashmap mapping of all of the arguments needed
# starting_prompt -> string, starting prompt for the model
# constraints -> string, constraints for the prompt optimizer to follow
# eval_fn -> function, evaluation function for the model
# load_dataset -> function, function to load the dataset, returns train, test, and validation sets

# args: api_key, hf_token, task, model, num_epochs, batch_size, max_tokens, temperature, top_k, top_p, repetition_penalty
def textgrad_optimization(args):
    together.api_key = args.api_key

    task_helper = fetch_task_specific_helpers(args.task)
    training_data, val_data, testing_data = task_helper['load_dataset'](args.hf_token)
    starting_prompt = task_helper['starting_prompt']
    constraints = task_helper['constraints']
    print(textgrad)

    # initialize textgrad model & optimizer
    engine = get_engine(args.model)
    system_prompt = Variable(starting_prompt, requires_grad=True, role_description=f"System prompt to the language model. Constraints: {constraints}")
    model = textgrad.BlackboxLLM(engine, system_prompt, max_tokens=args.max_tokens, 
                                 temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, 
                                 repetition_penalty=args.repetition_penalty)
    optimizer = TextualGradientDescent(engine=engine, parameters=[system_prompt], constraints=[constraints],
                                 max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, 
                                 top_p=args.top_p, repetition_penalty=args.repetition_penalty)

    eval_fn = textgrad.autograd.StringBasedFunction(task_helper['eval_fn'], "Evaluate if LLM answer matches ground truth")

    # initialize output dictionary with starting prompt
    start_val_accuracy = np.mean(eval_dataset(val_data, eval_fn, model))
    results = {"val_accuracy": [start_val_accuracy], "prompt": [starting_prompt], 
                "new_prompt": [starting_prompt], "new_prompt_val_accuracy": [start_val_accuracy]}
    print(f"Starting prompt: {starting_prompt}")
    print(f"Starting val accuracy: {start_val_accuracy}")

    # evaluate on testing dataset with starting prompt
        
    # optimize system prompt
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.num_epochs):
        for step, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
            if step >= 10:
                break

            # initialize training step
            pbar.set_description(f"Epoch {epoch}: Training step {step}")
            optimizer.zero_grad()
            losses = []

            # train off items in batch
            for (x, y) in zip(batch_x, batch_y):
                eval_output = eval_sample(x, y, eval_fn, model)
                losses.append(eval_output)

            # sum losses
            try:
                total_loss = textgrad.sum(losses)
                total_loss.backward(engine)
                optimizer.step()
            except:
                print("Error in optimization step --> greater than max context window.")
                continue
            
            print("Prompt: ", system_prompt.get_value())
        
        # evaluate new system prompt
        # if worse, revert to previous prompt
        results["new_prompt"].append(system_prompt.get_value())
        print("New prompt: ", system_prompt.get_value())
        run_val_revert(system_prompt, results, eval_fn, model, val_data)
        
        print("System prompt: ", system_prompt)
        results["prompt"].append(system_prompt.get_value())
    
    # evaluate on testing dataset with final prompt
    return results

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Optimize prompt on TogetherAI over the SuperFLUE dataset"
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--task", type=str, help="Task to use")
    parser.add_argument("--api_key", type=str, help="API key to use")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token to use")
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default = 5, help='Batch size')
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature to use")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p to use")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k to use")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty to use")
    return parser.parse_args()

def fomc_example():
    
    args = {
        "api_key": together_token,
        "hf_token": hf_token,
        "task": "fomc_communication",
        "model": "together-meta-llama/Meta-Llama-3-8b-Instruct-Turbo",
        "max_tokens": 128,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.7,
        "repetition_penalty": 1.1,
        "num_epochs": 3,
        "batch_size": 5
    }

    args = argparse.Namespace(**args)
    return textgrad_optimization(args)

if __name__ == "__main__":
    results = fomc_example()
    pd.DataFrame(results).to_csv("generalized_fomc_test.csv", index=False)