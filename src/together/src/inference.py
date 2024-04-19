import argparse
import re
import numpy as np
import pandas as pd
from time import time
from datetime import date
# from tasks_inferences import fpb_inference, fomc_inference, numclaim_inference
# from numclaim.numclaim_inference import numclaim_inference
from fpb.fpb_inference import fpb_inference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# from task_specific_inference import numclaim_inference

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a LLM on TogetherAI over the SuperFLUE dataset")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--task", type=str, help="Task to use")
    parser.add_argument("--api_key", type=str, help="API key to use")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token to use")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature to use")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p to use")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k to use")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty to use")
    parser.add_argument("--prompt_format", type=str, default="superflue", help="Version of the prompt to use")
    return parser.parse_args()

def extract_label(text, label_regex):
    match = re.search(label_regex, text)
    return match.group(1) if match else 'None'

def evaluate(file, response_column, label_regex, label_mapping):
    data = pd.read_csv(file, index_col=0)
    data['extracted_label'] = data[response_column].apply(lambda x: extract_label(x, label_regex))
    data['extracted_label_numeric'] = data['extracted_label'].map(label_mapping)
    data = data.dropna(subset=['extracted_label_numeric'])

    metrics = {
        'accuracy': accuracy_score(data['actual_label'], data['extracted_label_numeric']),
        'precision': precision_score(data['actual_label'], data['extracted_label_numeric'], average='micro'),
        'recall': recall_score(data['actual_label'], data['extracted_label_numeric'], average='micro'),
        'f1_score': f1_score(data['actual_label'], data['extracted_label_numeric'], average='micro')
    }

    output_file_path = file.replace('.csv', '_metrics.csv')
    pd.DataFrame([metrics]).to_csv(output_file_path, index=False)
    return metrics

def main():
    args = parse_arguments()
    task = args.task.strip('“”"')

    task_label_mapping = {
        'fpb': {'POSITIVE': 2, 'NEGATIVE': 0, 'NEUTRAL': 1, 'None': np.nan},
        'fomc': {'DOVISH': 0, 'HAWKISH': 1, 'NEUTRAL': 2, 'None': np.nan},
        'numclaim': {'OUTOFCLAIM': 0, 'INCLAIM': 1, 'None': np.nan}
    }
    
    task_regex = {
        'fpb': r'Label: (?i)(POSITIVE|NEGATIVE|NEUTRAL)',
        'fomc': r'Label: (?i)(DOVISH|HAWKISH|NEUTRAL)',
        'numclaim': r'Label: (?i)(OUTOFCLAIM|INCLAIM)'
    }

    task_inference_map = {
        # 'numclaim': numclaim_inference,
        'fpb': fpb_inference
        # 'fomc': fomc_inference
    }

    if task in task_inference_map:
        start_t = time()
        inference_function = task_inference_map[task]
        df = inference_function(args)
        time_taken = time() - start_t
        df.to_csv(f'../{task}/results/{args.model}/{task}_{args.model}_{date.today().strftime("%d_%m_%Y")}_{time_taken}.csv', index=False)
        
        metrics = evaluate(df, 'response', task_regex[task], task_label_mapping[task])
        print(metrics)

    else:
        print(f"Task '{task}' not found in the task generation map.")

if __name__ == "__main__":
    main()
