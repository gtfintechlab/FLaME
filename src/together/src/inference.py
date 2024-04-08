import argparse
from time import time
from datetime import date
import pandas as pd
import together
import task_specific_inference
from fpb.fpb_inference import fpb_inference
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# from utils.prompt_generator import (
#     numclaim_prompt,
#     fomc_prompt,
#     finer_prompt,
#     fpb_prompt,
#     finentity_prompt,
#     finqa_prompt,
#     ectsum_prompt,
#     banking77_prompt,
#     convfinqa_prompt,
# )

today = date.today() 

# task_prompt_map = {
#     "numclaim": numclaim_prompt,
#     "fomc": fomc_prompt,
#     "finer": finer_prompt,
#     "fpb": fpb_prompt,
#     "finentity": finentity_prompt,
#     "finqa": finqa_prompt,
#     "ectsum": ectsum_prompt,
#     "banking77": banking77_prompt,
#     "convfinqa": convfinqa_prompt,
# }

task_generation_map = {
    "fpb": fpb_inference
    # "numclaim": numclaim_inference,
    # "fomc": fomc_inference,
    # "finer": finer_inference,
    # "finentity": finentity_inference,
    # "finqa": finqa_inference,
    # "ectsum": ectsum_inference,
    # "banking77": banking77_inference,
    # "convfinqa": convfinqa_inference,
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a LLM on TogetherAI over the SuperFLUE dataset"
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--task", type=str, help="Task to use")
    parser.add_argument("--api_key", type=str, help="API key to use")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token to use")
    parser.add_argument("--max_tokens", type=int, help="Max tokens to use, default = 128")
    parser.add_argument("--temperature", type=float, help="Temperature to use")
    parser.add_argument('--top_p', type=float, help='Top-p to use, default = 0.7')
    parser.add_argument("--top_k", type=int, help="Top-k to use, default = 50")
    parser.add_argument('--repetition_penalty', type=float, help='Repetition penalty to use, default = 1.1')
    args = parser.parse_args()
    return args

def evaluate(df):
    # Evaluating metrics for the train split
    accuracy = accuracy_score(df['actual_labels'], df['llm_first_word_responses'])
    precision = precision_score(df['actual_labels'], df['llm_first_word_responses'], average='micro')
    recall = recall_score(df['actual_labels'], df['llm_first_word_responses'], average='micro')
    f1 = f1_score(df['actual_labels'], df['llm_first_word_responses'], average='micro')
    # roc_auc = roc_auc_score(actual_labels, llm_first_word_responses) # Uncomment if applicable

    # Creating DataFrames for metrics
    metrics = pd.DataFrame({'accuracy': [accuracy],
                            'precision': [precision],
                            'recall': [recall],
                            'f1_score': [f1]})
                            #'roc_auc': [roc_auc]}) # Uncomment if applicable

    return metrics


def main():
    args = parse_arguments()
    inference_function = task_generation_map[args.task]
    start_t = time()
    df = inference_function(args)
    time_taken = time() - start_t
    df.to_csv(
        f'../{args.task}/results/{args.model}/{args.task}_{args.model}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False
    )
    # TODO: CALCULATE METRICS USING DF
    # model, task = args.model, args.task
    
    # df_metrics = evaluate(df)
    #  df.to_csv(
    #      f'../{args.task}/results/{args.model}/metrics_{args.task}_{args.model}_{today.strftime("%d_%m_%Y")}_{time_taken}.csv', index=False
    #)


if __name__ == "__main__":
    main()
