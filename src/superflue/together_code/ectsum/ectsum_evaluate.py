import pandas as pd
import logging
from datetime import date
from pathlib import Path
from evaluate import load
import numpy as np
bertscore = load("bertscore")
from superflue.config import ROOT_DIR
from superflue.utils.logging_utils import setup_logger


logger = setup_logger(
    name="ectsum_evaluate",
    log_file=Path("logs/ectsum_evaluate.log"),
    level=logging.INFO,
)

def summarization_prompt(input_text: str):
    # Adjust the prompt to generate summaries for ECT data
    prompt = f'''Generate a financial summary in about 50 words in line-by-line format based on the following input. The summary should include key financial information such as earnings per share, revenue, and other significant figures.
                It should contain only lower case letters and numbers (including decimals). Do not include any special characters other than \n, % or $.
                Here is the input to analyze:
                "{input_text}"'''
    return prompt

def extract_and_evaluate_responses(args):
    results_file = (
        ROOT_DIR
        / "results"
        / 'ectsum'
        / 'ectsum_meta-llama'
        / "Meta-Llama-3.1-8B-Instruct-Turbo_02_10_2024.csv"
    )

    df = pd.read_csv(results_file)
    # Assuming the output column contains the expected summaries
    correct_summaries = df['actual_labels'].tolist()
    llm_responses = df['llm_responses'].tolist()
    bert_scores = bertscore.compute(predictions=llm_responses, references=correct_summaries, model_type="distilbert-base-uncased")
    print(bert_scores)

    df['precision'] = bert_scores["precision"] # type: ignore
    df['recall'] = bert_scores["recall"] # type: ignore
    df['f1'] = bert_scores["f1"] # type: ignore

    logger.info(f"BERTScore Precision: {np.mean(bert_scores['precision'])}") # type: ignore
    logger.info(f"BERTScore Recall: {np.mean(bert_scores['recall'])}") # type: ignore
    logger.info(f"BERTScore F1: {np.mean(bert_scores['f1'])}") # type: ignore
    
    # Save the evaluation results
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / 'ectsum'
        / f"evaluation_{'ectsum'}_{'meta-llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(evaluation_results_path, index=False)

    eval_df = pd.DataFrame({'precision' : [np.mean(bert_scores['precision'])], 'recall' : [np.mean(bert_scores['recall'])], 'f1' : [np.mean(bert_scores['f1'])]}) # type: ignore
    eval_df.to_csv(Path(f"{str(evaluation_results_path)[:-4]}_statistics.csv"), index=False)

    logger.info(f"Evaluation completed. Results saved to {evaluation_results_path}")
    return df

# Helper function for stop tokens
tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}

def tokens(model_name):
    return tokens_map.get(model_name, [])

if __name__ == "__main__":
    extract_and_evaluate_responses(None)