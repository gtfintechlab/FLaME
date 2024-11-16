import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together
from together import Together
from evaluate import load
import numpy as np
bertscore = load("bertscore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def summarization_prompt(input_text: str):
    # Adjust the prompt to generate event-based temporal summaries from EDTSum data
    prompt = f'''Generate a temporal summary in about 50 words in line-by-line bullet format based on the following input. The summary should include key events, time points, and any major changes in sequence.
                
                Here is the input to analyze:
                "{input_text}"'''
    return prompt

def extract_and_evaluate_responses(args):
    results_file = (
        ROOT_DIR
        / "results"
        / 'edtsum'
        / 'edtsum_meta-llama'
        / "Meta-Llama-3.1-8B-Instruct-Turbo_07_10_2024.csv"
    )

    df = pd.read_csv(results_file)
    # Assuming the output column contains the expected summaries
    correct_summaries = df['actual_labels'].tolist()
    llm_responses = df['llm_responses'].tolist()
    bert_scores = bertscore.compute(predictions=llm_responses, references=correct_summaries, model_type="distilbert-base-uncased")
    print(bert_scores)

    df['precision'] = bert_scores["precision"]
    df['recall'] = bert_scores["recall"]
    df['f1'] = bert_scores["f1"]

    print(f"BERTScore Precision: {np.mean(bert_scores['precision'])}")
    print(f"BERTScore Recall: {np.mean(bert_scores['recall'])}")
    print(f"BERTScore F1: {np.mean(bert_scores['f1'])}")
    
    # Save the evaluation results
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / 'edtsum'
        / f"evaluation_{'edtsum'}_{'meta-llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(evaluation_results_path, index=False)

    eval_df = pd.DataFrame({'precision' : [np.mean(bert_scores['precision'])], 'recall' : [np.mean(bert_scores['recall'])], 'f1' : [np.mean(bert_scores['f1'])]})
    eval_df.to_csv(Path(f"{str(evaluation_results_path)[:-4]}_statistics.csv"), index=False)

    logger.info(f"Evaluation completed. Results saved to {evaluation_results_path}")
    return df

# Helper function for stop tokens
tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}

def tokens(model_name):
    return tokens_map.get(model_name, [])

if __name__ == "__main__":
    extract_and_evaluate_responses(None)