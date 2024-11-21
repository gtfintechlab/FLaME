import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together
from together import Together
import warnings
import argparse
import re
warnings.filterwarnings("ignore")
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# Logging configuration
RESULTS_DIR = ROOT_DIR / "results"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FiQA Sentiment Model")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--api_key", type=str, required=True, help="API Key")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--date", type=str, help="Date for the results file")
    return parser.parse_args()

# Function to create the extraction prompt
def extraction_prompt(llm_response: str):
    prompt = f"""
    You are tasked with extracting the sentiment score from a response. 
    The sentiment score should be a single numeric value between -1 and 1.

    Model Response: {llm_response}

    Provide only the numerical sentiment score as the output.
    """
    return prompt

def extract_numerical_value(text):
    match = re.search(r"(-?\d+\.\d+)", text)  # Adjusted to capture decimal values
    return float(match.group(0)) if match else None

def extract_and_save_responses(args):
    together.api_key = args.api_key
    results_file = (
        "/Users/yangyang/Desktop/SuperFLUE/results/fiqa1/fiqa1_meta-llama/fiqa_task1_llama-3.1-8b_11_11_2024.csv"
    )

    try:
        df = pd.read_csv(results_file)
        logger.info("CSV file loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return

    evaluation_results_path = (
        RESULTS_DIR
        / "evaluation_results"
        / 'fiqa'
        / f"evaluation_fiqa_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    extraction_response = [None] * len(df)  # Initialize to match df length
    extraction_model_response = [None] * len(df)
    regex_extraction = [None] * len(df)

    client = Together()
    
    for i, entry in enumerate(df["llm_responses"]):
        try:
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(entry)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model)
            )
            response_text = model_response.choices[0].message.content  # type: ignore
            print(response_text)
            extraction_response[i] = response_text
            numerical_value = extract_numerical_value(response_text)
            regex_extraction[i] = numerical_value

        except Exception as e:
            extraction_model_response[i] = str(e)

    df['extraction_model_response'] = extraction_model_response
    df['extraction_response'] = extraction_response
    df['regex_extraction'] = regex_extraction    

    correct_labels = df['actual_sentiment'].tolist()    
    correct_predictions = sum(1 for x, y in zip(correct_labels, regex_extraction) if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0


    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    df.to_csv(evaluation_results_path, index=False)
    return df, accuracy

tokens_map = {"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])


if __name__ == "__main__":
    args = parse_args()  
    extract_and_save_responses(args)
