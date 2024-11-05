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

RESULTS_DIR = ROOT_DIR / "results"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvFinQA Model")
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

# Prompt template for extracting numerical answers
def extraction_prompt(llm_response: str):
    prompt = f"""
    You will receive a response from a language model that may include a numerical answer within its text. 
    Your task is to extract and return only the main numerical value (integer, decimal, or percentage) that 
    represents the final answer. Do not include any additional text or formatting. 

    Model Response: {llm_response}

    Please respond with only one numerical value.
    """
    return prompt

# Function to extract numerical values using regex
def extract_numerical_value(text):
    match = re.search(r"(\d+(\.\d+)?%?)", text)
    return match.group(0) if match else None

# Main evaluation function
def extract_and_save_responses(args):
    together.api_key = args.api_key
    results_file = (
        "/Users/yangyang/Desktop/SuperFLUE/results/convfinqa/convfinqa_meta-llama/convfinqa_llama-3.1-8b_04_11_2024.csv"
    )

    try:
        df = pd.read_csv(results_file)
        logger.info("CSV file loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return

    # Output path for evaluation results
    evaluation_results_path = (
        RESULTS_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    extraction_response = []
    client = Together()
    extraction_model_response = []
    regex_extraction = []

    # Iterating over responses
    for entry in df["response"]:
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
            
            extraction_model_response.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore

            print(response_text)
            extraction_response.append(response_text)
         
            numerical_value = extract_numerical_value(response_text)
            regex_extraction.append(numerical_value)

        except Exception as e:
            logger.error(f"Error processing response: {e}")
            extraction_model_response.append(str(e))
            extraction_response.append(None)
            regex_extraction.append(None)

    # Adding results to DataFrame
    df['extraction_model_response'] = extraction_model_response
    df['extraction_response'] = extraction_response
    df['regex_extraction'] = regex_extraction    

    # Accuracy calculation
    correct_labels = df['actual_label'].tolist()    
    valid_predictions = [
    (x, y) for x, y in zip(correct_labels, regex_extraction) if pd.notna(x)
]
    
    correct_predictions = sum(1 for x, y in valid_predictions if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    df.to_csv(evaluation_results_path, index=False)
    return df, accuracy

# Stop tokens based on model
tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])

# Main execution
if __name__ == "__main__":
    args = parse_args()  
    extract_and_save_responses(args)
