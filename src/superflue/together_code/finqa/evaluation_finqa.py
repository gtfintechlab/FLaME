import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together
import warnings
import argparse

warnings.filterwarnings("ignore")
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# Logging configuration
RESULTS_DIR = ROOT_DIR / "results"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FinQA Model")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--api_key", type=str, required=True, help="API Key")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--max_tokens", type=int, default=10, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--date", type=str, help="Date for the results file")
    return parser.parse_args()

def evaluation_prompt(llm_response: str, actual_answer: str, context: str):
    prompt = f"""
    Context: {context}
    
    The correct answer is {actual_answer}. Based on the model's response, extract the numerical value closest to the correct answer.
    Return only the number. For example, 13 or 90%.

    Model Response: {llm_response}
    """
    return prompt

# Main function for evaluation
def extract_and_evaluate_responses(args):
    print("Script started")
    together.api_key = args.api_key  # type: ignore
    results_file = (
       "/Users/yangyang/Desktop/SuperFLUE/results/finqa/finqa_meta-llama/finqa_llama-3.1-8b_01_10_2024.csv"
    )

    try:
        df = pd.read_csv(results_file)
        print("CSV file loaded successfully")
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return

    # Initialize evaluation_results with None for each row in the DataFrame
    evaluation_results = [None] * len(df)
    correct_labels = df['actual_label'].tolist()

    # Define the output path here, so it remains constant across iterations
    evaluation_results_path = (
        RESULTS_DIR
        / "evaluation_results"
        / 'finqa'
        / f"evaluation_{'finqa'}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    for i, (llm_response, actual_answer, context) in enumerate(zip(df["response"], df["actual_label"], df["context"])):
        try:
            model_response = together.Complete.create(  # type: ignore
                prompt=evaluation_prompt(llm_response, actual_answer, context),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_text = model_response["choices"][0]["text"]  # type: ignore
           
            evaluation_results[i] = extracted_text  # Store the raw extracted number as is
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            evaluation_results[i] = None  # Set None in case of error

        # Update 'extracted_labels' column after each iteration
        df['extracted_labels'] = evaluation_results

        # Save the updated DataFrame to CSV after each iteration
        df.to_csv(evaluation_results_path, index=False)
        logger.info(f"CSV updated at iteration {i + 1}/{len(df)}")

    # Calculate accuracy after all iterations
    correct_predictions = sum(1 for x, y in zip(correct_labels, evaluation_results) if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    print(f"Results saved to: {evaluation_results_path}")
    return df, accuracy

# Token function to retrieve stop tokens
tokens_map = {"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])

# Main execution
if __name__ == "__main__":
    args = parse_args()  # Get command-line arguments
    extract_and_evaluate_responses(args)  # Run the evaluation
