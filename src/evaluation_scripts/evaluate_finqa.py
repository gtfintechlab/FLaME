import pandas as pd

import logging
from datetime import date
from pathlib import Path
import together
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def evaluation_prompt(llm_response: str):
    prompt = f"""
    Extract the numerical value from the model response. Only return the number itself.

    Response: {llm_response}
    """
    return prompt

def extract_and_evaluate_responses(args):
    print("Script started")
    together.api_key = args.api_key  # type: ignore
    results_file = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{args.date}.csv"
    )

    try:
        df = pd.read_csv(results_file)
        print("CSV file loaded successfully")
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return
    
    evaluation_results = []
    correct_labels = df['actual_label'].tolist()

    for i, (llm_response, actual_answer) in enumerate(zip(df["response"], df["actual_label"])):
        try:
            model_response = together.Complete.create(  # type: ignore
                prompt=evaluation_prompt(llm_response),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            evaluation_result = model_response["output"]["choices"][0]["text"].strip()  # type: ignore
            evaluation_results.append(evaluation_result)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            evaluation_results.append(None)

    df['extracted_labels'] = evaluation_results

    correct_predictions = sum(1 for x, y in zip(correct_labels, evaluation_results) if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions

    # Print the accuracy
    print(f"Accuracy: {accuracy:.4f}")

    # Save the evaluation results
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )

    # Create the directory if it does not exist
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(evaluation_results_path, index=False)

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    print(f"Results saved to: {evaluation_results_path}")
    return df, accuracy

tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])
