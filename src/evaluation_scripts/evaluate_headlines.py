import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def extraction_prompt(llm_response: str):
    prompt = f'''Extract the relevant information from the following LLM response and provide a score of 0 or 1 for each attribute based on the content. Format your output as a JSON object with the following keys:
                    - "Price_or_Not": 0 if the response does not talk about price, 1 if it does.
                    - "Direction_Up": 0 if the response does not indicate the price going up, 1 if it does.
                    - "Direction_Down": 0 if the response does not indicate the price going down, 1 if it does.
                    - "Direction_Constant": 0 if the response does not mention the price remaining constant, 1 if it does.
                    - "Past_Price": 0 if the response does not refer to a past event related to price, 1 if it does.
                    - "Future_Price": 0 if the response does not refer to a future event related to price, 1 if it does.
                    - "Past_News": 0 if the response does not mention a general past event (unrelated to price), 1 if it does.
                    
                Here is the LLM response to analyze:
                "{llm_response}"'''
    return prompt

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def extract_and_evaluate_responses(args):
    together.api_key = args.api_key  # type: ignore
    results_file = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{args.date}.csv"
    )

    # Load the CSV file with the LLM responses
    df = pd.read_csv(results_file)
    extracted_labels = []
    correct_labels = df['actual_labels'].tolist()

    # Path to save evaluation results continuously
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )

    # Initialize the columns for storing results if they don't exist
    if 'extracted_labels' not in df.columns:
        df['extracted_labels'] = None

    for i, llm_response in enumerate(df["llm_responses"]):
        if pd.notna(df.at[i, 'extracted_labels']):
            # Skip already processed rows
            continue

        try:
            model_response = together.Complete.create(  # type: ignore
                prompt=extraction_prompt(llm_response),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = model_response["output"]["choices"][0]["text"].strip()  # type: ignore
            df.at[i, 'extracted_labels'] = extracted_label
            extracted_labels.append(extracted_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
            
            # Save progress after each row
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(None)

    # Evaluate the performance
    correct_predictions = sum(1 for x, y in zip(correct_labels, extracted_labels) if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    return df, accuracy

tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])

