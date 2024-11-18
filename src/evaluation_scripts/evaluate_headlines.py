import re
import pandas as pd
import json
import logging
from datetime import date
from pathlib import Path
from litellm import completion  # type: ignore
from superflue.together_code.tokens import tokens # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def extraction_prompt(llm_response: str):
    prompt = f'''Extract the relevant information from the following LLM response and provide a score of 0 or 1 for each attribute based on the content. Format your output as a JSON object with the following keys. Make sure all the keys mentioned below are included:
                    - "price_or_not": 0 if the response does not talk about price, 1 if it does.
                    - "direction_up": 0 if the response does not indicate the price going up, 1 if it does.
                    - "direction_down": 0 if the response does not indicate the price going down, 1 if it does.
                    - "direction_constant": 0 if the response does not mention the price remaining constant, 1 if it does.
                    - "past_price": 0 if the response does not refer to a past event related to price, 1 if it does.
                    - "future_price": 0 if the response does not refer to a future event related to price, 1 if it does.
                    - "past_news": 0 if the response does not mention a general past event (unrelated to price), 1 if it does.
                    
                Here is the LLM response to analyze:
                "{llm_response}"'''
    return prompt

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def extract_json_from_response(raw_response):
    try:
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        
        if match:
            json_str = match.group(0)  
            return json_str
        else:
            logging.error("No valid JSON found in the response.")
            return None 
    
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return None 

def extract_and_evaluate_responses(args):
    together.api_key = args.api_key  # type: ignore
    results_file = (
        ROOT_DIR
        / "results"
        / "headlines"
        / "headlines_meta-llama"
        / "Meta-Llama-3.1-8B-Instruct-Turbo_07_10_2024.csv"
    )

    # Load the CSV file with the LLM responses
    df = pd.read_csv(results_file)
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / "headlines"
        / f"evaluation_headlines_meta-llama-3.1-8b_{date.today().strftime('%d_%m_%Y')}.csv"
    )

    # Prepare for evaluation
    if 'extracted_labels' not in df.columns:
        df['extracted_labels'] = None

    #print("here is the column list for df",df.columns)
    correct_labels = df[[
        "price_or_not",
        "direction_up",
        "direction_down",
        "direction_constant",
        "past_price",
        "future_price",
        "past_news"
    ]].to_dict('records')

    scores = []

    for i, llm_response in enumerate(df["llm_responses"]):
        if pd.notna(df.at[i, 'extracted_labels']):
            continue  # Skip already processed rows

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
            extracted_label.append(extracted_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_label.append(None)

    # Evaluate the performance
    correct_predictions = sum(1 for x, y in zip(correct_labels, extracted_label) if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    return df, accuracy

tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])

