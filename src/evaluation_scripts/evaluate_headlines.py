import re
import pandas as pd
import json
import logging
from datetime import date
from pathlib import Path
from litellm import completion  
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
    #together.api_key = args.api_key  # type: ignore
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

    for i, llm_response in enumerate(df["llm_responses"][:1]):
        if pd.notna(df.at[i, 'extracted_labels']):
            continue  # Skip already processed rows

        try:
            model_response = completion(  # type: ignore
                messages=[{"role": "user", "content": extraction_prompt(llm_response)}],
                model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                max_tokens=128,
                temperature=0.7,
                top_k=50,
                top_p=0.7,
                repetition_penalty=1.1,
                stop=tokens("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
            )
            raw_response = model_response.choices[0].message.content.strip()  # type: ignore
            # print(raw_response)
            cleaned_response = extract_json_from_response(raw_response)
            #print(cleaned_response)
            extracted_label = json.loads(cleaned_response)  # type: ignore

            # Standardize key names
            extracted_label = {k.lower(): v for k, v in extracted_label.items()}
            correct_label = {k.lower(): v for k, v in correct_labels[i].items()}  # type: ignore

            # Calculate score for the current sample
            score = sum(1 for k in correct_label if extracted_label.get(k) == correct_label[k]) / 7
            scores.append(score)
            df.at[i, 'extracted_labels'] = json.dumps(extracted_label)

            logger.info(f"Processed {i + 1}/{len(df)} responses.")
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            scores.append(0) 

    # Evaluate overall performance
    accuracy = sum(scores) / len(scores)
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
    return df, accuracy

if __name__ == "__main__":
    extract_and_evaluate_responses(None)