import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion 
from superflue.together_code.tokens import tokens
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def extraction_prompt(llm_response: str):
    prompt = f'''Extract the entity-level sentiment information from the following LLM response. For each identified entity, extract the start and end indices, the entity name, and the sentiment tag (‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’). Format the output as a JSON array where each entity is represented as an object with the following structure:
                {{
                    "start": start index,
                    "end": end index,
                    "value": entity name,
                    "tag": "NEGATIVE" | "POSITIVE" | "NEUTRAL"
                }}
                
                Here is the LLM response to analyze:
                "{llm_response}"'''

    return prompt

def match_entities(predicted_entities, actual_entities):
    correct_matches = 0
    total_actual = len(actual_entities)
    matched_indices = set()

    for pred in predicted_entities:
        for idx, actual in enumerate(actual_entities):
            if (
                idx not in matched_indices and
                pred['start'] == actual['start'] and
                pred['end'] == actual['end'] and
                pred['value'] == actual['value'] and
                pred['tag'].lower() == actual['tag'].lower()
            ):
                correct_matches += 1
                matched_indices.add(idx)
                break

    return correct_matches, total_actual

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def extract_and_evaluate_responses(args):
    results_file = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{args.date}.csv"
    )

    # Load the CSV file with the LLM responses
    df = pd.read_csv(results_file)
    correct_predictions = 0
    total_predictions = 0

    # Continual save path
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )

    # Initialize the columns for storing results if they don't exist
    if 'correct_predictions' not in df.columns:
        df['correct_predictions'] = 0
        df['total_predictions'] = 0

    for i, row in df.iterrows():
        if pd.notna(row['correct_predictions']) and pd.notna(row['total_predictions']):
            # Skip already processed rows
            continue
        
        llm_response = row["llm_responses"]
        actual_labels = json.loads(row['actual_labels'])  # Assuming actual labels are in JSON format in the CSV

        try:
            model_response = completion(  # type: ignore
                prompt=extraction_prompt(llm_response),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_labels = json.loads(model_response.choices[0].message.content.strip())  # type: ignore
            
            correct, total = match_entities(extracted_labels, actual_labels)
            df.at[i, 'correct_predictions'] = correct
            df.at[i, 'total_predictions'] = total

            correct_predictions += correct
            total_predictions += total

            logger.info(f"Processed {i + 1}/{len(df)} responses.") # type: ignore

            # Save progress after each row
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")

    # Final accuracy calculation
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    logger.info(f"Evaluation completed. Final Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    return df, accuracy
