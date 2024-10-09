import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together
from together import Together
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "evaluation_results"

# Define label mapping for relationships in FinRED
label_mapping = {
    "HAS_RELATION": 1,
    "NO_RELATION": 0
}

def extraction_prompt(finred_text: str):
    """Generate a prompt to extract relationships from the FinRED dataset."""
    prompt = f'''Extract the relevant entities and their relationships from the following text. 
    Format your output as a list of tuples, where each tuple contains:
    (entity_1, entity_2, relationship).
    
    Here is the text to analyze:
    "{finred_text}"'''
    return prompt

def map_label_to_number(label: str):
    """Map the extracted relationship label to its corresponding numerical value."""
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(normalized_label, -1)  # Return -1 if the label is not found

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def extract_and_evaluate_responses(args):
    client = Together()

    # Load the CSV file with the sentences and LLM responses
    results_file = (
        ROOT_DIR
        / "results"
        / "finred"
        / f"finred_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    df = pd.read_csv(results_file)
    extracted_labels = []
    correct_labels = df['actual_labels'].tolist()

    # Path to save evaluation results continuously
    evaluation_results_path = (
        RESULTS_DIR
        / 'finred'
        / f"finred_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the columns for storing results if they don't exist
    if 'extracted_labels' not in df.columns:
        df['extracted_labels'] = None

    for i, finred_text in enumerate(df["sentences"]):
        if pd.notna(df.at[i, 'extracted_labels']):
            # Skip already processed rows
            continue

        try:
            # Generate a prompt to extract relationships from financial text
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(finred_text)}],
                max_tokens=10,
                temperature=0.0,
                top_p=0.9,
                repetition_penalty=1.0,
                stop=tokens(args.model)
            )
            extracted_label = model_response.choices[0].message.content.strip()  # type: ignore # Extract the label
            mapped_label = map_label_to_number(extracted_label)

            if mapped_label == -1:
                logger.error(f"Error processing response {i}: {finred_text}")
            df.at[i, 'extracted_labels'] = mapped_label
            extracted_labels.append(mapped_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
            
            # Save progress after each row
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(-1)

    # Evaluate the performance using sklearn metrics
    correct_predictions = [map_label_to_number(label) for label in correct_labels]
    accuracy = accuracy_score(correct_predictions, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(correct_predictions, extracted_labels, average='weighted')

    # Save final results
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}. Results saved to {evaluation_results_path}")

    return df, accuracy, precision, recall, f1

# Token stop mappings based on model
tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])

if __name__ == "__main__":
    extract_and_evaluate_responses(None)