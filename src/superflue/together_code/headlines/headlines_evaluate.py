import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion 
from superflue.together_code.tokens import tokens
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT_DIR / "evaluation_results"

# Label Mapping for Headlines task
label_mapping = {
    "Price_or_Not": {"0": 0, "1": 1},
    "Direction_Up": {"0": 0, "1": 1},
    "Direction_Down": {"0": 0, "1": 1},
    "Direction_Constant": {"0": 0, "1": 1},
    "Past_Price": {"0": 0, "1": 1},
    "Future_Price": {"0": 0, "1": 1},
    "Past_News": {"0": 0, "1": 1}
}

def extraction_prompt(llm_response: str):
    """Generate a prompt to extract the relevant information from the LLM response."""
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

def map_label_to_number(label: str, category: str):
    """Map the extracted label to its corresponding numerical value."""
    normalized_label = label.strip()
    return label_mapping[category].get(normalized_label, -1)  # Return -1 if the label is not found

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")

def extract_and_evaluate_responses(args):

    # Load the CSV file with the LLM responses
    results_file = (
        ROOT_DIR
        / "results"
        / 'headlines'
        / f"headlines_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    df = pd.read_csv(results_file)

    # Path to save evaluation results continuously
    evaluation_results_path = (
        RESULTS_DIR
        / 'headlines'
        / f"headlines_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize extracted_labels if it doesn't exist in the dataframe
    if 'extracted_labels' not in df.columns:
        df['extracted_labels'] = None

    # Initialize lists to store results for metrics calculation
    extracted_labels = []
    correct_labels = df[['price_or_not', 'direction_up', 'direction_down', 'direction_constant', 'past_price', 'future_price', 'past_news']].values.tolist()

    for i, llm_response in enumerate(df["llm_responses"]):
        if pd.notna(df.at[i, 'extracted_labels']):
            # Skip already processed rows
            continue

        try:
            # Generate the prompt and get the LLM response
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(llm_response)}],
                max_tokens=50,
                temperature=0.0,
                top_p=0.9,
                repetition_penalty=1.0,
                stop=tokens(args.model)
            )
            extracted_label_json = model_response.choices[0].message.content.strip()  # type: ignore # Extract the label
            extracted_label = eval(extracted_label_json)  # Assuming the output is a JSON object

            # Map the extracted labels to numeric values
            mapped_labels = [
                map_label_to_number(str(extracted_label['Price_or_Not']), "Price_or_Not"),
                map_label_to_number(str(extracted_label['Direction_Up']), "Direction_Up"),
                map_label_to_number(str(extracted_label['Direction_Down']), "Direction_Down"),
                map_label_to_number(str(extracted_label['Direction_Constant']), "Direction_Constant"),
                map_label_to_number(str(extracted_label['Past_Price']), "Past_Price"),
                map_label_to_number(str(extracted_label['Future_Price']), "Future_Price"),
                map_label_to_number(str(extracted_label['Past_News']), "Past_News")
            ]

            df.at[i, 'extracted_labels'] = mapped_labels
            extracted_labels.append(mapped_labels)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
            
            # Save progress after each row
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append([-1] * 7)

    # Calculate metrics for each column (Price_or_Not, Direction_Up, etc.)
    correct_predictions = [list(map(int, labels)) for labels in correct_labels]
    accuracy = accuracy_score(correct_predictions, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(correct_predictions, extracted_labels, average='weighted')

    # Save final results
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}. Results saved to {evaluation_results_path}")

    return df, accuracy, precision, recall, f1

if __name__ == "__main__":
    extract_and_evaluate_responses(None)
