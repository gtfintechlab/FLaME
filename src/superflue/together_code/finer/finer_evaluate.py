import pandas as pd
import logging
from datetime import date
from pathlib import Path
import together
import json
from together import Together
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from superflue.config import RESULTS_DIR, ROOT_DIR
from superflue.utils.logging_utils import setup_logger
import os
import re

# Initialize Together client
client = Together()
logger = setup_logger(
    name="finer_evaluate",
    log_file=Path("logs/finer_evaluate.log"),
    level=logging.INFO,
)

INPUT_FILE_PATH = os.path.join(RESULTS_DIR, "finer", "finer_meta-llama", "Meta-Llama-3.1-8B-Instruct-Turbo_02_10_2024.csv")
df = pd.read_csv(INPUT_FILE_PATH)

def extraction_prompt_finer(llm_response: str):
    prompt = f"""For each token in the following response, map the named entity labels to these numeric values:
                    - "O" (Other): 0
                    - "PER_B" (Person_B): 1
                    - "PER_I" (Person_I): 2
                    - "LOC_B" (Location_B): 3
                    - "LOC_I" (Location_I): 4
                    - "ORG_B" (Organisation_B): 5
                    - "ORG_I" (Organisation_I): 6

                Provide only the list of integer labels, in the format:
                [0, 1, 0, ...]

                Do not include any additional text, explanations, or formatting other than a plain list.

                LLM response:
                "{llm_response}"."""
    return prompt

def parse_actual_labels(label_str):
    """Convert actual label strings to lists of integers."""
    try:
        return json.loads(label_str) if label_str else []
    except json.JSONDecodeError:
        logger.warning(f"Invalid format for labels: {label_str}")
        return []

def clean_extracted_list(response: str) -> str:
    # Remove unwanted text, keeping only numbers and commas
    cleaned_response = re.sub(r"[^\d,]", "", response)
    
    # Ensure commas between numbers (e.g., "0 1 0" to "0,1,0")
    cleaned_response = re.sub(r"(\d)(\d)", r"\1,\2", cleaned_response)
    
    # Wrap in brackets if not already a list
    if not (cleaned_response.startswith("[") and cleaned_response.endswith("]")):
        cleaned_response = f"[{cleaned_response}]"
    
    return cleaned_response

def save_progress(df, path):
    """Save the current progress to a CSV file."""
    df.to_csv(path, index=False)
    logger.info(f"Progress saved to {path}")
    
def run_metrics(df):
    correct_count = 0
    total_count = len(df)
    length_mismatch_count = 0
    element_mismatch_count = 0

    # Lists to store individual elements for precision, recall, and F1 calculations
    all_actual_tokens = []
    all_extracted_tokens = []

    for idx, row in df.iterrows():
        actual_labels = row['actual_labels']
        extracted_labels = row['extracted_labels']
        if isinstance(actual_labels, str):
            actual_labels = list(map(int, actual_labels.strip('[]').split(',')))
        if isinstance(extracted_labels, str):
            extracted_labels  = list(map(int, extracted_labels.strip('[]').split(',')))
        print(extracted_labels)

        # Ensure both are lists
        if not isinstance(actual_labels, list) or not isinstance(extracted_labels, list):
            logger.warning(f"Skipping non-list entries at index {idx}")
            continue

        # Check if lengths match
        if len(actual_labels) != len(extracted_labels):
            length_mismatch_count += 1
            continue

        # Compare element-by-element and collect tokens if they match in length
        match = all(a == e for a, e in zip(actual_labels, extracted_labels))
        if match:
            correct_count += 1
        else:
            element_mismatch_count += 1

        # Add tokens for token-level metric calculations
        all_actual_tokens.extend(actual_labels)
        all_extracted_tokens.extend(extracted_labels)

    # Calculate metrics for entry-level comparison
    accuracy = correct_count / total_count
    length_mismatch_rate = length_mismatch_count / total_count
    element_mismatch_rate = element_mismatch_count / total_count

    # Calculate token-level precision, recall, and F1 score
    precision = precision_score(all_actual_tokens, all_extracted_tokens, average="macro", zero_division=0)
    recall = recall_score(all_actual_tokens, all_extracted_tokens, average="macro", zero_division=0)
    f1 = f1_score(all_actual_tokens, all_extracted_tokens, average="macro", zero_division=0)

    # Log results
    logger.info(f"Accuracy (Entry-Level): {accuracy:.4f}")
    logger.info(f"Length Mismatch Rate: {length_mismatch_rate:.4f}")
    logger.info(f"Element Mismatch Rate: {element_mismatch_rate:.4f}")
    logger.info(f"Precision (Token-Level): {precision:.4f}")
    logger.info(f"Recall (Token-Level): {recall:.4f}")
    logger.info(f"F1 Score (Token-Level): {f1:.4f}")

    return {
        "Accuracy (Entry-Level)": [accuracy],
        "Length Mismatch Rate": [length_mismatch_rate],
        "Element Mismatch Rate": [element_mismatch_rate],
        "Precision (Token-Level)": [precision],
        "Recall (Token-Level)": [recall],
        "F1 Score (Token-Level)": [f1]
    }

def extract_and_evaluate_responses(args):
    results_file = INPUT_FILE_PATH

    df = pd.read_csv(results_file)
    # Parse actual labels as lists of integers
    correct_labels = df['actual_labels'].apply(parse_actual_labels).tolist()
    llm_responses = df['llm_responses'].tolist()
    extracted_labels = []

    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / args.task
        / f"evaluation_{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if 'extracted_labels' not in df.columns:
        df['extracted_labels'] = None

    for i, llm_response in enumerate(llm_responses):
        if pd.notna(df.at[i, 'extracted_labels']):
            continue

        try:
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are an expert in named entity recognition and label extraction."},
                    {"role": "user", "content": extraction_prompt_finer(llm_response)},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            
            extracted_list = model_response.choices[0].message.content.strip()  # type: ignore
            cleaned_response = clean_extracted_list(extracted_list)
            logger.info(f"Extracted labels for response {i + 1}: {cleaned_response}")
            extracted_tokens = json.loads(cleaned_response)
            
            extracted_labels.append(extracted_tokens)
            df.at[i, 'extracted_labels'] = extracted_tokens
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
            save_progress(df, evaluation_results_path)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append([])

    # Flatten the lists for metric calculations
    flat_correct_labels = [label for sublist in correct_labels for label in sublist]
    flat_extracted_labels = [label for sublist in extracted_labels for label in sublist]

    precision = precision_score(flat_correct_labels, flat_extracted_labels, average="macro", zero_division=0)
    recall = recall_score(flat_correct_labels, flat_extracted_labels, average="macro", zero_division=0)
    f1 = f1_score(flat_correct_labels, flat_extracted_labels, average="macro", zero_division=0)
    accuracy = accuracy_score(flat_correct_labels, flat_extracted_labels)

    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")

    eval_df = pd.DataFrame({
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "Accuracy": [accuracy]
    })
    eval_df.to_csv(Path(f"{str(evaluation_results_path)[:-4]}_statistics.csv"), index=False)

    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Evaluation completed. Results saved to {evaluation_results_path}")

    return df, evaluation_results_path

# Helper function for stop tokens
tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])

if __name__ == "__main__":
    class Args:
        task = "finer"
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        date = date.today().strftime('%d_%m_%Y')
        max_tokens = 512
        temperature = 0.0
        top_p = 0.9
        repetition_penalty = 1.0

    args = Args()
    df, evaluation_results_path = extract_and_evaluate_responses(args)
    df = pd.read_csv(evaluation_results_path)
    eval_df = pd.DataFrame(run_metrics(df))
    eval_df.to_csv(Path(f"{str(evaluation_results_path)[:-4]}_statistics.csv"), index=False)
