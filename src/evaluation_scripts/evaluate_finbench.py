import pandas as pd
import logging
from datetime import date
from pathlib import Path
from together import Together
import together

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

label_mapping = {
    "LOW RISK": 0,
    "HIGH RISK": 1
}

def extraction_prompt(llm_response: str):
    prompt = f"""Based on the following list of labels: ‘HIGH RISK’, ‘LOW RISK’, extract the most relevant label from the following response:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    return prompt

def map_label_to_number(label: str):
    """Map the extracted label to its corresponding numerical value after normalizing."""
    normalized_label = label.strip().upper()  # Normalize label to uppercase
    return label_mapping.get(normalized_label, -1)  # Return -1 if the label is not found

def extract_and_evaluate_responses(args):
    client = Together()
    
    results_file = (
        ROOT_DIR
        / "results"
        / 'finbench'
        / 'finbench_meta-llama'
        / "Meta-Llama-3.1-8B-Instruct-Turbo_03_10_2024.csv"
    )

    # Load the CSV file with the LLM responses
    df = pd.read_csv(results_file)
    extracted_labels = []
    correct_labels = df['y'].tolist()

    for i, llm_response in enumerate(df["llm_responses"]):
        try:
            model_response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",#args.model,
                messages=[{"role": "user", "content": extraction_prompt(llm_response)}],
                max_tokens=10,#args.max_tokens,
                temperature=0.0,#args.temperature,
                # top_k=args.top_k,
                top_p=0.9,#args.top_p,
                repetition_penalty=1.0,#args.repetition_penalty,
                stop=tokens("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")#args.model)
            )
            extracted_label = model_response.choices[0].message.content.strip() # type: ignore
            mapped_label = map_label_to_number(extracted_label)
            if (mapped_label == -1):
                print(f"Error processing response {i}: {llm_response}")
                logger.error(f"Error processing response {i}: {llm_response}")
            extracted_labels.append(mapped_label)
            logger.info(f"Processed {i + 1}/{len(df)} responses.")
        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append(-1)

    # Add extracted labels to the dataframe
    df['extracted_labels'] = extracted_labels

    # Evaluate the performance
    correct_predictions = sum(1 for x, y in zip(correct_labels, extracted_labels) if x == y)
    total_predictions = len(correct_labels)
    accuracy = correct_predictions / total_predictions

    # Save the evaluation results
    evaluation_results_path = (
        ROOT_DIR
        / "evaluation_results"
        / 'finbench'
        / f"evaluation_{'finbench'}_{'meta-llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(evaluation_results_path, index=False)

    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}. Results saved to {evaluation_results_path}")
    return df, accuracy

tokens_map = {"meta-llama/Llama-2-7b-chat-hf": ["<human>", "\n\n"]}
def tokens(model_name):
    return tokens_map.get(model_name, [])

if __name__ == "__main__":
    extract_and_evaluate_responses(None)
