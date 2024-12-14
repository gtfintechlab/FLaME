import re
import pandas as pd
from pathlib import Path
from superflue.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Function to extract numerical information using regex
def extract_numerical_info(text):
    # Regex to match numbers (integers, decimals, with/without signs, percentages)
    pattern = r"-?\d+\.?\d*%?"
    numbers = re.findall(pattern, text)
    return numbers


# Function to evaluate and extract numerical information
def extract_numbers_from_responses(csv_file_path, output_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        logger.info("CSV file loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return

    # List to store the extracted numerical information
    extracted_values = []

    # Loop through the DataFrame to extract numerical information from each response
    for i, response in enumerate(df["response"]):
        try:
            extracted_numbers = extract_numerical_info(response)
            extracted_values.append(extracted_numbers)
            logger.info(f"Processed response {i + 1}/{len(df)}: {extracted_numbers}")
        except Exception as e:
            logger.error(f"Error processing response {i + 1}: {e}")
            extracted_values.append(None)

    # Add extracted numerical information to the DataFrame
    df["extracted_numbers"] = extracted_values

    # Save the DataFrame with extracted values to a new CSV
    output_file = Path(output_file_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    input_csv = "/Users/yangyang/Desktop/SuperFLUE/src/superflue/results/evaluation_results/finqa/evaluation_finqa_meta-llama/Llama-2-7b-chat-hf_07_10_2024.csv"
    output_csv = "/Users/yangyang/Desktop/SuperFLUE/src/superflue/results/evaluation_results/finqa/evaluation_finqa_meta-llama/Llama-2-7b-chat-hf_07_10_2024_regex.csv"
    extract_numbers_from_responses(input_csv, output_csv)
