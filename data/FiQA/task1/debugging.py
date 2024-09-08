import json
from pathlib import Path
import pandas as pd
import os

def load_and_process_json(file_path):
    """Load a JSON file and process it into a DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    records = []
    for item in data.values():
        sentence = item['sentence']
        for info in item['info']:
            records.append({
                'sentence': sentence,
                'snippets': info['snippets'],
                'target': info['target'],
                'sentiment_score': info['sentiment_score'],
                'aspects': info['aspects']
            })

    return pd.DataFrame(records)

def main():
    # Correct paths based on your directory structure
    DATA_DIRECTORY = Path().cwd().parent / "task1"
    train_file = DATA_DIRECTORY / "train.json"
    test_file = DATA_DIRECTORY / "test.json"
    valid_file = DATA_DIRECTORY / "valid.json"
    try:
        # Load and process data
        train_data = load_and_process_json(train_file)
        test_data = load_and_process_json(test_file)
        valid_data = load_and_process_json(valid_file)

        # Inspect data types and contents
        print("Train Data Types:")
        print(train_data.dtypes)
        print("Train Data Sample:")
        print(train_data.head())

        print("\nTest Data Types:")
        print(test_data.dtypes)
        print("Test Data Sample:")
        print(test_data.head())

        print("\nValid Data Types:")
        print(valid_data.dtypes)
        print("Valid Data Sample:")
        print(valid_data.head())

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()