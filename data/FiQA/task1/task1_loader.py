from datasets import load_dataset
import json
import os

ds = load_dataset("ChanceFocus/fiqa-sentiment-classification")

def format_data(split_name):
    split_data = ds[split_name] # type: ignore
    
    formatted_data = {}
    for entry in split_data:
        formatted_data[entry["_id"]] = { # type: ignore
            "sentence": entry["sentence"], # type: ignore
            "info": [
                {
                    "snippets": [entry["sentence"]],  # type: ignore
                    "target": entry["target"], # type: ignore
                    "sentiment_score": entry["score"], # type: ignore
                    "aspects": [entry["aspect"]] # type: ignore
                }
            ]
        }
    
    return formatted_data

def save_data():
    splits = ["train", "valid", "test"]
    
    for split in splits:
        formatted_data = format_data(split)
        file_path = os.path.join(".", f"{split}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=4)
        
        print(f"{split.capitalize()} data saved as: {file_path}")

if __name__ == "__main__":
    save_data()
