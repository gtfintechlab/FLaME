from datasets import load_dataset
import json
import os

ds = load_dataset("LLukas22/fiqa")

train_questions = ds['train']['question'] # type: ignore
train_answers = ds['train']['answer'] # type: ignore

test_questions = ds['test']['question'] # type: ignore
test_answers = ds['test']['answer'] # type: ignore

train_data = [{"question": q, "answer": a} for q, a in zip(train_questions, train_answers)]
test_data = [{"question": q, "answer": a} for q, a in zip(test_questions, test_answers)]

train_path = os.path.join(".", "train.json")
test_path = os.path.join(".", "test.json")

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(test_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print(f"Train data saved as: {train_path}")
print(f"Test data saved as: {test_path}")
