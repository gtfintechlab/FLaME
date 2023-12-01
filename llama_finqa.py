import pandas as pd
import torch
import json
from transformers import AutoTokenizer, pipeline

# Load the model
model = "/fintech_3/hf_models/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
# Function to preprocess data
def preprocess_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

test_dataset = preprocess_data("/fintech_3/vivian/test_finqa.json")
print(test_dataset[0])
pipeline_obj = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)



def build_instructions_zero_shot(dataset):
    zero_shot_prompt = "Given the financial data and expert analysis, please answer this question:"

    results = []
    for datum in dataset:
        context = " ".join(datum['pre_text']) + " " + " ".join(datum['post_text'])
        table_data = "\n".join([" | ".join(row) for row in datum['table_ori']])
        
        final_question = datum['qa']['question']
        expected_answer = datum['qa']['answer']


        full_prompt = f"{zero_shot_prompt}\nContext: {context}\nTable Data: {table_data}\nQuestion: {final_question}\nAnswer:"

        results.append({"prompt": full_prompt, "expected_answer": expected_answer})

    return results

test_dataset = build_instructions_zero_shot(test_dataset)




import re
def extract_numeric_value(text):
    # Regular expression to extract numeric values
    matches = re.findall(r"\b\d+\.?\d*\b", text)
    return matches[0] if matches else None


correct_predictions = 0
total_predictions = 0

for item in test_dataset:
    # Generate prediction
    prediction = pipeline_obj(item['prompt'])[0]['generated_text']
    generated_answer = prediction.split('Answer:')[-1].strip()
    print(generated_answer)
    # Append the generated answer to the item
    item['generated_answer'] = generated_answer

    # Expected answer as a string
    expected_answer = str(item['expected_answer']).strip()

    # Check if expected answer appears in the generated answer
    if expected_answer in generated_answer:

        correct_predictions += 1

    total_predictions += 1

# Calculate accuracy
results_df = pd.DataFrame(test_dataset)

results_df.to_csv('/fintech_3/vivian/llama_finqa_results.csv', index=False)

accuracy = correct_predictions / total_predictions
print("Accuracy:", accuracy)

# Optionally, print each item with its generated answer for review
for item in test_dataset:
    print(f"Prompt: {item['prompt']}")
    print(f"Generated Answer: {item['generated_answer']}")
    print(f"Expected Answer: {item['expected_answer']}\n")

