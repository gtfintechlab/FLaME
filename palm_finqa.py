import pandas as pd
import json
import re
import google.generativeai as palm

# Configure PaLM API
palm.configure(api_key='AIzaSyCv82IIFOLkPQ8aPPzxhSLLiZVF504_FW4')

# Function to preprocess data
def preprocess_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Load dataset
test_dataset = preprocess_data("/fintech_3/vivian/test_finqa.json")

# Build zero-shot test instructions
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

# Extract numeric value function
def extract_numeric_value(text):
    matches = re.findall(r"\b\d+\.?\d*\b", text)
    return matches[0] if matches else None

models = [m for m in palm.list_models() if 'generateMessage' in m.supported_generation_methods]
model = models[0].name

from google.api_core import retry

@retry.Retry()
def retry_chat(**kwargs):
  return palm.chat(**kwargs)

@retry.Retry()
def retry_reply(self, arg):
  return self.reply(arg)

# Evaluation
correct_predictions = 0
total_predictions = 0

for item in test_dataset:
    # Generate prediction using PaLM
    chat = retry_chat(
        model= model, # Replace with your model name
        messages=item['prompt'],
        temperature=0
    )
    generated_answer = chat.last.split('Answer:')[-1].strip()
    print(generated_answer)
    item['generated_answer'] = generated_answer
    expected_answer = str(item['expected_answer']).strip()

    if expected_answer in generated_answer:
        correct_predictions += 1

    total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print("Accuracy:", accuracy)

# Save results
results_df = pd.DataFrame(test_dataset)
results_df.to_csv('/fintech_3/vivian/palm_finqa_results.csv', index=False)

# Print results for review
for item in test_dataset:
    print(f"Prompt: {item['prompt']}")
    print(f"Generated Answer: {item['generated_answer']}")
    print(f"Expected Answer: {item['expected_answer']}\n")
