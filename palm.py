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

# Load datasets
train_dataset = preprocess_data("/fintech_3/vivian/train.json")
dev_dataset = preprocess_data("/fintech_3/vivian/dev.json")

# Combine and split datasets
combined_dataset = train_dataset + dev_dataset
train_size = int(len(combined_dataset) * 0.64)
valid_size = int(len(combined_dataset) * 0.16)
train_dataset, valid_dataset, test_dataset = combined_dataset[:train_size], combined_dataset[train_size:train_size+valid_size], combined_dataset[train_size+valid_size:]

# Zero-shot test instructions
def build_instructions_zero_shot(dataset):
    zero_shot_prompt = "In the context of this series of interconnected finance-related queries and the additional information provided by the pretext, table data, and post text from a company’s financial filings, please provide a response to the final question. This may require extracting information from the context and performing mathematical calculations. Please take into account the information provided in the preceding questions and their answers when formulating your response: "

    results = []
    for datum in dataset:
        context = " ".join(datum['pre_text']) + " " + " ".join(datum['post_text'])
        table_data = "\n".join([" | ".join(row) for row in datum['table_ori']])
        dialogues = "\n".join(datum['annotation']['dialogue_break'])
        final_question = datum['annotation']['dialogue_break'][-1]
        expected_answer = datum['annotation']['exe_ans_list'][-1]

        full_prompt = f"{zero_shot_prompt}\nContext: {context}\nTable Data: {table_data}\nDialogues: {dialogues}\nQuestion: {final_question}\nAnswer:"

        results.append({"prompt": full_prompt, "expected_answer": expected_answer})
    return results

test_dataset = build_instructions_zero_shot(test_dataset)

# Extract numeric value
def extract_numeric_value(text):
    matches = re.findall(r"\b\d+\.?\d*\b", text)
    return matches[0] if matches else None

# Evaluation
correct_predictions = 0
total_predictions = 0
models = [m for m in palm.list_models() if 'generateMessage' in m.supported_generation_methods]
model = models[0].name

from google.api_core import retry

@retry.Retry()
def retry_chat(**kwargs):
  return palm.chat(**kwargs)

@retry.Retry()
def retry_reply(self, arg):
  return self.reply(arg)

for item in test_dataset:
    # Generate prediction using PaLM
    chat = retry_chat(
        model= model, # Replace with your model name
        messages=item['prompt'],
        temperature=0
    )
    generated_answer = chat.last.split('Answer:')[-1].strip()
   
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
results_df.to_csv('/fintech_3/vivian/palm_results.csv', index=False)

# Print results for review
for item in test_dataset:
    print(f"Prompt: {item['prompt']}")
    print(f"Generated Answer: {item['generated_answer']}")
    print(f"Expected Answer: {item['expected_answer']}\n")
