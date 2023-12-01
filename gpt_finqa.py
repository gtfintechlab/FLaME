import pandas as pd
import json
import re
import openai

# Configure OpenAI API
openai.api_key = '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'

def preprocess_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

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


def extract_numeric_value(text):
    matches = re.findall(r"\b\d+\.?\d*\b", text)
    return matches[0] if matches else None


# Evaluation
correct_predictions = 0
total_predictions = 0
model = "gpt-3.5-turbo"  # Replace with the correct GPT-3.5 model name

for item in test_dataset:
    # Generate prediction using GPT-3.5
    response = openai.Completion.create(
        model=model,
        prompt=item['prompt'],
        temperature=0,
        max_tokens=150  
    )
    generated_answer = response.choices[0].text.strip().split('Answer:')[-1].strip()

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
results_df.to_csv('/fintech_3/vivian/finqa_gpt3_results.csv', index=False)

# Print results for review
for item in test_dataset:
    print(f"Prompt: {item['prompt']}")
    print(f"Generated Answer: {item['generated_answer']}")
    print(f"Expected Answer: {item['expected_answer']}\n")