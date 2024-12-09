import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

# Tokenize the questions in the dataset
def tokenize_input(example):
    return llama_tokenizer(example['question'], return_tensors="pt", padding=True, truncation=True)

# Inference on a batch of inputs
def run_inference(example):
    input_ids = example['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    with torch.no_grad():
        outputs = llama_model.generate(input_ids, max_new_tokens=50)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

dataset = load_dataset("PatronusAI/financebench")
llama_model_name = "meta-llama/Llama-3.1-8B"
mistral_model_name = "mistralai/Mistral-7B-v0.1"
token = "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"

llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=token)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, token=token)
llama_model.eval()

# mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=token)
# mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_name, token=token)

# Apply tokenization to the dataset
# tokenized_dataset = dataset.map(tokenize_input, batched=True)

# Create or open a single file to store all generated outputs
output_file = "all_generated_answers.txt"

# Run inference and append the output to the file
with open(output_file, 'w') as file:
    for i in range(5):
        question = dataset['train'][i]['question']
        tokenized_input = tokenize_input({'question': question})
        answer = run_inference(tokenized_input)
        
        file.write(f"Question {i+1}: {question}\n")
        file.write(f"Generated Answer {i+1}: {answer}\n\n")  # Add space between questions

