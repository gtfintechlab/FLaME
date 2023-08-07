from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_model(model_id, cache_dir):
    # Create the directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Download and cache the tokenizer and model
    print(f"Downloading and caching the model {model_id} to {cache_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
    print(f"Model {model_id} downloaded and cached successfully!")

def main():
    # Model IDs for 7, 13, and 70 billion parameter models
    model_ids = [
        "meta-llama/Llama-2-7b-hf", 
        "meta-llama/Llama-2-13b-hf", 
        "meta-llama/Llama-2-70b-hf"
    ]

    # Home directory
    home = os.path.expanduser("~")

    # Download each model to a separate folder
    for model_id in model_ids:
        model_name = model_id.split('/')[-1]
        cache_dir = os.path.join(home, f"models_hf/{model_name}")
        download_model(model_id, cache_dir)

if __name__ == "__main__":
    main()

