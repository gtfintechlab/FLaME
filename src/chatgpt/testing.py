from datasets import load_dataset
from huggingface_hub import login
HF_TOKEN = "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"
login(HF_TOKEN)
dataset = load_dataset("gtfintechlab/ECTSum", split = 'train', token=True)
print(dataset.split)