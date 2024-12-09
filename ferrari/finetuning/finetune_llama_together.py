import os

from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

file_resp = client.files.upload(file="output.jsonl", check=True)

response = client.fine_tuning.create(
    training_file=file_resp.id,
    model="meta-llama/Meta-Llama-3.1-70B-Reference",
    lora=True,
)
