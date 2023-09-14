from datetime import date

TODAY = date.today()
SEEDS = (5768, 78516, 944601)

VALID_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
]

# Non-chat models should not be used in the experiment
INVALID_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
]

